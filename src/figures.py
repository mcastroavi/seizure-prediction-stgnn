"""Figures for the README and for walking through results.

    python -m src.figures --results_dir results/lopo

Writes to ``<results_dir>/figures/``:

* ``risk_timeline_<subject>.png`` — risk score over the whole recording of one test
  subject, with preictal periods, seizure onsets, the threshold and the alarms raised.
* ``per_subject.png``  — seizure sensitivity and false alarms per hour for each test subject.
* ``operating_curve.png`` — pooled seizure sensitivity vs false alarms per hour as the
  threshold varies, next to a random predictor (descriptive only: thresholds used for the
  reported results are chosen on validation subjects).
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .evaluate import WINDOW_SEC, load_predictions  # noqa: E402
from .metrics import THRESHOLD_GRID, event_metrics, raise_alarms, summarize_events  # noqa: E402

# Reference palette (light mode) — roles, not decoration
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"   # model risk / model curve
SERIES_2 = "#eb6834"   # alarms / preictal shading / random predictor

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2,
    "ytick.color": TEXT_2, "text.color": TEXT, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold",
})


def _gather(results_dir):
    folds = sorted(d for d in os.listdir(results_dir)
                   if os.path.exists(os.path.join(results_dir, d, "predictions.npz")))
    return {f: load_predictions(os.path.join(results_dir, f, "predictions.npz")) for f in folds}


def risk_timeline(v, subject, tau, out_path, k=3, n=5, refractory=360):
    probs, hard, block, run = v["probs"], v["hard"], v["block"], v["run"]
    t_h = (v["t"] - v["t"][0]) / 3600 if "t" in v else np.arange(len(probs)) * WINDOW_SEC / 3600
    alarms = raise_alarms(probs, run, tau, k, n, refractory)

    # Break the line wherever the recording is not contiguous
    y = probs.astype(float).copy()
    x = t_h.astype(float).copy()
    brk = np.where(np.diff(run) != 0)[0] + 1
    x = np.insert(x, brk, np.nan)
    y = np.insert(y, brk, np.nan)

    fig, ax = plt.subplots(figsize=(11, 3.4))
    for b in np.unique(block[block >= 0]):
        idx = np.where(block == b)[0]
        ax.axvspan(t_h[idx.min()], t_h[idx.max()] + WINDOW_SEC / 3600, color=SERIES_2, alpha=0.15, lw=0)
        onset = t_h[idx.max()] + WINDOW_SEC / 3600
        ax.axvline(onset, color=TEXT_2, lw=1, ls="--")
    ax.plot(x, y, color=SERIES_1, lw=0.8, label="Risk score")
    ax.axhline(tau, color=TEXT_2, lw=1, ls=":")
    ax.text(t_h[-1], tau, f" τ = {tau:.2f}", va="center", ha="left", color=TEXT_2, fontsize=9)
    ax.scatter(t_h[alarms], np.full(alarms.sum(), 1.04), marker="v", s=40, color=SERIES_2,
               edgecolor=SURFACE, linewidth=1, label="Alarm", zorder=3, clip_on=False)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(t_h[0], t_h[-1])
    ax.set_xlabel("Hours from start of recording")
    ax.set_ylabel("Risk")
    ax.set_title(f"{subject}: risk over the full recording (held-out subject)", loc="left")
    pre = plt.Rectangle((0, 0), 1, 1, color=SERIES_2, alpha=0.15)
    on = plt.Line2D([], [], color=TEXT_2, ls="--", lw=1)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h + [pre, on], lab + ["Preictal", "Seizure onset"], loc="upper left",
              ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0, -0.22))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def per_subject(rows, out_path):
    rows = sorted(rows, key=lambda r: r["subject"])
    subs = [r["subject"] for r in rows]
    sens = [r["predicted"] / r["seizures"] if r["seizures"] else np.nan for r in rows]
    fpr = [r["fpr_per_h"] for r in rows]
    x = np.arange(len(subs))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
    a1.bar(x, sens, color=SERIES_1, width=0.6)
    a1.set_ylim(0, 1.18)
    a1.set_ylabel("Seizure sensitivity")
    a1.set_title("Per held-out subject", loc="left")
    for xi, r in zip(x, rows):
        a1.text(xi, (r["predicted"] / r["seizures"] if r["seizures"] else 0) + 0.03,
                f"{r['predicted']}/{r['seizures']}", ha="center", fontsize=8, color=TEXT_2)
    a2.bar(x, fpr, color=SERIES_1, width=0.6)
    a2.set_ylabel("False alarms / hour")
    a2.set_ylim(0, max(0.5, np.nanmax(fpr) * 1.15) if len(fpr) else 1)
    a2.set_xticks(x, subs, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def operating_curve(preds_by_fold, out_path, k=3, n=5, refractory=360, sop_min=30, min_win=120,
                    model_name="ST-GNN"):
    grid = np.unique(np.concatenate([np.round(np.arange(0.05, 0.96, 0.025), 3), THRESHOLD_GRID]))
    sens, fpr = [], []
    tests = [v for p in preds_by_fold.values() for v in p["test"].values()]
    for t in grid:
        per = [event_metrics(raise_alarms(v["probs"], v["run"], t, k, n, refractory),
                             v["hard"], v["block"], WINDOW_SEC, min_win) for v in tests]
        s = summarize_events(per, sop_min)
        sens.append(s["sensitivity"]); fpr.append(s["fpr_per_hour"])
    fpr, sens = np.array(fpr), np.array(sens)
    xs = np.linspace(0, max(1.0, np.nanmax(fpr) * 1.05), 200)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(fpr, sens, color=SERIES_1, lw=2, marker="o", ms=4, label=model_name)
    ax.plot(xs, 1 - np.exp(-xs * sop_min / 60), color=SERIES_2, lw=2, ls="--", label="Random predictor")
    ax.set_xlabel("False alarms per hour")
    ax.set_ylabel("Seizure sensitivity")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, xs[-1])
    ax.set_title("Sensitivity vs. false-alarm rate (all test subjects)", loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--subject", default=None, help="subject for the risk timeline (default: most seizures)")
    a = ap.parse_args(argv)

    res = json.load(open(os.path.join(a.results_dir, "results.json")))
    rows, summ = res["subjects"], res["summary"]
    preds = _gather(a.results_dir)
    out = os.path.join(a.results_dir, "figures")
    os.makedirs(out, exist_ok=True)
    refractory = int(round(summ.get("refractory_min", 30) * 60 / WINDOW_SEC))
    k, n = summ.get("alarm_k", 3), summ.get("alarm_n", 5)
    min_win = int(np.ceil(summ.get("min_preictal_min", 10) * 60 / WINDOW_SEC))

    pick = a.subject or max(rows, key=lambda r: (r["seizures"], r["predicted"]))["subject"]
    row = next(r for r in rows if r["subject"] == pick)
    v = preds[row["fold"]]["test"][pick]
    risk_timeline(v, pick, row["tau"], os.path.join(out, f"risk_timeline_{pick}.png"), k, n, refractory)
    per_subject(rows, os.path.join(out, "per_subject.png"))
    fold_info = json.load(open(os.path.join(a.results_dir, row["fold"], "fold.json")))
    model_name = "Logistic regression (baseline)" if fold_info.get("model") == "logistic_regression" else "ST-GNN"
    operating_curve(preds, os.path.join(out, "operating_curve.png"), k, n, refractory,
                    sop_min=summ.get("sop_min", 30), min_win=min_win, model_name=model_name)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
