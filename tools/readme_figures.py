"""Build the README figures in two steps, so they can be regenerated without the dataset.

    # 1. on the machine with results/ (needs the evaluated runs):
    python tools/readme_figures.py export --results_dir results --out docs/figures/data.json
    # 2. anywhere (numpy + matplotlib only):
    python tools/readme_figures.py render --data docs/figures/data.json --out_dir docs/figures

``export`` collects every run's summary, the per-subject results of the best model, pooled
sensitivity vs false-alarm curves, and per-minute risk timelines for two held-out patients.
``render`` draws them in the style of ``src.figures``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BEST = "context_stgnn_bands_lopo"
TIMELINE_SUBJECTS = ("chb20", "chb15")
CURVES = [("baseline_lopo", "Logistic regression"), ("stgnn_lopo", "ST-GNN (5-s windows)"),
          (BEST, "ST-GNN + bands + 5-min context")]
WINDOW_SEC, K, N, REFRACTORY, MIN_WIN = 5.0, 3, 5, 360, 120

ABLATION = [  # (label, run)
    ("Logistic regression", "baseline_lopo"),
    ("+ band features", "baseline_bands_lopo"),
    ("ST-GNN", "stgnn_lopo"),
    ("ST-GNN + bands", "stgnn_bands_lopo"),
    ("Context GRU, hand-crafted features", "context_feat_lopo"),
    ("Context GRU on ST-GNN", "context_stgnn_lopo"),
    ("Context GRU on ST-GNN + bands", BEST),
    ("  + augmentation", "context_stgnn_bands_aug_lopo"),
    ("  + time of day & seizure history", "context_stgnn_bands_timehist_lopo"),
]
ABLATION_LEAD = [(lab, f"{run}/lead4h") for lab, run in ABLATION]
PERSONALIZATION = [
    ("Patient-specific, from scratch", "context_stgnn_bands_chrono"),
    ("General model, threshold calibrated", "personalized/general"),
    ("General + fine-tune context", "personalized/finetune_context"),
    ("General + fine-tune all", "personalized/finetune_full"),
]


# ── export ───────────────────────────────────────────────────────────────────

def export(results_dir, out):
    from src.evaluate import load_predictions
    from src.metrics import THRESHOLD_GRID, event_metrics, raise_alarms, summarize_events

    runs = {}
    for _, run in ABLATION + ABLATION_LEAD + PERSONALIZATION:
        p = os.path.join(results_dir, run, "results.json")
        if os.path.exists(p):
            runs[run] = json.load(open(p))["summary"]
    best = json.load(open(os.path.join(results_dir, BEST, "results.json")))

    def preds(run):
        d = os.path.join(results_dir, run)
        return {f: load_predictions(os.path.join(d, f, "predictions.npz"))
                for f in sorted(os.listdir(d)) if os.path.exists(os.path.join(d, f, "predictions.npz"))}

    curves = {}
    grid = np.unique(np.concatenate([np.round(np.arange(0.05, 0.96, 0.025), 3), THRESHOLD_GRID]))
    for run, label in CURVES:
        tests = [v for p in preds(run).values() for v in p["test"].values()]
        pts = []
        for t in grid:
            s = summarize_events([event_metrics(raise_alarms(v["probs"], v["run"], t, K, N, REFRACTORY),
                                                v["hard"], v["block"], WINDOW_SEC, MIN_WIN) for v in tests])
            pts.append((round(s["fpr_per_hour"], 4), round(s["sensitivity"], 4)))
        curves[run] = {"label": label, "points": sorted(set(pts))}

    timelines = {}
    p_best = preds(BEST)
    for subj in TIMELINE_SUBJECTS:
        row = next(r for r in best["subjects"] if r["subject"] == subj)
        v = p_best[row["fold"]]["test"][subj]
        tau = row["tau"]
        alarms = raise_alarms(v["probs"], v["run"], tau, K, N, REFRACTORY)
        t0 = v["t"][0]
        h = (v["t"] - t0) / 3600
        # per-minute maximum risk, never merging across recording gaps
        key = v["run"] * 100000 + ((v["t"] - t0) // 60).astype(int)
        _, start = np.unique(key, return_index=True)
        start = np.sort(start)
        bins = np.split(np.arange(len(h)), start[1:])
        pre = []
        for b in np.unique(v["block"][v["block"] >= 0]):
            idx = np.where(v["block"] == b)[0]
            pre.append([round(float(h[idx.min()]), 4), round(float(h[idx.max()] + WINDOW_SEC / 3600), 4)])
        timelines[subj] = {
            "tau": tau, "predicted": row["predicted"], "seizures": row["seizures"],
            "fpr_per_h": row["fpr_per_h"], "auc": row["auc"],
            "h": [round(float(h[b[0]]), 4) for b in bins],
            "risk": [round(float(v["probs"][b].max()), 3) for b in bins],
            "run": [int(v["run"][b[0]]) for b in bins],
            "preictal": pre, "alarms": [round(float(x), 4) for x in h[alarms]],
            "end_h": round(float(h[-1]), 3),
        }

    data = {"runs": runs, "best_subjects": best["subjects"], "curves": curves, "timelines": timelines,
            "settings": {"k": K, "n": N, "refractory_min": REFRACTORY * WINDOW_SEC / 60,
                         "min_preictal_min": MIN_WIN * WINDOW_SEC / 60}}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")


# ── render ───────────────────────────────────────────────────────────────────

def render(data_path, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    })
    d = json.load(open(data_path))
    os.makedirs(out_dir, exist_ok=True)

    def two_panel(rows, title, fname, note):
        rows = [(lab, d["runs"][r]) for lab, r in rows if r in d["runs"]]
        y = np.arange(len(rows))[::-1]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.55 * len(rows) + 1.6), sharey=True,
                                     gridspec_kw={"width_ratios": [1.6, 1]})
        sens = [s["sensitivity"] for _, s in rows]
        rnd = [s["random_predictor_sensitivity"] for _, s in rows]
        a1.barh(y, sens, height=0.55, color=BLUE, label="Model")
        a1.scatter(rnd, y, marker="|", s=260, linewidths=2.5, color=ORANGE, zorder=3,
                   label="Random predictor, same false-alarm rate")
        for yi, (_, s) in zip(y, rows):
            a1.text(max(s["sensitivity"], s["random_predictor_sensitivity"]) + 0.012, yi,
                    f"{s['predicted']}/{s['seizures']}", va="center",
                    fontsize=9, color=TEXT_2)
        a1.set_xlim(0, max(sens) * 1.25)
        a1.set_xlabel("Seizure sensitivity")
        a1.set_yticks(y, [lab for lab, _ in rows])
        a1.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                  borderaxespad=0.2)
        fa = [s["fpr_per_hour"] for _, s in rows]
        a2.barh(y, fa, height=0.55, color=BLUE)
        for yi, f in zip(y, fa):
            a2.text(f + 0.01, yi, f"{f:.2f}", va="center", fontsize=9, color=TEXT_2)
        a2.set_xlim(0, max(fa) * 1.25)
        a2.set_xlabel("False alarms per hour")
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
        fig.text(0.01, 0.005, note, fontsize=8.5, color=TEXT_2)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        fig.savefig(os.path.join(out_dir, fname), dpi=130)
        plt.close(fig)

    two_panel(ABLATION, "All seizures — leave-one-patient-out (24 unseen patients)", "ablation_lopo.png",
              "151 seizures with ≥10 min of recorded preictal data. Alarm threshold chosen on validation patients only.")
    two_panel(ABLATION_LEAD, "Lead seizures only — leave-one-patient-out (24 unseen patients)",
              "ablation_lead.png",
              "65 seizures starting ≥4 h after the previous one; clustered seizures not scored. "
              "Threshold chosen on validation patients' lead seizures.")
    two_panel(PERSONALIZATION, "Personalization — last 30% of each patient's recording (13 patients)",
              "personalization.png",
              "31 test seizures; all variants use the same test data and choose the threshold on the patient's validation part.")

    # Operating curve
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    colors = {"baseline_lopo": "#b5b4ae", "stgnn_lopo": ORANGE, BEST: BLUE}
    xmax = 0
    for run, c in d["curves"].items():
        pts = np.array(c["points"])
        pts = pts[np.argsort(pts[:, 0])]
        ax.plot(pts[:, 0], pts[:, 1], color=colors.get(run, TEXT_2), lw=2, label=c["label"])
        xmax = max(xmax, pts[:, 0].max())
    xmax = min(xmax, 3.0)
    xs = np.linspace(0, xmax, 200)
    ax.plot(xs, 1 - np.exp(-xs * 0.5), color=TEXT_2, lw=1.5, ls="--", label="Random predictor")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False alarms per hour")
    ax.set_ylabel("Seizure sensitivity")
    ax.set_title("Sensitivity vs. false alarms, 24 unseen patients", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "operating_curve.png"), dpi=130)
    plt.close(fig)

    # Per-subject results of the best model
    rows = sorted(d["best_subjects"], key=lambda r: r["subject"])
    x = np.arange(len(rows))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    a1.bar(x, [r["predicted"] / r["seizures"] if r["seizures"] else 0 for r in rows], width=0.6, color=BLUE)
    for xi, r in zip(x, rows):
        a1.text(xi, (r["predicted"] / r["seizures"] if r["seizures"] else 0) + 0.03,
                f"{r['predicted']}/{r['seizures']}", ha="center", fontsize=8, color=TEXT_2)
    a1.set_ylim(0, 1.18)
    a1.set_ylabel("Seizure sensitivity")
    a1.set_title("Best model, per held-out patient", loc="left")
    a2.bar(x, [r["fpr_per_h"] for r in rows], width=0.6, color=BLUE)
    a2.set_ylabel("False alarms / hour")
    a2.set_xticks(x, [r["subject"] for r in rows], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_subject.png"), dpi=130)
    plt.close(fig)

    # Risk timelines
    for subj, tl in d["timelines"].items():
        h, risk, run = np.array(tl["h"]), np.array(tl["risk"]), np.array(tl["run"])
        brk = np.where(np.diff(run) != 0)[0] + 1
        hx, ry = np.insert(h, brk, np.nan), np.insert(risk, brk, np.nan)
        fig, ax = plt.subplots(figsize=(11, 3.4))
        for a, b in tl["preictal"]:
            ax.axvspan(a, b, color=ORANGE, alpha=0.15, lw=0)
            ax.axvline(b, color=TEXT_2, lw=1, ls="--")
        ax.plot(hx, ry, color=BLUE, lw=0.8)
        ax.axhline(tl["tau"], color=TEXT_2, lw=1, ls=":")
        ax.text(tl["end_h"], tl["tau"], f" τ = {tl['tau']:.2f}", va="center", ha="left", color=TEXT_2, fontsize=9)
        ax.scatter(tl["alarms"], np.full(len(tl["alarms"]), 1.04), marker="v", s=40, color=ORANGE,
                   edgecolor=SURFACE, linewidth=1, zorder=3, clip_on=False)
        ax.set_ylim(0, 1.08)
        ax.set_xlim(0, tl["end_h"])
        ax.set_xlabel("Hours from start of test recording")
        ax.set_ylabel("Risk (per-minute max)")
        ax.set_title(f"{subj} (never seen in training): {tl['predicted']}/{tl['seizures']} seizures predicted, "
                     f"{tl['fpr_per_h']:.2f} false alarms/h", loc="left")
        handles = [plt.Line2D([], [], color=BLUE, lw=1.5),
                   plt.Line2D([], [], marker="v", color=ORANGE, ls="", markersize=7),
                   plt.Rectangle((0, 0), 1, 1, color=ORANGE, alpha=0.15),
                   plt.Line2D([], [], color=TEXT_2, ls="--", lw=1)]
        ax.legend(handles, ["Risk", "Alarm", "Preictal (30 min)", "Seizure onset"], loc="upper left",
                  ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0, -0.22))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"risk_timeline_{subj}.png"), dpi=130)
        plt.close(fig)
    print(f"figures written to {out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--results_dir", default="results")
    e.add_argument("--out", default="docs/figures/data.json")
    r = sub.add_parser("render")
    r.add_argument("--data", default="docs/figures/data.json")
    r.add_argument("--out_dir", default="docs/figures")
    a = ap.parse_args(argv)
    export(a.results_dir, a.out) if a.cmd == "export" else render(a.data, a.out_dir)


if __name__ == "__main__":
    main()
