"""Score saved predictions at the seizure level.

For every fold the alarm threshold is chosen on that fold's **validation** subjects only,
then applied unchanged to the held-out test data.

    python -m src.evaluate --results_dir results/lopo
    python -m src.evaluate --results_dir results/lopo --threshold_mode fpr --target_fpr 0.15

Writes ``results.json``, ``per_subject.csv``, ``per_seizure.csv`` and a ready-to-paste
``results.md`` into ``results_dir``. Needs only numpy / scipy / scikit-learn.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from .metrics import (event_metrics, raise_alarms, select_threshold, summarize_events,
                      window_metrics)

WINDOW_SEC = 5.0


def load_predictions(path: str) -> dict:
    """{split: {subject: {probs, hard, block, run}}}"""
    out: dict = defaultdict(lambda: defaultdict(dict))
    with np.load(path) as z:
        for key in z.files:
            split, subj, field = key.split("__")
            out[split][subj][field] = z[key]
    return out


def evaluate(results_dir, threshold_mode="fpr", target_fpr=0.5, k=3, n=5,
             refractory_min=30.0, sop_min=30.0):
    refractory = int(round(refractory_min * 60 / WINDOW_SEC))
    folds = sorted(d for d in os.listdir(results_dir)
                   if os.path.exists(os.path.join(results_dir, d, "predictions.npz")))
    if not folds:
        raise SystemExit(f"No */predictions.npz found under {results_dir}")

    subject_rows, seizure_rows, events = [], [], []
    all_probs, all_hard = [], []
    for fold in folds:
        preds = load_predictions(os.path.join(results_dir, fold, "predictions.npz"))
        val_sets = list(preds["val"].values())
        tau = select_threshold(val_sets, threshold_mode, target_fpr, k, n, refractory, WINDOW_SEC)

        for subj, v in preds["test"].items():
            alarms = raise_alarms(v["probs"], v["run"], tau, k, n, refractory)
            ev = event_metrics(alarms, v["hard"], v["block"], WINDOW_SEC)
            wm = window_metrics(v["probs"], v["hard"], tau)
            events.append(ev)
            all_probs.append(v["probs"]); all_hard.append(v["hard"])
            subject_rows.append({
                "fold": fold, "subject": subj, "tau": tau,
                "auc": wm["auc"], "window_sens": wm["sensitivity"], "window_spec": wm["specificity"],
                "seizures": ev["n_seizures"], "predicted": ev["n_predicted"],
                "false_alarms": ev["false_alarms"], "interictal_h": ev["interictal_hours"],
                "fpr_per_h": ev["false_alarms"] / ev["interictal_hours"] if ev["interictal_hours"] else float("nan"),
            })
            for s in ev["seizures"]:
                seizure_rows.append({"fold": fold, "subject": subj, **s})

    summary = summarize_events(events, sop_min)
    probs, hard = np.concatenate(all_probs), np.concatenate(all_hard)
    aucs = [r["auc"] for r in subject_rows if not np.isnan(r["auc"])]
    summary.update({
        "protocol_dir": results_dir,
        "threshold_mode": threshold_mode,
        "target_fpr": target_fpr if threshold_mode == "fpr" else None,
        "alarm_rule": f"{k}-of-{n} windows, refractory {refractory_min:g} min",
        "sop_min": sop_min,
        "n_test_subjects": len(subject_rows),
        "window_auc_pooled": float(window_metrics(probs, hard, 0.5)["auc"]),
        "window_auc_mean_per_subject": float(np.mean(aucs)) if aucs else float("nan"),
        "window_auc_std_per_subject": float(np.std(aucs)) if aucs else float("nan"),
    })

    _write(results_dir, summary, subject_rows, seizure_rows)
    return summary


def _fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _write(results_dir, summary, subject_rows, seizure_rows):
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "subjects": subject_rows}, f, indent=2)
    for name, rows in (("per_subject.csv", subject_rows), ("per_seizure.csv", seizure_rows)):
        if rows:
            with open(os.path.join(results_dir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    s = summary
    md = [
        f"### Results — `{os.path.basename(os.path.normpath(results_dir))}`",
        "",
        f"Threshold chosen on validation subjects only ({s['threshold_mode']}"
        + (f", target ≤ {s['target_fpr']} FA/h" if s["target_fpr"] is not None else "") + "). "
        f"Alarm rule: {s['alarm_rule']}. SOP: {s['sop_min']:g} min.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Test subjects | {s['n_test_subjects']} |",
        f"| Window ROC-AUC (mean ± sd per subject) | {_fmt(s['window_auc_mean_per_subject'])} ± {_fmt(s['window_auc_std_per_subject'])} |",
        f"| Seizure sensitivity | {s['predicted']}/{s['seizures']} = {_fmt(s['sensitivity'])} |",
        f"| False alarms per hour | {_fmt(s['fpr_per_hour'], 2)} ({s['false_alarms']} in {s['interictal_hours']:.1f} h) |",
        f"| Lead time, mean / median (predicted seizures) | {_fmt(s['lead_time_mean_min'], 1)} / {_fmt(s['lead_time_median_min'], 1)} min |",
        f"| Random-predictor sensitivity at same FA/h | {_fmt(s['random_predictor_sensitivity'])} |",
        f"| p-value vs. chance | {'< 0.0001' if s['p_value_vs_chance'] < 1e-4 else _fmt(s['p_value_vs_chance'], 4)} |",
        "",
        "| Subject | τ | AUC | Seizures predicted | False alarms / h |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in subject_rows:
        md.append(f"| {r['subject']} | {r['tau']:.2f} | {_fmt(r['auc'])} | "
                  f"{r['predicted']}/{r['seizures']} | {_fmt(r['fpr_per_h'], 2)} |")
    with open(os.path.join(results_dir, "results.md"), "w") as f:
        f.write("\n".join(md) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--threshold_mode", choices=["fpr", "f1"], default="fpr")
    ap.add_argument("--target_fpr", type=float, default=0.5, help="false alarms per hour")
    ap.add_argument("--k", type=int, default=3, help="windows above threshold ...")
    ap.add_argument("--n", type=int, default=5, help="... out of the last n to raise an alarm")
    ap.add_argument("--refractory_min", type=float, default=30.0)
    ap.add_argument("--sop_min", type=float, default=30.0)
    a = ap.parse_args(argv)
    s = evaluate(a.results_dir, a.threshold_mode, a.target_fpr, a.k, a.n, a.refractory_min, a.sop_min)
    print(open(os.path.join(a.results_dir, "results.md")).read())
    if not s["beats_chance_at_0.05"]:
        print("NOTE: sensitivity is not significantly better than a random predictor "
              "with the same false-alarm rate (p >= 0.05).")


if __name__ == "__main__":
    main()
