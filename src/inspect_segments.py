"""Sanity-check processed data before trusting event-level metrics.

    python -m src.inspect_segments --processed_dir data/processed_v3   # v3 (from src.preprocess)
    python -m src.inspect_segments --processed_dir data/processed      # v2 legacy segments.npz

v3: prints, per subject, recorded hours, seizures, recorded preictal minutes per seizure,
interictal hours, files skipped and why. Windows are chronological by construction.

v2 legacy: event metrics assume windows in ``segments.npz`` are in recording order. This
reports how many ictal runs are directly preceded by a preictal run: close to 100% means
chronological order; a low value means the windows were shuffled or sorted by label.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .data import contiguous_runs, list_subjects, seizure_blocks


def label_runs(y):
    runs = []
    for v in y:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    return runs


def inspect_v3(processed_dir, subjects):
    print(f"{'subject':8s} {'hours':>6s} {'seizures':>8s} {'>=10min pre':>11s} {'inter h':>8s} "
          f"{'skipped':>7s}  preictal minutes per seizure")
    for s in subjects:
        r = json.load(open(os.path.join(processed_dir, s, "report.json")))
        print(f"{s:8s} {r['hours_total']:6.1f} {r['seizures']:8d} {r['seizures_with_10min_preictal']:11d} "
              f"{r['hours_interictal']:8.1f} {len(r['skipped']):7d}  {r['preictal_minutes_per_seizure']}")
        for sk in r["skipped"]:
            print(f"{'':10s}skipped {sk['file']}: {sk['reason']}")


def inspect_legacy(processed_dir, subjects, window_sec=5.0):
    print(f"{'subject':8s} {'windows':>8s} {'ictal runs':>10s} {'pre->ictal':>10s} "
          f"{'blocks':>6s} {'inter h':>8s}  keys")
    for s in subjects:
        seg = np.load(os.path.join(processed_dir, s, "segments.npz"), allow_pickle=True)
        y = np.array([str(v).lower() for v in seg["y"]])
        runs = label_runs(y)
        ictal = [i for i, (lab, _) in enumerate(runs) if lab == "ictal"]
        preceded = sum(1 for i in ictal if i > 0 and runs[i - 1][0] == "preictal")
        keep = np.isin(y, ("interictal", "preictal"))
        hard = (y[keep] == "preictal").astype(int)
        blocks = seizure_blocks(hard, contiguous_runs(np.where(keep)[0]))
        n_blocks = int(blocks.max() + 1) if (blocks >= 0).any() else 0
        frac = f"{preceded}/{len(ictal)}" if ictal else "n/a"
        print(f"{s:8s} {len(y):8d} {len(ictal):10d} {frac:>10s} {n_blocks:6d} "
              f"{np.sum(y == 'interictal') * window_sec / 3600:8.2f}  {seg.files}")
    print("\nIf no 'ictal' windows were stored, the pre->ictal check is unavailable; "
          "re-run preprocessing with src.preprocess to get timestamped data.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_dir", required=True)
    a = ap.parse_args(argv)
    subjects = list_subjects(a.processed_dir)
    v3 = [s for s in subjects if os.path.exists(os.path.join(a.processed_dir, s, "meta.npz"))]
    if v3:
        inspect_v3(a.processed_dir, v3)
    legacy = [s for s in subjects if s not in v3]
    if legacy:
        inspect_legacy(a.processed_dir, legacy)


if __name__ == "__main__":
    main()
