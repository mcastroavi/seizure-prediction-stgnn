"""Sanity-check processed segments before trusting event-level metrics.

    python -m src.inspect_segments --processed_dir data/processed

Event metrics (alarm smoothing, lead time, false alarms per hour) assume the windows in
each ``segments.npz`` are stored in recording order. This prints, per subject:

* the keys stored in the file (a recording id or start time makes contiguity exact),
* how many ictal runs are directly preceded by a preictal run — close to 100% means the
  windows are in chronological order; a low value means they were shuffled or sorted by
  label, and the event metrics will not be meaningful,
* seizure blocks found and hours of interictal data.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .data import contiguous_runs, list_subjects, seizure_blocks


def label_runs(y):
    """Run-length encode a label sequence -> list of (label, length)."""
    runs = []
    for v in y:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    return runs


def inspect(processed_dir, window_sec=5.0):
    print(f"{'subject':8s} {'windows':>8s} {'ictal runs':>10s} {'pre->ictal':>10s} "
          f"{'blocks':>6s} {'inter h':>8s}  keys")
    for s in list_subjects(processed_dir):
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
    print("\nLabels present overall are those found in 'y'. If no 'ictal' windows were stored, "
          "the pre->ictal check is unavailable; confirm ordering from your preprocessing script.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_dir", required=True)
    a = ap.parse_args(argv)
    inspect(a.processed_dir)


if __name__ == "__main__":
    main()
