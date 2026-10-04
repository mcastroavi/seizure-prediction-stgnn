"""Extra per-window inputs for the context model: time of day and seizure history.

Both use only information available at prediction time: the clock, and seizures that have
already ended.

Label-leakage guard
-------------------
Interictal windows are, by construction, at least ``buffer`` (60 min) after any seizure.
Windows less than 60 min after a seizure are therefore either preictal (a seizure in a
cluster) or excluded from scoring. A raw "time since last seizure" feature would let a model
learn "< 60 min ⇒ preictal" from the labelling rule rather than from the EEG, and its alarms
in that period would never be counted as false (excluded windows are not scored).
The feature is therefore floored at the buffer: every value below 60 min looks the same.
"""

from __future__ import annotations

import numpy as np

DAY = 86400.0


def time_of_day(t_start: np.ndarray) -> np.ndarray:
    """(N, 2): sin and cos of the clock time (t_start in seconds from the first day's midnight)."""
    ang = 2 * np.pi * (np.asarray(t_start, float) % DAY) / DAY
    return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


def seizure_history(t_start: np.ndarray, seizures: np.ndarray, floor_h: float,
                    cap_h: float = 72.0) -> np.ndarray:
    """(N, 2): [scaled log time since the last seizure ended, has-a-previous-seizure flag].

    Only seizures that ended at or before the window's start count. Time is clipped to
    [floor_h, cap_h] hours and mapped to [0, 1]; with no previous seizure it is 1 (= cap).
    """
    t = np.asarray(t_start, float)
    sz = np.asarray(seizures, float).reshape(-1, 2)
    out = np.zeros((len(t), 2), dtype=np.float32)
    if len(sz) == 0:
        out[:, 0] = 1.0
        return out
    ends = np.sort(sz[:, 1])
    k = np.searchsorted(ends, t, side="right")          # number of seizures ended by t
    has_prev = k > 0
    dt_h = np.full(len(t), cap_h)
    dt_h[has_prev] = (t[has_prev] - ends[k[has_prev] - 1]) / 3600.0
    dt_h = np.clip(dt_h, floor_h, cap_h)
    lo, hi = np.log1p(floor_h), np.log1p(cap_h)
    out[:, 0] = (np.log1p(dt_h) - lo) / (hi - lo)
    out[:, 1] = has_prev
    return out


def extra_features(part, kinds: list[str], cap_h: float = 72.0) -> np.ndarray | None:
    """Concatenate the requested extras for one SubjectData part (v3 format only)."""
    if not kinds:
        return None
    if part.t_start is None:
        raise RuntimeError("time/history features need v3 data (src.preprocess) with timestamps")
    cols = []
    if "time" in kinds:
        cols.append(time_of_day(part.t_start))
    if "history" in kinds:
        floor_h = part.meta["params"]["buffer_min"] / 60.0
        cols.append(seizure_history(part.t_start, np.array(part.meta["seizures"]), floor_h, cap_h))
    return np.concatenate(cols, axis=1)
