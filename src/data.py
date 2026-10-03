"""Loading per-subject CHB-MIT windows with the structure needed for honest evaluation.

The original notebooks concatenated every window from every subject and shuffled them,
which threw away two things evaluation needs:

* which subject a window came from (needed for patient-held-out splits), and
* where the window sits in time (needed for alarm smoothing, per-seizure sensitivity,
  lead time and false alarms per hour).

``load_subject`` keeps both. It expects the same ``processed/<subject>/segments.npz``
files the notebooks used (keys ``X`` and ``y``), and reuses ``A_plv.npy`` when present.

Chronology assumption
---------------------
Windows inside ``segments.npz`` are assumed to be in recording order. Run
``python -m src.inspect_segments`` to check this before trusting event-level metrics.
If the file also stores a recording id (e.g. ``file``) or start time (e.g. ``t_start``),
those are used to break contiguity between recordings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import hilbert

KEEP = ("interictal", "preictal")
RECORDING_KEYS = ("file", "files", "file_id", "file_idx", "rec", "record", "recording")
TIME_KEYS = ("t_start", "start", "t", "time", "start_sec")


@dataclass
class SubjectData:
    subject: str
    X: np.ndarray            # (N, C, T) float32 EEG windows
    plv: np.ndarray          # (N, C, C) float32 PLV matrices
    hard: np.ndarray         # (N,) int64   0 = interictal, 1 = preictal
    risk: np.ndarray         # (N,) float32 soft risk target
    block: np.ndarray        # (N,) int64   seizure-block id for preictal windows, -1 otherwise
    run: np.ndarray          # (N,) int64   id of the contiguous stretch of recording
    orig_idx: np.ndarray     # (N,) int64   index into the un-filtered segments.npz
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.hard)

    @property
    def n_seizures(self) -> int:
        return int(self.block.max() + 1) if (self.block >= 0).any() else 0

    def subset(self, idx: np.ndarray) -> "SubjectData":
        """Return a view restricted to ``idx`` (kept in the given order)."""
        idx = np.asarray(idx, dtype=np.int64)
        return SubjectData(
            subject=self.subject, X=self.X[idx], plv=self.plv[idx], hard=self.hard[idx],
            risk=self.risk[idx], block=self.block[idx], run=self.run[idx],
            orig_idx=self.orig_idx[idx], meta=dict(self.meta),
        )


# ── Signal helpers ───────────────────────────────────────────────────────────

def compute_plv_matrix(eeg_window: np.ndarray) -> np.ndarray:
    """Phase Locking Value between every pair of channels of a (C, T) window."""
    phase = np.angle(hilbert(eeg_window, axis=1))
    diff = phase[:, None, :] - phase[None, :, :]
    return np.abs(np.mean(np.exp(1j * diff), axis=2)).astype(np.float32)


# ── Structure helpers (pure numpy, unit-tested) ──────────────────────────────

def contiguous_runs(orig_idx: np.ndarray, recording: np.ndarray | None = None) -> np.ndarray:
    """Label maximal stretches of windows that are adjacent in the original recording.

    A new run starts whenever the original index jumps (a dropped ictal/postictal window
    or a gap) or the recording id changes.
    """
    orig_idx = np.asarray(orig_idx)
    if len(orig_idx) == 0:
        return np.zeros(0, dtype=np.int64)
    brk = np.diff(orig_idx) != 1
    if recording is not None:
        recording = np.asarray(recording)
        brk |= recording[1:] != recording[:-1]
    return np.concatenate([[0], np.cumsum(brk)]).astype(np.int64)


def seizure_blocks(hard: np.ndarray, run: np.ndarray) -> np.ndarray:
    """Give each maximal run of consecutive preictal windows its own seizure id."""
    hard = np.asarray(hard)
    block = np.full(len(hard), -1, dtype=np.int64)
    current = -1
    for i in range(len(hard)):
        if hard[i] != 1:
            continue
        starts_new = i == 0 or hard[i - 1] != 1 or run[i] != run[i - 1]
        if starts_new:
            current += 1
        block[i] = current
    return block


def soft_risk(block: np.ndarray) -> np.ndarray:
    """Linear 0.10 -> 1.00 ramp across each preictal block, 0 elsewhere (as in the notebook)."""
    risk = np.zeros(len(block), dtype=np.float32)
    for b in np.unique(block[block >= 0]):
        idx = np.where(block == b)[0]
        n = len(idx)
        risk[idx] = 1.0 if n == 1 else 0.10 + 0.90 * np.arange(n) / (n - 1)
    return risk


# ── Loading ──────────────────────────────────────────────────────────────────

def list_subjects(processed_dir: str) -> list[str]:
    return sorted(
        s for s in os.listdir(processed_dir)
        if s.startswith("chb") and os.path.exists(os.path.join(processed_dir, s, "segments.npz"))
    )


def _first_key(npz, keys):
    for k in keys:
        if k in npz.files:
            return k
    return None


def load_subject(processed_dir: str, subject: str, cache_plv: bool = True) -> SubjectData:
    subj_dir = os.path.join(processed_dir, subject)
    seg = np.load(os.path.join(subj_dir, "segments.npz"), allow_pickle=True)
    y_all = np.array([str(v).lower() for v in seg["y"]])
    keep = np.isin(y_all, KEEP)
    orig_idx = np.where(keep)[0]

    X = seg["X"][keep].astype(np.float32)
    hard = (y_all[keep] == "preictal").astype(np.int64)

    rec_key = _first_key(seg, RECORDING_KEYS)
    recording = seg[rec_key][keep] if rec_key else None
    run = contiguous_runs(orig_idx, recording)

    # A time key, if present, also breaks runs across recording gaps.
    time_key = _first_key(seg, TIME_KEYS)
    if time_key is not None:
        t = np.asarray(seg[time_key][keep], dtype=float)
        gap = np.concatenate([[False], np.abs(np.diff(t) - np.median(np.diff(t))) > 1e-3])
        run = np.cumsum((np.concatenate([[False], np.diff(run) != 0]) | gap)).astype(np.int64)

    block = seizure_blocks(hard, run)
    risk = soft_risk(block)

    plv = _load_plv(subj_dir, X, keep, len(y_all), cache_plv)

    return SubjectData(
        subject=subject, X=X, plv=plv, hard=hard, risk=risk, block=block, run=run,
        orig_idx=orig_idx,
        meta={"recording_key": rec_key, "time_key": time_key, "npz_keys": list(seg.files)},
    )


def _load_plv(subj_dir, X, keep, n_total, cache):
    path = os.path.join(subj_dir, "A_plv.npy")
    if os.path.exists(path):
        plv = np.load(path).astype(np.float32)
        if plv.shape[0] == len(X):
            return plv
        if plv.shape[0] == n_total:
            return plv[keep]
    plv = np.stack([compute_plv_matrix(x) for x in X]).astype(np.float32)
    if cache:
        np.save(path, plv)
    return plv
