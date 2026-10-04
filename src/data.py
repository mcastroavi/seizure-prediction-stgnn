"""Per-subject data access with the structure honest evaluation needs.

Two on-disk formats are supported:

* **v3** (``src.preprocess``): ``X.npy`` / ``plv.npy`` memory maps + ``meta.npz`` with
  absolute timestamps and true seizure ids. Every window of the recording is present.
* **v2 legacy**: ``segments.npz`` (keys ``X``, ``y``) + optional ``A_plv.npy`` as produced
  by the original notebooks. Windows are assumed to be in recording order; run
  ``python -m src.inspect_segments`` to check.

Only interictal and preictal windows are exposed; ictal, postictal and buffer windows are
dropped, and contiguity ("runs") breaks wherever windows were dropped or recordings have gaps.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import hilbert

KEEP = ("interictal", "preictal")


@dataclass
class SubjectData:
    subject: str
    X: np.ndarray            # full window array (may be a memory map); index with ``sel``
    plv: np.ndarray          # full PLV array; index with ``sel``
    sel: np.ndarray          # (N,) indices into X / plv for the windows exposed here
    hard: np.ndarray         # (N,) 0 = interictal, 1 = preictal
    risk: np.ndarray         # (N,) soft risk target in [0, 1]
    block: np.ndarray        # (N,) seizure id for preictal windows, -1 otherwise
    run: np.ndarray          # (N,) id of the contiguous stretch of recording
    t_start: np.ndarray | None = None   # (N,) absolute seconds (v3 only)
    meta: dict = field(default_factory=dict)
    plv_bands: np.ndarray | None = None  # full (W, 5, C, C) band PLV, if preprocessed with bands
    bandpow: np.ndarray | None = None    # full (W, C, 5) log relative band power

    def __len__(self) -> int:
        return len(self.sel)

    @property
    def n_seizures(self) -> int:
        return len(np.unique(self.block[self.block >= 0]))

    def window(self, i: int):
        j = self.sel[i]
        return self.X[j], self.plv[j]

    @property
    def has_bands(self) -> bool:
        return self.plv_bands is not None and self.bandpow is not None

    def bands(self, i: int):
        j = self.sel[i]
        return self.plv_bands[j], self.bandpow[j]

    def subset(self, idx) -> "SubjectData":
        """Restrict to positions ``idx`` without copying the big arrays."""
        idx = np.asarray(idx, dtype=np.int64)
        return SubjectData(self.subject, self.X, self.plv, self.sel[idx], self.hard[idx],
                           self.risk[idx], self.block[idx], self.run[idx],
                           None if self.t_start is None else self.t_start[idx], dict(self.meta),
                           self.plv_bands, self.bandpow)


# ── Signal helper ────────────────────────────────────────────────────────────

def compute_plv_matrix(eeg_window: np.ndarray) -> np.ndarray:
    """Phase Locking Value between every pair of channels of a (C, T) window."""
    phase = np.angle(hilbert(eeg_window, axis=1))
    diff = phase[:, None, :] - phase[None, :, :]
    return np.abs(np.mean(np.exp(1j * diff), axis=2)).astype(np.float32)


# ── Structure helpers (pure numpy, unit-tested) ──────────────────────────────

def contiguous_runs(orig_idx: np.ndarray, recording: np.ndarray | None = None) -> np.ndarray:
    """A new run starts where the original index jumps or the recording changes."""
    orig_idx = np.asarray(orig_idx)
    if len(orig_idx) == 0:
        return np.zeros(0, dtype=np.int64)
    brk = np.diff(orig_idx) != 1
    if recording is not None:
        recording = np.asarray(recording)
        brk |= recording[1:] != recording[:-1]
    return np.concatenate([[0], np.cumsum(brk)]).astype(np.int64)


def runs_from_time(t_start: np.ndarray, window_sec: float) -> np.ndarray:
    """A new run starts wherever consecutive windows are not exactly one window apart."""
    t = np.asarray(t_start, dtype=float)
    if len(t) == 0:
        return np.zeros(0, dtype=np.int64)
    brk = np.abs(np.diff(t) - window_sec) > 1e-3
    return np.concatenate([[0], np.cumsum(brk)]).astype(np.int64)


def seizure_blocks(hard: np.ndarray, run: np.ndarray) -> np.ndarray:
    """Legacy format only: give each maximal run of consecutive preictal windows an id."""
    hard = np.asarray(hard)
    block = np.full(len(hard), -1, dtype=np.int64)
    current = -1
    for i in range(len(hard)):
        if hard[i] != 1:
            continue
        if i == 0 or hard[i - 1] != 1 or run[i] != run[i - 1]:
            current += 1
        block[i] = current
    return block


def soft_risk(block: np.ndarray) -> np.ndarray:
    """Legacy ramp: 0.10 -> 1.00 by position within each preictal block."""
    risk = np.zeros(len(block), dtype=np.float32)
    for b in np.unique(block[block >= 0]):
        idx = np.where(block == b)[0]
        n = len(idx)
        risk[idx] = 1.0 if n == 1 else 0.10 + 0.90 * np.arange(n) / (n - 1)
    return risk


def soft_risk_by_time(t_start: np.ndarray, block: np.ndarray, onsets: np.ndarray,
                      window_sec: float, sop_sec: float) -> np.ndarray:
    """v3 ramp: 0.10 at the start of the SOP rising to 1.00 at onset, by *time to onset*.

    Unlike the positional ramp, a preictal stretch truncated by the start of a recording
    keeps the risk values that match how close it really is to the seizure.
    """
    risk = np.zeros(len(block), dtype=np.float32)
    m = block >= 0
    if m.any():
        tto = onsets[block[m]] - (t_start[m] + window_sec)        # 0 .. sop - window
        risk[m] = np.clip(0.10 + 0.90 * (1 - tto / max(sop_sec - window_sec, 1e-9)), 0.10, 1.0)
    return risk


# ── Loading ──────────────────────────────────────────────────────────────────

def list_subjects(processed_dir: str) -> list[str]:
    def ok(s):
        d = os.path.join(processed_dir, s)
        return os.path.exists(os.path.join(d, "meta.npz")) or os.path.exists(os.path.join(d, "segments.npz"))
    return sorted(s for s in os.listdir(processed_dir) if s.startswith("chb") and ok(s))


def load_subject(processed_dir: str, subject: str, cache_plv: bool = True) -> SubjectData:
    d = os.path.join(processed_dir, subject)
    if os.path.exists(os.path.join(d, "meta.npz")):
        return _load_v3(d, subject)
    return _load_legacy(d, subject, cache_plv)


def _load_v3(d: str, subject: str) -> SubjectData:
    meta = np.load(os.path.join(d, "meta.npz"), allow_pickle=False)
    params = json.loads(str(meta["params"]))
    w = float(params["window_sec"])
    label = meta["label"]
    sel = np.where((label == 0) | (label == 1))[0]
    t = meta["t_start"][sel]
    hard = (label[sel] == 1).astype(np.int64)
    block = meta["seizure_id"][sel].astype(np.int64)
    onsets = meta["seizures"][:, 0] if meta["seizures"].size else np.zeros(0)
    risk = soft_risk_by_time(t, block, onsets, w, params["preictal_min"] * 60)
    run = runs_from_time(t, w)
    X = np.load(os.path.join(d, "X.npy"), mmap_mode="r")
    plv = np.load(os.path.join(d, "plv.npy"), mmap_mode="r")
    pb_path, bp_path = os.path.join(d, "plv_bands.npy"), os.path.join(d, "bandpow.npy")
    has_b = os.path.exists(pb_path) and os.path.exists(bp_path)
    return SubjectData(subject, X, plv, sel, hard, risk, block, run, t,
                       meta={"format": "v3", "params": params,
                             "channels": meta["channels"].tolist(),
                             "seizures": meta["seizures"].tolist()},
                       plv_bands=np.load(pb_path, mmap_mode="r") if has_b else None,
                       bandpow=np.load(bp_path, mmap_mode="r") if has_b else None)


def _load_legacy(d: str, subject: str, cache_plv: bool) -> SubjectData:
    seg = np.load(os.path.join(d, "segments.npz"), allow_pickle=True)
    y_all = np.array([str(v).lower() for v in seg["y"]])
    keep = np.isin(y_all, KEEP)
    orig_idx = np.where(keep)[0]
    X = seg["X"][keep].astype(np.float32)
    hard = (y_all[keep] == "preictal").astype(np.int64)
    run = contiguous_runs(orig_idx)
    block = seizure_blocks(hard, run)
    plv_path = os.path.join(d, "A_plv.npy")
    plv = None
    if os.path.exists(plv_path):
        p = np.load(plv_path).astype(np.float32)
        plv = p if p.shape[0] == len(X) else (p[keep] if p.shape[0] == len(y_all) else None)
    if plv is None:
        plv = np.stack([compute_plv_matrix(x) for x in X]).astype(np.float32)
        if cache_plv:
            np.save(plv_path, plv)
    return SubjectData(subject, X, plv, np.arange(len(X)), hard, soft_risk(block), block, run,
                       None, meta={"format": "legacy", "npz_keys": list(seg.files)})
