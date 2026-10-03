"""Leakage-free data splits.

Two protocols are provided; both keep every test window unseen during training *and*
keep test windows away from their temporal neighbours in the training set.

``lopo_folds`` — leave-one-patient-out (cross-subject)
    For each subject: train on the others, pick a few other subjects for validation /
    threshold selection, test on the held-out subject. This is what "cross-subject"
    means in the seizure-prediction literature.

``chronological_split`` — patient-specific, forward in time
    Within one subject, train on the earliest seizures, validate on the next one, and
    test on the remaining later seizures plus all interictal data recorded after them.
    The model never sees the future of the recording it is tested on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Fold:
    name: str
    train: list[str]
    val: list[str]
    test: list[str]


def lopo_folds(subjects: list[str], seizure_counts: dict[str, int] | None = None,
               n_val: int = 3, seed: int = 42) -> list[Fold]:
    """One fold per subject that has at least one seizure.

    Validation subjects are drawn (reproducibly) from the remaining subjects that have
    seizures, so a threshold can be chosen without touching the test subject.
    """
    counts = seizure_counts or {s: 1 for s in subjects}
    with_sz = [s for s in subjects if counts.get(s, 0) > 0]
    folds = []
    for i, test_subj in enumerate(with_sz):
        others = [s for s in subjects if s != test_subj]
        pool = [s for s in others if counts.get(s, 0) > 0]
        rng = np.random.default_rng(seed + i)
        k = max(0, min(n_val, len(pool) - 1))  # leave at least one seizure subject for training
        val = sorted(rng.choice(pool, size=k, replace=False).tolist()) if k else []
        train = [s for s in others if s not in val]
        folds.append(Fold(name=test_subj, train=train, val=val, test=[test_subj]))
    return folds


def _block_end(block: np.ndarray, b: int) -> int:
    return int(np.where(block == b)[0].max())


def chronological_split(block: np.ndarray, fractions=(0.5, 0.2, 0.3),
                        gap: int = 12) -> dict[str, np.ndarray] | None:
    """Split one subject's windows forward in time: train → validation → test.

    Cuts are placed anywhere outside a preictal period (so no seizure's preictal period is
    split), as close as possible to the requested fractions of *recording time*, with at
    least one seizure in each part. Splitting by time rather than by
    seizure count matters: validation needs hours of interictal data, or the threshold chosen
    on it is meaningless (a seizure-count split once gave a validation set with ~10 minutes
    of interictal data).

    Parameters
    ----------
    block : per-window seizure-block id (-1 for interictal), windows in recording order.
    fractions : target (train, val, test) shares of the windows.
    gap : windows dropped right after each cut so neighbouring windows never straddle it.

    Returns ``None`` when the subject has fewer than 3 seizures.
    """
    block = np.asarray(block)
    ids = [int(b) for b in np.unique(block[block >= 0])]
    ids.sort(key=lambda b: np.where(block == b)[0].min())
    n = len(ids)
    if n < 3:
        return None
    N = len(block)
    ends = np.array(sorted(_block_end(block, b) + 1 for b in ids))   # first index after each block
    pos = np.arange(1, N)
    valid = ~((block[pos - 1] >= 0) & (block[pos - 1] == block[pos]))  # not inside a block
    if gap > 0:                       # the dropped gap after a cut must not eat into a preictal block
        pre = np.concatenate([[0], np.cumsum(block >= 0)])
        valid &= (pre[np.minimum(pos + gap, N)] - pre[pos]) == 0
    cand = pos[valid]
    before = np.searchsorted(ends, cand, side="right")                # seizures fully before cut

    t_val, t_test = fractions[0] * N, (fractions[0] + fractions[1]) * N
    ok = (before >= 1) & (before <= n - 2)
    if not ok.any():
        return None
    cut_val = int(cand[ok][np.argmin(np.abs(cand[ok] - t_val))])
    k_val = int(np.searchsorted(ends, cut_val, side="right"))
    ok = (before >= k_val + 1) & (before <= n - 1) & (cand > cut_val + gap)
    if not ok.any():
        return None
    cut_test = int(cand[ok][np.argmin(np.abs(cand[ok] - t_test))])
    idx = np.arange(N)
    train = idx[idx < cut_val]
    val = idx[(idx >= cut_val + gap) & (idx < cut_test)]
    test = idx[idx >= cut_test + gap]
    return {"train": train, "val": val, "test": test}
