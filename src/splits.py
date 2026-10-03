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


def chronological_split(block: np.ndarray, test_frac: float = 1 / 3,
                        gap: int = 12) -> dict[str, np.ndarray] | None:
    """Split one subject's windows forward in time by seizure.

    Parameters
    ----------
    block : per-window seizure-block id (-1 for interictal), windows in recording order.
    test_frac : fraction of seizures (rounded, at least one) reserved for testing.
    gap : windows dropped right after each cut so neighbouring windows never straddle it.

    Returns ``None`` when the subject has fewer than 3 seizures (need >=1 train, 1 val, 1 test).
    """
    block = np.asarray(block)
    ids = [int(b) for b in np.unique(block[block >= 0])]
    ids.sort(key=lambda b: np.where(block == b)[0].min())
    n = len(ids)
    if n < 3:
        return None
    n_test = max(1, int(round(n * test_frac)))
    n_train = n - n_test - 1
    if n_train < 1:
        n_test, n_train = n - 2, 1

    cut_val = _block_end(block, ids[n_train - 1]) + 1
    cut_test = _block_end(block, ids[n_train]) + 1
    idx = np.arange(len(block))
    train = idx[idx < cut_val]
    val = idx[(idx >= cut_val + gap) & (idx < cut_test)]
    test = idx[idx >= cut_test + gap]
    return {"train": train, "val": val, "test": test}
