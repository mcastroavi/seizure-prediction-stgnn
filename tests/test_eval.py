"""Unit tests for the leakage-free split and seizure-level metrics (numpy only, no GPU)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import contiguous_runs, seizure_blocks, soft_risk  # noqa: E402
from src.metrics import (event_metrics, raise_alarms, select_threshold,  # noqa: E402
                         summarize_events, window_metrics)
from src.splits import chronological_split, lopo_folds  # noqa: E402


# ── data structure ───────────────────────────────────────────────────────────

def test_runs_break_where_windows_were_dropped():
    # orig 0..3 kept, 4..6 were ictal (dropped), 7..9 kept
    orig = np.array([0, 1, 2, 3, 7, 8, 9])
    assert contiguous_runs(orig).tolist() == [0, 0, 0, 0, 1, 1, 1]


def test_runs_break_on_recording_change():
    orig = np.arange(6)
    rec = np.array(["a", "a", "a", "b", "b", "b"])
    assert contiguous_runs(orig, rec).tolist() == [0, 0, 0, 1, 1, 1]


def test_adjacent_preictal_blocks_are_separate_seizures():
    # Two seizures whose preictal periods are separated only by a dropped ictal period:
    # the notebook's kept-index gap rule merged these into one block.
    hard = np.array([0, 0, 1, 1, 1, 1, 1, 0])
    run = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert seizure_blocks(hard, run).tolist() == [-1, -1, 0, 0, 1, 1, 1, -1]


def test_soft_risk_ramps_within_each_block():
    block = np.array([-1, 0, 0, 0, -1, 1])
    r = soft_risk(block)
    assert r[0] == 0 and r[4] == 0
    assert np.allclose(r[1:4], [0.10, 0.55, 1.00])
    assert r[5] == 1.0


# ── splits ───────────────────────────────────────────────────────────────────

def test_lopo_never_puts_test_subject_in_train_or_val():
    subs = [f"chb{i:02d}" for i in range(1, 25)]
    counts = {s: 0 if s == "chb24" else 3 for s in subs}
    folds = lopo_folds(subs, counts, n_val=3)
    assert len(folds) == 23  # subject with no seizures has no fold
    for f in folds:
        assert f.test[0] not in f.train and f.test[0] not in f.val
        assert not set(f.train) & set(f.val)
        assert len(f.val) == 3
        assert set(f.train) | set(f.val) | set(f.test) == set(subs)


def test_lopo_is_reproducible():
    subs = [f"chb{i:02d}" for i in range(1, 8)]
    assert [f.val for f in lopo_folds(subs)] == [f.val for f in lopo_folds(subs)]


def _timeline(n_seizures, inter=100, pre=20):
    block = []
    for b in range(n_seizures):
        block += [-1] * inter + [b] * pre
    block += [-1] * inter
    return np.array(block)


def test_chronological_split_is_forward_in_time_and_disjoint():
    block = _timeline(6)
    sp = chronological_split(block, gap=12)
    assert sp["train"].max() < sp["val"].min() < sp["val"].max() < sp["test"].min()
    # each split contains whole seizures
    for name, n_expected in [("train", 3), ("val", 1), ("test", 2)]:
        ids = np.unique(block[sp[name]][block[sp[name]] >= 0])
        assert len(ids) == n_expected
        for b in ids:
            assert np.all(np.isin(np.where(block == b)[0], sp[name]))
    # the gap really separates splits
    assert sp["val"].min() - sp["train"].max() > 12


def test_chronological_split_needs_three_seizures():
    assert chronological_split(_timeline(2)) is None


# ── alarms and events ────────────────────────────────────────────────────────

def test_single_spike_does_not_raise_alarm_with_smoothing():
    probs = np.zeros(20); probs[10] = 0.99
    assert not raise_alarms(probs, np.zeros(20), tau=0.5, k=3, n=5).any()


def test_k_of_n_and_refractory():
    probs = np.zeros(40); probs[5:8] = 0.9; probs[12:15] = 0.9; probs[30:33] = 0.9
    a = raise_alarms(probs, np.zeros(40), tau=0.5, k=3, n=5, refractory=20)
    assert np.where(a)[0].tolist() == [7, 32]  # second burst suppressed by refractory


def test_smoothing_does_not_cross_recording_gaps():
    probs = np.array([0.9, 0.9, 0.0, 0.9, 0.9])
    run = np.array([0, 0, 0, 1, 1])
    assert not raise_alarms(probs, run, tau=0.5, k=3, n=5).any()


def test_lead_time_is_per_seizure_not_summed():
    # two 20-window preictal blocks; alarm 10 windows before the end of each
    block = _timeline(2, inter=50, pre=20)
    hard = (block >= 0).astype(int)
    alarms = np.zeros(len(block), bool)
    for b in (0, 1):
        idx = np.where(block == b)[0]
        alarms[idx[10]] = True
    ev = event_metrics(alarms, hard, block, window_sec=5)
    leads = [s["lead_time_min"] for s in ev["seizures"]]
    assert leads == pytest.approx([10 * 5 / 60] * 2)
    # lead time can never exceed the preictal block length
    assert all(s["lead_time_min"] <= s["preictal_min"] for s in ev["seizures"])


def test_false_alarm_rate_per_hour():
    hard = np.zeros(720)  # 720 windows x 5 s = 1 hour interictal
    block = np.full(720, -1)
    alarms = np.zeros(720, bool); alarms[[10, 400]] = True
    s = summarize_events([event_metrics(alarms, hard, block)])
    assert s["interictal_hours"] == pytest.approx(1.0)
    assert s["fpr_per_hour"] == pytest.approx(2.0)


def test_random_predictor_matches_formula():
    # FPR 1/h, SOP 30 min -> p = 1 - exp(-0.5)
    per = [{"n_seizures": 10, "n_predicted": 4, "false_alarms": 5,
            "interictal_hours": 5.0, "seizures": []}]
    s = summarize_events(per, sop_min=30)
    assert s["random_predictor_sensitivity"] == pytest.approx(1 - np.exp(-0.5))
    assert 0 < s["p_value_vs_chance"] <= 1


def test_perfect_predictor_beats_chance():
    block = _timeline(8, inter=720, pre=60)
    hard = (block >= 0).astype(int)
    probs = hard.astype(float)
    a = raise_alarms(probs, np.zeros(len(block)), tau=0.5)
    s = summarize_events([event_metrics(a, hard, block)])
    assert s["sensitivity"] == 1.0 and s["false_alarms"] == 0
    assert s["beats_chance_at_0.05"]


def test_threshold_selection_respects_fpr_target():
    rng = np.random.default_rng(0)
    block = _timeline(6, inter=720, pre=60)
    hard = (block >= 0).astype(int)
    probs = np.clip(0.35 * hard + rng.uniform(0, 0.6, len(hard)), 0, 1)
    v = {"probs": probs, "hard": hard, "block": block, "run": np.zeros(len(hard))}
    tau = select_threshold([v], mode="fpr", target_fpr=0.5)
    s = summarize_events([event_metrics(raise_alarms(probs, v["run"], tau), hard, block)])
    assert s["fpr_per_hour"] <= 0.5


def test_window_metrics_basic():
    m = window_metrics(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]), 0.5)
    assert m["auc"] == 1.0 and m["sensitivity"] == 1.0 and m["specificity"] == 1.0
