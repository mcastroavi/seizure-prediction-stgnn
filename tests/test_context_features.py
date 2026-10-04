"""Tests for context-model extra features and lead-seizure scoring (numpy only)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.context_features import seizure_history, time_of_day  # noqa: E402
from src.metrics import event_metrics, lead_seizures, summarize_events  # noqa: E402

H = 3600.0


def test_time_of_day_is_periodic_and_wraps_days():
    t = np.array([0.0, 6 * H, 12 * H, 24 * H, 30 * H])
    tod = time_of_day(t)
    assert np.allclose(tod[0], [0, 1]) and np.allclose(tod[1], [1, 0], atol=1e-6)
    assert np.allclose(tod[2], [0, -1], atol=1e-6)
    assert np.allclose(tod[3], tod[0]) and np.allclose(tod[4], tod[1])   # next day, same clock time


def test_history_uses_only_past_seizures():
    seizures = np.array([[10 * H, 10 * H + 60], [20 * H, 20 * H + 60]])
    t = np.array([5 * H, 9.9 * H, 15 * H, 19.9 * H, 25 * H])
    f = seizure_history(t, seizures, floor_h=1.0)
    assert f[0, 1] == 0 and f[1, 1] == 0                  # nothing has happened yet
    assert np.allclose(f[:2, 0], 1.0)                      # "no previous seizure" = cap
    assert f[2, 1] == 1 and f[3, 1] == 1 and f[4, 1] == 1
    # 19.9 h is ~9.9 h after the first seizure: knows nothing about the one at 20 h
    assert np.isclose(f[3, 0], seizure_history(np.array([19.9 * H]), seizures[:1], 1.0)[0, 0])
    assert f[4, 0] < f[3, 0]                               # 25 h: 5 h after the second seizure


def test_history_floor_hides_the_labelling_buffer():
    # windows 10 min and 50 min after a seizure (inside the 60-min buffer) must look identical
    seizures = np.array([[0.0, 60.0]])
    t = np.array([60 + 10 * 60, 60 + 50 * 60, 60 + 59 * 60])
    f = seizure_history(t, seizures, floor_h=1.0)
    assert np.allclose(f[:, 0], f[0, 0]) and np.allclose(f[:, 0], 0.0)
    later = seizure_history(np.array([60 + 3 * H]), seizures, floor_h=1.0)
    assert later[0, 0] > 0


def test_history_without_seizures():
    f = seizure_history(np.arange(5) * H, np.zeros((0, 2)), floor_h=1.0)
    assert np.all(f[:, 0] == 1) and np.all(f[:, 1] == 0)


def test_lead_seizures():
    sz = np.array([[0, 100], [2 * H, 2 * H + 50], [10 * H, 10 * H + 50], [11 * H, 11 * H + 50]])
    assert lead_seizures(sz, 4).tolist() == [True, False, True, False]
    assert lead_seizures(sz, 0).all()
    # order of rows does not matter
    assert lead_seizures(sz[::-1], 4).tolist() == [False, True, False, True]


def test_event_metrics_skips_non_lead_seizures():
    block = np.array([-1] * 50 + [0] * 20 + [-1] * 50 + [1] * 20 + [-1] * 50)
    hard = (block >= 0).astype(int)
    alarms = np.zeros(len(block), bool)
    alarms[np.where(block == 1)[0][5]] = True              # only seizure 1 predicted
    full = event_metrics(alarms, hard, block)
    lead = event_metrics(alarms, hard, block, skip_blocks={1})
    assert full["n_seizures"] == 2 and full["n_predicted"] == 1
    assert lead["n_seizures"] == 1 and lead["n_predicted"] == 0 and lead["seizures_skipped_non_lead"] == 1
    assert lead["false_alarms"] == 0                         # alarm in a skipped preictal is not false
    assert summarize_events([lead])["seizures_skipped_non_lead"] == 1
