"""Tests for the training-time augmentation (numpy only)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.augment import AugmentConfig, augment_window  # noqa: E402

OFF = AugmentConfig(p_shift=0, p_noise=0, p_mask=0, p_gain=0, p_chdrop=0, p_jitter=0)


def _window(seed=0, C=18, T=640, B=5):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((C, T)).astype(np.float32)
    plv = rng.uniform(0, 1, (C, C)).astype(np.float32)
    plv = (plv + plv.T) / 2
    np.fill_diagonal(plv, 1.0)
    pb = np.stack([plv] * B)
    bp = np.log10(rng.dirichlet(np.ones(B), size=C)).astype(np.float32)
    return x, plv, pb, bp


def test_all_off_is_identity_and_does_not_mutate_inputs():
    x, plv, pb, bp = _window()
    copies = [a.copy() for a in (x, plv, pb, bp)]
    out = augment_window(x, plv, np.random.default_rng(1), OFF, pb, bp)
    for a, b in zip(out, copies):
        assert np.array_equal(a, b)
    for a, b in zip((x, plv, pb, bp), copies):
        assert np.array_equal(a, b)                 # originals untouched


def test_shapes_dtypes_and_ranges_preserved():
    x, plv, pb, bp = _window()
    on = AugmentConfig(p_shift=1, p_noise=1, p_mask=1, p_gain=1, p_chdrop=1, p_jitter=1)
    for s in range(50):
        xa, pa, pba, bpa = augment_window(x, plv, np.random.default_rng(s), on, pb, bp)
        assert xa.shape == x.shape and pa.shape == plv.shape and pba.shape == pb.shape and bpa.shape == bp.shape
        assert xa.dtype == np.float32 and np.isfinite(xa).all()
        assert pa.min() >= 0 and pa.max() <= 1 and np.allclose(pa, pa.T)
        assert pba.min() >= 0 and pba.max() <= 1


def test_channel_dropout_zeroes_signal_and_edges():
    x, plv, pb, bp = _window()
    cfg = AugmentConfig(p_shift=0, p_noise=0, p_mask=0, p_gain=0, p_chdrop=1, p_jitter=0, max_chdrop=2)
    xa, pa, pba, bpa = augment_window(x, plv, np.random.default_rng(3), cfg, pb, bp)
    dropped = np.where(np.all(xa == 0, axis=1))[0]
    assert 1 <= len(dropped) <= 2
    for c in dropped:
        assert np.all(pa[c] == 0) and np.all(pa[:, c] == 0) and np.all(pba[:, c] == 0)
        keep = np.setdiff1d(np.arange(18), dropped)
        assert np.allclose(bpa[c], bp[keep].mean(axis=0))


def test_time_shift_is_a_circular_roll():
    x, plv, _, _ = _window()
    cfg = AugmentConfig(p_shift=1, p_noise=0, p_mask=0, p_gain=0, p_chdrop=0, p_jitter=0)
    xa, _, _, _ = augment_window(x, plv, np.random.default_rng(5), cfg)
    shifts = [k for k in range(-320, 321) if np.array_equal(np.roll(x, k, axis=1), xa)]
    assert len(shifts) >= 1


def test_works_without_band_features():
    x, plv, _, _ = _window()
    xa, pa, pba, bpa = augment_window(x, plv, np.random.default_rng(0))
    assert pba is None and bpa is None and xa.shape == x.shape
