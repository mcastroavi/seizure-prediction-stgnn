"""Training-time data augmentation for EEG windows (numpy only, applied on the fly).

Each transform is applied independently with its own probability, to training windows only:

* **time shift**      circular shift of the 5-s window by up to ±50% (where in the window an
                      event falls is arbitrary)
* **gaussian noise**  additive noise, SD drawn up to ``noise_max`` (in z-scored units)
* **time masking**    a random stretch of up to 10% of the window set to zero (all channels)
* **channel gain**    each channel scaled by 0.8–1.2 (electrode impedance differences)
* **channel dropout** up to 2 channels zeroed, with their PLV edges and band power neutralised
                      (a lost or noisy electrode)
* **PLV jitter**      small noise on the PLV values (estimation noise of 5-s phase statistics)

Waveform transforms (shift, noise, masking, gain) change only the signal seen by the node and
temporal encoders; the precomputed PLV and band features are perturbed separately by dropout
and jitter. A label-preserving check: none of these move a window across the preictal boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentConfig:
    p_shift: float = 0.5
    p_noise: float = 0.5
    noise_max: float = 0.2
    p_mask: float = 0.3
    mask_frac: float = 0.1
    p_gain: float = 0.5
    gain_range: tuple = (0.8, 1.2)
    p_chdrop: float = 0.3
    max_chdrop: int = 2
    p_jitter: float = 0.5
    jitter_sd: float = 0.02


def augment_window(x, plv, rng, cfg: AugmentConfig = AugmentConfig(), plv_bands=None, bandpow=None):
    """Return augmented copies of (x (C,T), plv (C,C), plv_bands (B,C,C) | None, bandpow (C,B) | None)."""
    x = np.array(x, dtype=np.float32, copy=True)
    plv = np.array(plv, dtype=np.float32, copy=True)
    pb = None if plv_bands is None else np.array(plv_bands, dtype=np.float32, copy=True)
    bp = None if bandpow is None else np.array(bandpow, dtype=np.float32, copy=True)
    C, T = x.shape

    if rng.random() < cfg.p_shift:
        x = np.roll(x, int(rng.integers(-T // 2, T // 2 + 1)), axis=1)
    if rng.random() < cfg.p_gain:
        x *= rng.uniform(*cfg.gain_range, size=(C, 1)).astype(np.float32)
    if rng.random() < cfg.p_noise:
        x += rng.normal(0, rng.uniform(0, cfg.noise_max), size=x.shape).astype(np.float32)
    if rng.random() < cfg.p_mask:
        w = int(rng.integers(1, max(2, int(cfg.mask_frac * T)) + 1))
        s = int(rng.integers(0, T - w + 1))
        x[:, s:s + w] = 0.0
    if rng.random() < cfg.p_jitter:
        plv = np.clip(plv + rng.normal(0, cfg.jitter_sd, plv.shape), 0, 1).astype(np.float32)
        plv = (plv + plv.T) / 2
        if pb is not None:
            pb = np.clip(pb + rng.normal(0, cfg.jitter_sd, pb.shape), 0, 1).astype(np.float32)
            pb = (pb + pb.transpose(0, 2, 1)) / 2
    if rng.random() < cfg.p_chdrop:
        k = int(rng.integers(1, cfg.max_chdrop + 1))
        ch = rng.choice(C, size=k, replace=False)
        keep = np.setdiff1d(np.arange(C), ch)
        x[ch] = 0.0
        plv[ch, :] = 0.0
        plv[:, ch] = 0.0
        if pb is not None:
            pb[:, ch, :] = 0.0
            pb[:, :, ch] = 0.0
        if bp is not None:
            bp[ch] = bp[keep].mean(axis=0)          # neutral: average of the remaining channels
    return x, plv, pb, bp
