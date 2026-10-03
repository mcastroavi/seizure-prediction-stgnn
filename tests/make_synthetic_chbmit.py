"""Build a small synthetic CHB-MIT-like dataset (EDF files + summary files) for testing.

    python -m tests.make_synthetic_chbmit --out data/synthetic

It reproduces the structure the pipeline must handle: 23-channel headers with a dummy "-"
channel and a duplicated T8-P8, a referential (CS2) montage file, a summary without file
times (like chb24), recordings that cross midnight, gaps between files, and several
seizures per subject. In the minutes before each seizure a shared oscillation is mixed
into all channels, so inter-channel phase synchrony (PLV) rises, which is the effect the
model is meant to detect.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chbmit import STANDARD_CHANNELS  # noqa: E402
from src.edf import write_edf  # noqa: E402

HEADER_23 = STANDARD_CHANNELS[:4] + ["-"] + STANDARD_CHANNELS[4:] + ["P7-T7", "T8-P8"]
ELECTRODES = sorted({e for ch in STANDARD_CHANNELS for e in ch.split("-")})


def _hms(sec):
    sec = int(sec) % 86400
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def make_signals(n, fs, onsets, offsets, file_t0, preictal_sec, rng, n_src=None):
    """Electrode-level signals (len(ELECTRODES), n) with preictal synchrony and ictal bursts."""
    t = file_t0 + np.arange(n) / fs
    sig = rng.standard_normal((len(ELECTRODES), n)).astype(np.float32) * 30
    common = np.sin(2 * np.pi * 9.0 * t + rng.uniform(0, 2 * np.pi)).astype(np.float32)
    # Electrode-specific gains: a component identical on every electrode would cancel in
    # bipolar derivations (A - B), exactly as volume-conducted common signals do in real EEG.
    gain = rng.uniform(0.0, 1.5, len(ELECTRODES)).astype(np.float32)[:, None]
    for on, off in zip(onsets, offsets):
        ramp = np.clip(1 - (on - t) / preictal_sec, 0, 1) * (t < on)
        sig += gain * (60 * ramp)[None, :] * common[None, :]
        ictal = (t >= on) & (t < off)
        sig[:, ictal] += 150 * np.sin(2 * np.pi * 3.0 * t[ictal])[None, :]
    return sig


def bipolar(elec_sig, labels):
    pos = {e: i for i, e in enumerate(ELECTRODES)}
    out = []
    for lab in labels:
        if lab == "-":
            out.append(np.zeros(elec_sig.shape[1], np.float32))
        else:
            a, b = lab.split("-")
            out.append(elec_sig[pos[a]] - elec_sig[pos[b]])
    return np.stack(out)


def make_subject(out, subj, n_files, file_min, gap_sec, start_tod, seizures_by_file, fs,
                 preictal_sec, rng, with_times=True, referential_file=None):
    os.makedirs(os.path.join(out, subj), exist_ok=True)
    lines = ["Data Sampling Rate: %d Hz" % fs, "*************************", ""]
    t_abs = start_tod
    for k in range(n_files):
        name = f"{subj}_{k + 1:02d}.edf"
        dur = file_min * 60
        sz = seizures_by_file.get(k, [])
        # seizures (in absolute time) from this file and neighbours influence the preictal ramp
        all_on = [t_abs_f + s for kk, t_abs_f in _starts(start_tod, n_files, file_min, gap_sec).items()
                  for s, _ in seizures_by_file.get(kk, [])]
        all_off = [t_abs_f + e for kk, t_abs_f in _starts(start_tod, n_files, file_min, gap_sec).items()
                   for _, e in seizures_by_file.get(kk, [])]
        elec = make_signals(dur * fs, fs, all_on, all_off, t_abs, preictal_sec, rng)
        if k == referential_file:
            labels = [f"{e}-CS2" for e in ELECTRODES]
            pos = {e: i for i, e in enumerate(ELECTRODES)}
            sig = np.stack([elec[pos[e]] for e in ELECTRODES])
        else:
            labels = HEADER_23
            sig = bipolar(elec, labels)
        hdr_time = _hms(t_abs).replace(":", ".")
        write_edf(os.path.join(out, subj, name), sig, fs, labels, start_time=hdr_time)
        lines.append(f"File Name: {name}")
        if with_times:
            lines.append(f"File Start Time: {_hms(t_abs)}")
            lines.append(f"File End Time: {_hms(t_abs + dur)}")
        lines.append(f"Number of Seizures in File: {len(sz)}")
        for i, (s, e) in enumerate(sz, 1):
            lab = f"Seizure {i}" if len(sz) > 1 else "Seizure"
            lines.append(f"{lab} Start Time: {s} seconds")
            lines.append(f"{lab} End Time: {e} seconds")
        lines.append("")
        t_abs += dur + gap_sec
    with open(os.path.join(out, subj, f"{subj}-summary.txt"), "w") as f:
        f.write("\n".join(lines))


def _starts(start_tod, n_files, file_min, gap_sec):
    return {k: start_tod + k * (file_min * 60 + gap_sec) for k in range(n_files)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    pre = 600  # synthetic preictal ramp: 10 minutes
    # (subject, n_files, file_min, gap_s, start time of day, {file: [(start_s, end_s)]}, with_times, referential)
    specs = [
        ("chb01", 8, 20, 5, 22 * 3600, {2: [(900, 940)], 5: [(600, 630)]}, True, None),   # crosses midnight
        ("chb02", 8, 20, 30, 10 * 3600, {3: [(700, 740)], 6: [(1000, 1030)]}, True, 4),   # one CS2 file
        ("chb03", 8, 20, 5, 8 * 3600, {1: [(1000, 1040)], 4: [(300, 330)], 7: [(900, 930)]}, True, None),
        ("chb04", 8, 20, 5, 13 * 3600, {2: [(400, 420)], 6: [(800, 840)]}, False, None),  # no file times
    ]
    for subj, nf, fm, gap, tod, sz, times, ref in specs:
        make_subject(a.out, subj, nf, fm, gap, tod, sz, a.fs, pre, rng, times, ref)
        print(f"wrote {subj}")


if __name__ == "__main__":
    main()
