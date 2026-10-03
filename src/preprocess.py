"""Turn raw CHB-MIT EDF files into chronological, timestamped 5-second windows.

    python -m src.preprocess --raw_dir data/chb-mit --out_dir data/processed_v3 --workers 4

For each subject this writes ``<out_dir>/<subject>/``:

* ``X.npy``    float16 (N, 18, T)  band-passed EEG windows (memory-mappable)
* ``plv.npy``  float16 (N, 18, 18) Phase Locking Value between every pair of channels
* ``meta.npz`` per-window ``label`` (0 interictal, 1 preictal, 2 ictal, 3 postictal,
  4 excluded), ``seizure_id`` (the upcoming seizure for preictal windows), ``t_start``
  (absolute seconds), ``file_idx``; plus the seizure table, file list and parameters.

Every window of every recording is kept, in recording order, so evaluation can run on the
full continuous data (false alarms per hour are then measured on all interictal hours).
Training subsamples interictal windows on the fly instead.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from fractions import Fraction
from multiprocessing import Pool

import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

from .chbmit import (EXCLUDED, ICTAL, INTERICTAL, POSTICTAL, PREICTAL, STANDARD_CHANNELS,
                     apply_plan, build_timeline, channel_plan, label_windows, parse_summary)
from .edf import read_header, read_signals


def plv_batch(windows: np.ndarray, chunk: int = 256) -> np.ndarray:
    """PLV for a batch of windows (W, C, T) -> (W, C, C)."""
    out = np.empty((len(windows), windows.shape[1], windows.shape[1]), dtype=np.float32)
    for s in range(0, len(windows), chunk):
        z = np.exp(1j * np.angle(hilbert(windows[s:s + chunk], axis=-1)))
        out[s:s + chunk] = np.abs(np.einsum("wct,wdt->wcd", z, z.conj())) / windows.shape[-1]
    return out


def preprocess_subject(subject: str, raw_dir: str, out_dir: str, fs_out: int = 128,
                       window_sec: float = 5.0, band=(0.5, 40.0), preictal_min: float = 30,
                       postictal_min: float = 5, buffer_min: float = 60, sph_min: float = 0,
                       channels=STANDARD_CHANNELS) -> dict:
    t_begin = time.time()
    sdir = os.path.join(raw_dir, subject)
    summary_path = os.path.join(sdir, f"{subject}-summary.txt")
    summary = parse_summary(open(summary_path).read()) if os.path.exists(summary_path) else {}
    paths = sorted(glob.glob(os.path.join(sdir, "*.edf")))

    files, skipped = [], []
    for p in paths:
        name = os.path.basename(p)
        h = read_header(p)
        plan = channel_plan(h.labels, channels)
        if plan is None:
            skipped.append({"file": name, "reason": "required channels not available"})
            continue
        idx = sorted({i for step in plan for i in step[1:]})
        fs_in = float(h.fs[idx[0]])
        if len({float(h.fs[i]) for i in idx}) != 1:
            skipped.append({"file": name, "reason": "mixed sampling rates"})
            continue
        entry = summary.get(name)
        tod = entry.start_tod if entry and entry.start_tod is not None else h.start_tod_sec
        files.append({"name": name, "path": p, "header": h, "plan": plan, "idx": idx,
                      "fs_in": fs_in, "tod": tod, "duration": h.duration_sec,
                      "seizures": entry.seizures if entry else []})

    starts = build_timeline([f["tod"] for f in files], [f["duration"] for f in files])
    seizures, seizure_file = [], []
    for k, (f, t) in enumerate(zip(files, starts)):
        f["t_abs"] = t
        for on, off in f["seizures"]:
            seizures.append((t + on, t + off))
            seizure_file.append(f["name"])
    order = np.argsort([s[0] for s in seizures])
    seizures = [seizures[i] for i in order]
    seizure_file = [seizure_file[i] for i in order]

    T = int(round(window_sec * fs_out))
    n_win = [int(f["duration"] // window_sec) for f in files]
    N, C = int(sum(n_win)), len(channels)

    sub_out = os.path.join(out_dir, subject)
    os.makedirs(sub_out, exist_ok=True)
    X = np.lib.format.open_memmap(os.path.join(sub_out, "X.npy"), "w+", np.float16, (N, C, T))
    P = np.lib.format.open_memmap(os.path.join(sub_out, "plv.npy"), "w+", np.float16, (N, C, C))
    t_start = np.empty(N)
    file_idx = np.empty(N, dtype=np.int16)

    pos = 0
    for k, (f, n) in enumerate(zip(files, n_win)):
        sig = apply_plan(read_signals(f["path"], f["header"], f["idx"]), _remap(f["plan"], f["idx"]))
        sos = butter(4, [band[0], band[1]], btype="band", fs=f["fs_in"], output="sos")
        sig = sosfiltfilt(sos, sig, axis=1)
        frac = Fraction(fs_out / f["fs_in"]).limit_denominator(1000)
        if frac != 1:
            sig = resample_poly(sig, frac.numerator, frac.denominator, axis=1)
        sig = sig[:, : n * T].reshape(C, n, T).transpose(1, 0, 2).astype(np.float32)
        X[pos:pos + n] = sig
        P[pos:pos + n] = plv_batch(sig)
        t_start[pos:pos + n] = f["t_abs"] + np.arange(n) * window_sec
        file_idx[pos:pos + n] = k
        pos += n
    X.flush(); P.flush()

    labels, sz_id = label_windows(t_start, window_sec, seizures, preictal_min * 60,
                                  postictal_min * 60, buffer_min * 60, sph_min * 60)
    params = {"fs": fs_out, "window_sec": window_sec, "band": list(band),
              "preictal_min": preictal_min, "postictal_min": postictal_min,
              "buffer_min": buffer_min, "sph_min": sph_min}
    np.savez(os.path.join(sub_out, "meta.npz"), label=labels, seizure_id=sz_id, t_start=t_start,
             file_idx=file_idx, files=np.array([f["name"] for f in files]),
             channels=np.array(channels), seizures=np.array(seizures, dtype=float).reshape(-1, 2),
             seizure_file=np.array(seizure_file), params=json.dumps(params))

    pre_min = [np.sum(sz_id == k) * window_sec / 60 for k in range(len(seizures))]
    report = {
        "subject": subject, "files": len(files), "skipped": skipped,
        "windows": N, "seizures": len(seizures),
        "seizures_with_10min_preictal": int(sum(m >= 10 for m in pre_min)),
        "hours_total": N * window_sec / 3600,
        "hours_interictal": float(np.sum(labels == INTERICTAL) * window_sec / 3600),
        "windows_by_label": {name: int(np.sum(labels == code)) for code, name in
                             [(INTERICTAL, "interictal"), (PREICTAL, "preictal"), (ICTAL, "ictal"),
                              (POSTICTAL, "postictal"), (EXCLUDED, "excluded")]},
        "preictal_minutes_per_seizure": [round(m, 1) for m in pre_min],
        "params": params, "seconds": round(time.time() - t_begin, 1),
    }
    with open(os.path.join(sub_out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def _remap(plan, idx):
    """Re-index a channel plan after reading only the signals in ``idx``."""
    pos = {i: j for j, i in enumerate(idx)}
    return [(s[0],) + tuple(pos[i] for i in s[1:]) for s in plan]


def _run(args):
    subject, kw = args
    r = preprocess_subject(subject, **kw)
    print(f"  {r['subject']}: {r['files']} files ({len(r['skipped'])} skipped), "
          f"{r['hours_total']:.1f} h, {r['seizures']} seizures "
          f"({r['seizures_with_10min_preictal']} with >=10 min preictal), "
          f"interictal {r['hours_interictal']:.1f} h  [{r['seconds']:.0f}s]", flush=True)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", required=True, help="folder containing chb01/, chb02/, ...")
    ap.add_argument("--out_dir", default="data/processed_v3")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--fs_out", type=int, default=128)
    ap.add_argument("--window_sec", type=float, default=5.0)
    ap.add_argument("--preictal_min", type=float, default=30)
    ap.add_argument("--postictal_min", type=float, default=5)
    ap.add_argument("--buffer_min", type=float, default=60,
                    help="interictal windows must be at least this far from any seizure")
    ap.add_argument("--sph_min", type=float, default=0, help="seizure prediction horizon")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)

    subjects = a.subjects or sorted(d for d in os.listdir(a.raw_dir)
                                    if d.startswith("chb") and os.path.isdir(os.path.join(a.raw_dir, d)))
    kw = dict(raw_dir=a.raw_dir, out_dir=a.out_dir, fs_out=a.fs_out, window_sec=a.window_sec,
              preictal_min=a.preictal_min, postictal_min=a.postictal_min,
              buffer_min=a.buffer_min, sph_min=a.sph_min)
    print(f"Preprocessing {len(subjects)} subjects -> {a.out_dir}")
    jobs = [(s, kw) for s in subjects]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            reports = pool.map(_run, jobs)
    else:
        reports = [_run(j) for j in jobs]
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "preprocess_report.json"), "w") as fh:
        json.dump(reports, fh, indent=2)
    tot_h = sum(r["hours_total"] for r in reports)
    print(f"Done: {tot_h:.0f} h, {sum(r['seizures'] for r in reports)} seizures, "
          f"{sum(len(r['skipped']) for r in reports)} files skipped (see report.json).")


if __name__ == "__main__":
    main()
