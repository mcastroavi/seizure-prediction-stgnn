"""Minimal EDF (European Data Format) reader in pure numpy.

CHB-MIT files are plain EDF: a fixed ASCII header followed by int16 data records.
Implementing the reader here (about 60 lines) removes a heavy dependency and makes the
format easy to explain. ``mne.io.read_raw_edf`` would give the same signals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class EDFHeader:
    labels: list[str]
    n_records: int
    record_sec: float
    samples_per_record: np.ndarray
    phys_min: np.ndarray
    phys_max: np.ndarray
    dig_min: np.ndarray
    dig_max: np.ndarray
    header_bytes: int
    start_time: str          # "hh.mm.ss" as stored in the header
    start_date: str          # "dd.mm.yy"

    @property
    def fs(self) -> np.ndarray:
        return self.samples_per_record / self.record_sec

    @property
    def duration_sec(self) -> float:
        return self.n_records * self.record_sec

    @property
    def start_tod_sec(self) -> int:
        """Start time of day in seconds."""
        h, m, s = (int(x) for x in self.start_time.replace(":", ".").split("."))
        return h * 3600 + m * 60 + s


def _fields(raw: bytes, offset: int, width: int, n: int) -> tuple[list[str], int]:
    out = [raw[offset + i * width: offset + (i + 1) * width].decode("latin-1").strip()
           for i in range(n)]
    return out, offset + width * n


def read_header(path: str) -> EDFHeader:
    with open(path, "rb") as f:
        fixed = f.read(256)
        ns = int(fixed[252:256].decode().strip())
        sig = f.read(ns * 256)
    start_date = fixed[168:176].decode("latin-1").strip()
    start_time = fixed[176:184].decode("latin-1").strip()
    header_bytes = int(fixed[184:192].decode().strip())
    n_records = int(fixed[236:244].decode().strip())
    record_sec = float(fixed[244:252].decode().strip())

    o = 0
    labels, o = _fields(sig, o, 16, ns)
    _, o = _fields(sig, o, 80, ns)            # transducer
    _, o = _fields(sig, o, 8, ns)             # physical dimension
    pmin, o = _fields(sig, o, 8, ns)
    pmax, o = _fields(sig, o, 8, ns)
    dmin, o = _fields(sig, o, 8, ns)
    dmax, o = _fields(sig, o, 8, ns)
    _, o = _fields(sig, o, 80, ns)            # prefiltering
    spr, o = _fields(sig, o, 8, ns)

    spr = np.array([int(x) for x in spr])
    if n_records < 0:  # unknown in header: infer from file size
        n_records = (os.path.getsize(path) - header_bytes) // (2 * int(spr.sum()))
    return EDFHeader(
        labels=labels, n_records=n_records, record_sec=record_sec, samples_per_record=spr,
        phys_min=np.array([float(x) for x in pmin]), phys_max=np.array([float(x) for x in pmax]),
        dig_min=np.array([float(x) for x in dmin]), dig_max=np.array([float(x) for x in dmax]),
        header_bytes=header_bytes, start_time=start_time, start_date=start_date,
    )


def read_signals(path: str, header: EDFHeader | None = None,
                 indices: list[int] | None = None) -> np.ndarray:
    """Return physical-unit signals (n_selected, n_samples) as float32.

    All selected signals must share one sampling rate (true for CHB-MIT EEG channels).
    """
    h = header or read_header(path)
    idx = list(range(len(h.labels))) if indices is None else list(indices)
    spr = h.samples_per_record
    if len(set(spr[idx].tolist())) != 1:
        raise ValueError("selected signals have different sampling rates")

    rec_len = int(spr.sum())
    raw = np.fromfile(path, dtype="<i2", offset=h.header_bytes, count=h.n_records * rec_len)
    raw = raw[: (len(raw) // rec_len) * rec_len].reshape(-1, rec_len)
    starts = np.concatenate([[0], np.cumsum(spr)])

    out = np.empty((len(idx), raw.shape[0] * spr[idx[0]]), dtype=np.float32)
    for j, i in enumerate(idx):
        d = raw[:, starts[i]:starts[i + 1]].reshape(-1).astype(np.float32)
        gain = (h.phys_max[i] - h.phys_min[i]) / (h.dig_max[i] - h.dig_min[i])
        out[j] = (d - h.dig_min[i]) * gain + h.phys_min[i]
    return out


def write_edf(path: str, signals: np.ndarray, fs: int, labels: list[str],
              start_time: str = "00.00.00", start_date: str = "01.01.01",
              phys_range: float = 3200.0) -> None:
    """Write a simple EDF file (used to build synthetic test data)."""
    ns, n = signals.shape
    n_rec = n // fs

    def pad(s, w):
        return str(s)[:w].ljust(w).encode("latin-1")

    hdr = (pad("0", 8) + pad("X", 80) + pad("X", 80) + pad(start_date, 8) + pad(start_time, 8)
           + pad(256 * (ns + 1), 8) + pad("", 44) + pad(n_rec, 8) + pad(1, 8) + pad(ns, 4))
    for w, val in [(16, None), (80, ""), (8, "uV"), (8, -phys_range), (8, phys_range),
                   (8, -32768), (8, 32767), (80, ""), (8, fs), (32, "")]:
        for i in range(ns):
            hdr += pad(labels[i] if val is None else val, w)
    dig = np.clip(np.round((signals[:, : n_rec * fs] + phys_range) / (2 * phys_range) * 65535 - 32768),
                  -32768, 32767).astype("<i2")
    data = dig.reshape(ns, n_rec, fs).transpose(1, 0, 2).reshape(-1)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(data.tobytes())
