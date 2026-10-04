"""CHB-MIT specifics: summary files, recording timeline, montage, window labels.

Everything here is pure Python/numpy so it can be unit-tested without the dataset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# 18 bipolar channels of the standard 10-20 double-banana montage, present (directly or
# derivable) in nearly every CHB-MIT file. v2 used all 23 header channels, which include a
# duplicated T8-P8 and channels that vary between subjects.
STANDARD_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]

# Window label codes stored in meta.npz
INTERICTAL, PREICTAL, ICTAL, POSTICTAL, EXCLUDED = 0, 1, 2, 3, 4
LABEL_NAMES = {INTERICTAL: "interictal", PREICTAL: "preictal", ICTAL: "ictal",
               POSTICTAL: "postictal", EXCLUDED: "excluded"}


# ── Summary files ────────────────────────────────────────────────────────────

@dataclass
class FileEntry:
    name: str
    start_tod: int | None = None      # seconds since midnight (may be >= 86400 in some summaries)
    end_tod: int | None = None
    seizures: list[tuple[float, float]] = field(default_factory=list)  # seconds from file start


def _tod(text: str) -> int:
    h, m, s = (int(x) for x in text.strip().split(":"))
    return h * 3600 + m * 60 + s


_SZ_START = re.compile(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+(?:\.\d+)?)\s*seconds", re.I)
_SZ_END = re.compile(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+(?:\.\d+)?)\s*seconds", re.I)


def parse_summary(text: str) -> dict[str, FileEntry]:
    """Parse a ``chbXX-summary.txt`` file into {edf file name: FileEntry}."""
    entries: dict[str, FileEntry] = {}
    cur: FileEntry | None = None
    starts: list[float] = []
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("file name:"):
            cur = FileEntry(name=line.split(":", 1)[1].strip())
            entries[cur.name] = cur
            starts = []
        elif cur is None:
            continue
        elif line.lower().startswith("file start time:"):
            cur.start_tod = _tod(line.split(":", 1)[1])
        elif line.lower().startswith("file end time:"):
            cur.end_tod = _tod(line.split(":", 1)[1])
        elif (m := _SZ_START.search(line)):
            starts.append(float(m.group(1)))
        elif (m := _SZ_END.search(line)):
            cur.seizures.append((starts.pop(0), float(m.group(1))))
    return entries


# ── Timeline ─────────────────────────────────────────────────────────────────

def build_timeline(start_tods: list[int], durations: list[float]) -> list[float]:
    """Absolute start time (seconds from the first day's midnight) for each file.

    Files must be given in recording order. Clock times wrap at midnight, so whenever a
    file would start before the previous one ended, a day is added.
    """
    out, day, prev_end = [], 0, -np.inf
    for tod, dur in zip(start_tods, durations):
        t = day + tod % 86400
        while t < prev_end - 60:          # 60 s tolerance for clock jitter
            day += 86400
            t = day + tod % 86400
        out.append(float(t))
        prev_end = t + dur
    return out


# ── Montage ──────────────────────────────────────────────────────────────────

def normalize_label(label: str) -> str:
    lab = label.strip().upper().replace(" ", "")
    m = re.match(r"^([A-Z0-9]+-[A-Z0-9]+)-\d$", lab)   # "T8-P8-0" -> "T8-P8"
    return m.group(1) if m else lab


def channel_plan(labels: list[str], targets: list[str] = STANDARD_CHANNELS):
    """How to obtain each target bipolar channel from a file's channels.

    Returns a list with, per target, either ("pick", i) or ("diff", i, j) meaning
    signal[i] - signal[j] (e.g. F7-T7 = (F7-CS2) - (T7-CS2) for referential montages),
    or ``None`` if some target cannot be obtained.
    """
    norm = [normalize_label(x) for x in labels]
    first = {}
    for i, lab in enumerate(norm):
        first.setdefault(lab, i)                          # duplicates: keep the first
    ref_of = {}
    for i, lab in enumerate(norm):
        if "-" in lab:
            a, r = lab.split("-", 1)
            ref_of.setdefault((a, r), i)

    plan = []
    for tgt in targets:
        if tgt in first:
            plan.append(("pick", first[tgt]))
            continue
        a, b = tgt.split("-")
        refs = {r for (x, r) in ref_of if x == a} & {r for (x, r) in ref_of if x == b}
        if not refs:
            return None
        r = sorted(refs)[0]
        plan.append(("diff", ref_of[(a, r)], ref_of[(b, r)]))
    return plan


def apply_plan(signals: np.ndarray, plan) -> np.ndarray:
    out = np.empty((len(plan), signals.shape[1]), dtype=np.float32)
    for k, step in enumerate(plan):
        out[k] = signals[step[1]] if step[0] == "pick" else signals[step[1]] - signals[step[2]]
    return out


# ── Labels ───────────────────────────────────────────────────────────────────

def label_windows(t_start: np.ndarray, window_sec: float, seizures: list[tuple[float, float]],
                  preictal_sec: float = 1800, postictal_sec: float = 300,
                  buffer_sec: float = 3600, sph_sec: float = 0.0):
    """Label windows by absolute time.

    Priority: ictal > postictal > preictal > excluded > interictal.

    * preictal  : the whole window lies in [onset - SPH - SOP, onset - SPH)
    * excluded  : within ``buffer_sec`` of any seizure but not in the classes above, so
                  interictal data is clearly separated from seizures
    * seizure_id: for preictal windows, index of the upcoming seizure; -1 otherwise
    """
    t0 = np.asarray(t_start, dtype=float)
    t1 = t0 + window_sec
    labels = np.full(len(t0), INTERICTAL, dtype=np.int8)
    sz_id = np.full(len(t0), -1, dtype=np.int16)
    order = np.argsort([on for on, _ in seizures])
    sz = [seizures[i] for i in order]

    for on, off in sz:                                   # excluded (lowest priority first)
        labels[(t1 > on - buffer_sec) & (t0 < off + buffer_sec)] = EXCLUDED
    for k in range(len(sz) - 1, -1, -1):                 # preictal, nearest upcoming wins
        on, _ = sz[k]
        m = (t0 >= on - sph_sec - preictal_sec) & (t1 <= on - sph_sec)
        labels[m] = PREICTAL
        sz_id[m] = k
    for on, off in sz:
        m = (t0 < off + postictal_sec) & (t1 > off)
        labels[m] = POSTICTAL
    for on, off in sz:
        labels[(t0 < off) & (t1 > on)] = ICTAL
    sz_id[labels != PREICTAL] = -1
    return labels, sz_id
