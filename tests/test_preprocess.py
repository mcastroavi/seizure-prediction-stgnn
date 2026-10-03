"""Tests for the EDF reader, CHB-MIT parsing, timeline, montage and window labelling."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chbmit import (EXCLUDED, ICTAL, INTERICTAL, POSTICTAL, PREICTAL,  # noqa: E402
                        STANDARD_CHANNELS, apply_plan, build_timeline, channel_plan,
                        label_windows, normalize_label, parse_summary)
from src.edf import read_header, read_signals, write_edf  # noqa: E402

SUMMARY = """Data Sampling Rate: 256 Hz
*************************

Channels in EDF Files:
**********************
Channel 1: FP1-F7

File Name: chb01_03.edf
File Start Time: 13:43:04
File End Time: 14:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds

File Name: chb12_06.edf
File Start Time: 22:44:34
File End Time: 23:44:40
Number of Seizures in File: 2
Seizure 1 Start Time: 1665 seconds
Seizure 1 End Time: 1726 seconds
Seizure 2 Start Time: 3415 seconds
Seizure 2 End Time: 3447 seconds

File Name: chb12_38.edf
File Start Time: 03:11:04
File End Time: 4:11:04
Number of Seizures in File: 0
"""

SUMMARY_NO_TIMES = """File Name: chb24_01.edf
Number of Seizures in File: 2
Seizure Start Time: 480 seconds
Seizure End Time: 505 seconds
Seizure Start Time: 2451 seconds
Seizure End Time: 2476 seconds
"""


def test_parse_summary_formats():
    e = parse_summary(SUMMARY)
    assert e["chb01_03.edf"].start_tod == 13 * 3600 + 43 * 60 + 4
    assert e["chb01_03.edf"].seizures == [(2996, 3036)]
    assert e["chb12_06.edf"].seizures == [(1665, 1726), (3415, 3447)]
    assert e["chb12_38.edf"].end_tod == 4 * 3600 + 11 * 60 + 4     # single-digit hour
    assert e["chb12_38.edf"].seizures == []


def test_parse_summary_without_file_times():
    e = parse_summary(SUMMARY_NO_TIMES)["chb24_01.edf"]
    assert e.start_tod is None and e.seizures == [(480, 505), (2451, 2476)]


def test_timeline_rolls_over_midnight():
    # 22:00 (1 h), 23:00 (1 h), 00:00 next day, 01:00
    t = build_timeline([22 * 3600, 23 * 3600, 0, 3600], [3600] * 4)
    assert t == [22 * 3600, 23 * 3600, 24 * 3600, 25 * 3600]


def test_timeline_handles_hours_past_24_and_gaps():
    t = build_timeline([23 * 3600, 25 * 3600], [1800, 1800])  # summary writes 25:00:00
    assert t[1] - t[0] == 2 * 3600


def test_normalize_label():
    assert normalize_label("T8-P8-0") == "T8-P8"
    assert normalize_label(" fp1-f7 ") == "FP1-F7"
    assert normalize_label("FT9-FT10") == "FT9-FT10"


def test_channel_plan_direct_duplicates_and_missing():
    labels = STANDARD_CHANNELS[:14] + ["T8-P8", "-", "T8-P8"] + STANDARD_CHANNELS[15:]
    plan = channel_plan(labels)
    assert plan is not None and all(s[0] == "pick" for s in plan)
    assert plan[STANDARD_CHANNELS.index("T8-P8")] == ("pick", 14)
    assert channel_plan(STANDARD_CHANNELS[:-1]) is None


def test_channel_plan_derives_bipolar_from_referential():
    elecs = sorted({e for ch in STANDARD_CHANNELS for e in ch.split("-")})
    labels = [f"{e}-CS2" for e in elecs]
    plan = channel_plan(labels)
    assert plan is not None
    sig = np.arange(len(elecs), dtype=np.float32)[:, None] * np.ones((1, 4), np.float32)
    out = apply_plan(sig, plan)
    a, b = "F7", "T7"
    k = STANDARD_CHANNELS.index("F7-T7")
    assert np.allclose(out[k], elecs.index(a) - elecs.index(b))


def test_label_priorities_and_seizure_ids():
    w = 5.0
    t = np.arange(0, 3 * 3600, w)
    seizures = [(5000.0, 5060.0), (9000.0, 9030.0)]
    lab, sid = label_windows(t, w, seizures, preictal_sec=1800, postictal_sec=300, buffer_sec=3600)
    at = lambda s: lab[int(s // w)]
    assert at(5010) == ICTAL and at(9010) == ICTAL
    assert at(5100) == POSTICTAL
    assert at(5000 - 600) == PREICTAL and sid[int((5000 - 600) // w)] == 0
    assert at(9000 - 60) == PREICTAL and sid[int((9000 - 60) // w)] == 1
    assert at(5000 - 1800 - 300) == EXCLUDED          # inside 1 h buffer, outside preictal
    assert at(10500) == EXCLUDED
    assert lab[0] == INTERICTAL or (5000 - 3600) <= 0
    # a preictal window never overlaps onset
    pre = np.where(lab == PREICTAL)[0]
    for i in pre:
        on = seizures[sid[i]][0]
        assert t[i] + w <= on
    assert np.all(sid[lab != PREICTAL] == -1)


def test_close_seizure_preictal_belongs_to_next_seizure_and_ictal_wins():
    w = 5.0
    t = np.arange(0, 8000, w)
    seizures = [(3000.0, 3050.0), (3600.0, 3640.0)]    # second seizure 9 min after the first
    lab, sid = label_windows(t, w, seizures, 1800, 300, 3600)
    assert lab[int(3020 // w)] == ICTAL
    assert lab[int(3200 // w)] == POSTICTAL            # postictal of #0 beats preictal of #1
    assert lab[int(3500 // w)] == PREICTAL and sid[int(3500 // w)] == 1


def test_edf_roundtrip():
    rng = np.random.default_rng(0)
    fs, labels = 256, ["FP1-F7", "F7-T7", "T7-P7"]
    sig = (rng.standard_normal((3, fs * 10)) * 50).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.edf")
        write_edf(p, sig, fs, labels, start_time="13.43.04")
        h = read_header(p)
        assert h.labels == labels and h.n_records == 10 and h.start_tod_sec == 13 * 3600 + 43 * 60 + 4
        back = read_signals(p, h, [0, 2])
        assert back.shape == (2, fs * 10)
        assert np.max(np.abs(back - sig[[0, 2]])) < 0.1     # quantisation error only
