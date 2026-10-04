"""End-to-end test: synthetic EDF files -> preprocess -> baseline (LOPO) -> evaluate.

Runs on CPU in well under a minute and needs no GPU or deep-learning libraries.
"""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import baseline, evaluate, preprocess  # noqa: E402
from src.data import load_subject, runs_from_time, soft_risk_by_time  # noqa: E402
from tests import make_synthetic_chbmit  # noqa: E402


def test_runs_from_time_breaks_on_gaps():
    t = np.array([0, 5, 10, 30, 35, 40])
    assert runs_from_time(t, 5).tolist() == [0, 0, 0, 1, 1, 1]


def test_soft_risk_by_time_uses_time_to_onset():
    onsets = np.array([1800.0])
    t = np.array([0.0, 1795.0, 900.0, 5000.0])
    block = np.array([0, 0, 0, -1])
    r = soft_risk_by_time(t, block, onsets, 5.0, 1800.0)
    assert np.isclose(r[0], 0.10) and np.isclose(r[1], 1.0) and r[3] == 0
    assert 0.5 < r[2] < 0.6                       # halfway through the SOP


def test_full_pipeline_on_synthetic_edfs():
    with tempfile.TemporaryDirectory() as d:
        raw, proc, res = (os.path.join(d, x) for x in ("raw", "processed", "results"))
        make_synthetic_chbmit.main(["--out", raw])
        preprocess.main(["--raw_dir", raw, "--out_dir", proc, "--preictal_min", "10",
                         "--postictal_min", "2", "--buffer_min", "20"])

        # Seizure timing survives the midnight crossing and the file gaps (chb01)
        meta = np.load(os.path.join(proc, "chb01", "meta.npz"))
        expected = [22 * 3600 + 2 * 1205 + 900, 22 * 3600 + 5 * 1205 + 600]
        assert np.allclose(meta["seizures"][:, 0], expected)
        # Referential-montage file was converted, not skipped (chb02)
        assert json.load(open(os.path.join(proc, "chb02", "report.json")))["skipped"] == []

        d1 = load_subject(proc, "chb01")
        assert d1.X.shape[1:] == (18, 640) and d1.n_seizures == 2
        # band features: 5-band PLV and relative band power, aligned with the windows
        assert d1.has_bands and d1.plv_bands.shape[1:] == (5, 18, 18) and d1.bandpow.shape[1:] == (18, 5)
        pb, bp = d1.bands(0)
        assert np.all(np.asarray(pb) <= 1.001) and np.all(np.asarray(bp) <= 0.001)  # PLV <= 1, log rel power <= 0
        # the synthetic preictal signal is a 9 Hz oscillation: alpha-band PLV should rise most
        pre, inter = d1.hard == 1, d1.hard == 0
        iu = np.triu_indices(18, 1)
        alpha = np.asarray(d1.plv_bands[d1.sel], float)[:, 2][:, iu[0], iu[1]].mean(1)
        assert alpha[pre].mean() > alpha[inter].mean() + 0.05
        assert np.all(np.diff(d1.t_start) > 0)    # chronological

        out = os.path.join(res, "baseline")
        baseline.main(["--processed_dir", proc, "--protocol", "lopo", "--n_val", "1", "--out_dir", out])
        s = evaluate.evaluate(out, "fpr", 2.0, refractory_min=10, sop_min=10, min_preictal_min=3)
        assert s["n_test_subjects"] == 4 and s["seizures"] == 9
        assert s["sensitivity"] >= 0.75 and s["beats_chance_at_0.05"]

        out_b = os.path.join(res, "baseline_bands")
        baseline.main(["--processed_dir", proc, "--protocol", "chrono", "--features", "bands",
                       "--out_dir", out_b])
        assert os.path.exists(os.path.join(out_b, "chb03", "predictions.npz"))
        for f in ("results.json", "results.md", "per_subject.csv", "per_seizure.csv"):
            assert os.path.exists(os.path.join(out, f))
