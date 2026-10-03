# 🧠 Synchrony-Driven Seizure Prediction via ST-GNN

> A Spatio-Temporal Graph Neural Network with soft-label supervision for early epileptic
> seizure prediction on the CHB-MIT scalp EEG database, evaluated the way a deployed
> warning system would be judged.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]()
[![PyG](https://img.shields.io/badge/PyTorch--Geometric-2.5+-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Stevens](https://img.shields.io/badge/Stevens-AAI%20Program-8C1515)]()

---

## 0. Overview

**Goal:** warn patients minutes before a seizure, from scalp EEG.

**Idea:** in the minutes before a seizure, brain regions become increasingly
phase-synchronised. Each 5-second EEG window is turned into a graph whose nodes are
electrode channels and whose edges are Phase Locking Values (PLV). A graph attention
network reads that synchrony pattern, a temporal branch reads the waveform, and the model
outputs a **continuous risk score** that rises toward onset.

**v3 is a full rebuild.** The original (v2) results used a window-level random split that
leaked information between training and test data (see [Section 8](#8-what-changed-from-v2-and-why)).
v3 rebuilds every step, from raw EDF files to seizure-level metrics, as tested,
reproducible code:

- leakage-free **leave-one-patient-out** and **forward-in-time** evaluation
- **seizure-level metrics**: sensitivity, false alarms per hour, lead time, and a
  significance test against a random predictor
- a **logistic-regression baseline** on the same features and folds
- 30 unit and end-to-end tests, runnable on CPU without the dataset

---

## 1. Pipeline

```
 raw EDF + summary files
        │  src/preprocess.py
        ▼
 18-channel bipolar EEG ─► band-pass 0.5–40 Hz ─► 128 Hz ─► 5-s windows ─► PLV graphs
 every window kept, with absolute timestamps and true seizure ids
        │  src/train.py  (or src/baseline.py)
        ▼
 risk score r_t ∈ [0,1] for every window of each held-out recording
        │  src/evaluate.py
        ▼
 alarms (3-of-5 windows ≥ τ, 30-min refractory) ─► per-seizure sensitivity,
 false alarms / hour, lead time, p-value vs. random predictor
        │  src/figures.py
        ▼
 risk timelines, per-subject results, sensitivity vs. false-alarm curve
```

---

## 2. Model

```
Window: 18 channels × 640 samples (5 s at 128 Hz), z-scored per channel
    │
    ├── PLV graph (edges: PLV > 0.3) ──► GATv2 spatial branch ──► z_s ∈ R^64
    │      node encoder: Conv1d 1→16→32→64, pool, Linear → 128
    │      GATv2Conv(128→64, 8 heads) → GATv2Conv(512→64) → mean pool
    │
    └── channel-mean signal ──► temporal branch ──► z_t ∈ R^64
           3 × [Conv1d → ReLU → BatchNorm], pool, Linear
    │
    concat(z_s, z_t) → Linear(128→64) → GELU → Dropout(0.4) → Linear(64→1) → logit
```

295k parameters (~1.2 MB).

**Soft labels.** A binary label treats a window 30 minutes before onset the same as one 10
seconds before. Instead, preictal windows get a target that rises with proximity to onset:

```
r̃_t = 0.10 + 0.90 × (1 − time_to_onset / SOP)          (interictal: r̃_t = 0)
L   = α · MSE(σ(z_t), r̃_t) + (1 − α) · BCE(z_t, y_t),   α = 0.5
```

The ramp is defined by **time to onset**, so a preictal stretch that is cut short by the
start of a recording still gets the targets that match how close it is to the seizure.

---

## 3. Data and preprocessing

**CHB-MIT Scalp EEG Database** ([PhysioNet](https://physionet.org/content/chbmit/1.0.0/)):
22 pediatric patients in 24 cases, over 900 hours of continuous recording, 198 annotated seizures.

| Decision | Choice | Why |
|---|---|---|
| Channels | 18 bipolar double-banana channels | Present in almost every file; v2's 23 header channels include a duplicate and vary across subjects. Files recorded with a referential montage are converted by subtracting references when all electrodes are present; any file that cannot be converted is skipped and listed in `report.json`. |
| Filtering / rate | 0.5–40 Hz zero-phase Butterworth, resampled to 128 Hz | Keeps all content below 40 Hz; halves storage and compute versus 256 Hz. |
| Windows | 5 s, non-overlapping, **every window kept** in recording order | Evaluation runs on the full continuous recordings, so false alarms per hour are measured on all interictal hours. |
| Timeline | Absolute time from summary clock times (EDF header when missing), with midnight rollover | Preictal periods that cross file boundaries are labelled correctly. |
| Preictal | 30 min before onset (seizure occurrence period, SOP) | Standard horizon in the literature. |
| Postictal | 5 min after offset, excluded | Post-seizure EEG is neither normal nor preictal. |
| Interictal | ≥ 60 min from any seizure | Keeps "normal" data clearly separated from seizure-related activity. |
| PLV | Computed on the band-passed signal | Hilbert phase is meaningful on a band-limited signal; v2 computed it on raw EEG. |
| Normalisation | Per-window, per-channel z-score | Removes amplitude differences between patients and electrodes, which matters when testing on unseen patients. |
| Class balance | Each training epoch draws 50% preictal / 50% interictal windows | Interictal windows far outnumber preictal ones in the full recordings. |

---

## 4. Evaluation protocol

**Splits.** No test window, or its neighbour in time, is ever seen in training.

- **Leave-one-patient-out (cross-subject):** train on 20 subjects, pick the epoch and the
  alarm threshold on 3 other subjects, test on the held-out subject. One fold per subject
  with seizures.
- **Chronological (patient-specific):** within one subject, train on the first ~50% of
  the recording, validate on the next ~20%, test on the last ~30%. Cuts never split a
  preictal period and each part has at least one seizure, so validation contains hours of
  interictal data for choosing the alarm threshold. Subjects with fewer than 3 seizures are skipped.

**From risk to alarms.** An alarm fires when at least 3 of the last 5 windows within one
continuous recording exceed τ; further alarms are suppressed for 30 minutes. τ is chosen
on validation subjects only, maximising seizure sensitivity subject to a false-alarm target
(default ≤ 0.5 per hour).

**Metrics.**

- **Seizure sensitivity:** fraction of seizures with an alarm in their 30-minute preictal period.
- **False alarms per hour:** alarms during interictal data ÷ interictal hours.
- **Lead time:** per seizure, from the first alarm to onset.
- **Random-predictor test:** a predictor raising alarms at random with the same false-alarm
  rate catches a seizure with probability p = 1 − exp(−FPR × SOP) (Schelter et al., 2006);
  a binomial test gives the p-value of the model's sensitivity against it.
- Seizures with < 10 min of recorded preictal data (e.g. right after a recording starts)
  are not scored, and the number left out is reported.
- Window-level ROC-AUC per subject is reported for comparison with other work.

---

## 5. Quick start

```bash
# RTX 50-series (Blackwell) GPUs need a CUDA 12.8+ PyTorch build
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python -m pytest tests                    # 30 tests, CPU, < 1 min, no dataset needed
bash run_all.sh /path/to/chb-mit          # everything below, end to end
```

Step by step:

```bash
python -m src.preprocess --raw_dir /path/to/chb-mit --out_dir data/processed_v3 --workers 4
python -m src.inspect_segments --processed_dir data/processed_v3

python -m src.baseline --processed_dir data/processed_v3 --protocol lopo --out_dir results/baseline_lopo
python -m src.train    --processed_dir data/processed_v3 --protocol lopo --out_dir results/stgnn_lopo

python -m src.evaluate --results_dir results/stgnn_lopo                  # ≤ 0.5 false alarms/h on validation
python -m src.evaluate --results_dir results/stgnn_lopo --target_fpr 0.15  # stricter target
python -m src.figures  --results_dir results/stgnn_lopo
```

A quick look at two folds: `python -m src.train ... --folds chb01 chb05 --epochs 10`.

To try the pipeline without the dataset:
`python -m tests.make_synthetic_chbmit --out data/synthetic` builds small synthetic EDF
files with CHB-MIT's quirks (duplicate channels, a referential-montage file, a summary
without file times, recordings crossing midnight).

---

## 6. Key hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Epochs | 30, early stopping (patience 8) on validation AUC | |
| Samples per epoch | 40,000 (class-balanced) | |
| Optimizer | Adam, lr 3e-4, weight decay 1e-4, cosine schedule | |
| Batch size | 128 | |
| α | 0.5 | MSE / BCE trade-off |
| Dropout | 0.4 | |
| Gradient clip | 1.0 | |
| Mixed precision | BF16 on CUDA | |

---

## 7. Results

<!-- After run_all.sh: paste results/*/results.md here and add the figures from results/*/figures/. -->

*Being regenerated with the v3 pipeline.* Results will be reported here for both models and
both protocols, as produced by `src.evaluate`, including subjects where the model does not
beat chance. Cross-patient prediction on CHB-MIT is a hard problem; the point of this
version is that the numbers mean what they claim.

| Model | Protocol | Seizure sensitivity | False alarms / h | Mean lead time | Beats chance (p < 0.05) | Window AUC |
|---|---|---|---|---|---|---|
| Logistic regression (PLV + power) | Leave-one-patient-out | — | — | — | — | — |
| ST-GNN | Leave-one-patient-out | — | — | — | — | — |
| Logistic regression (PLV + power) | Chronological | — | — | — | — | — |
| ST-GNN | Chronological | — | — | — | — | — |

---

## 8. What changed from v2, and why

| v2 problem | Consequence | v3 fix |
|---|---|---|
| All windows from all subjects shuffled, then split 70/15/15 | Every subject in train and test; neighbouring windows of the same preictal period on both sides. Results were not cross-subject, and overstated. | Leave-one-patient-out and forward-in-time splits |
| Per-subject and lead-time analyses run on all of each subject's windows | ~70% of evaluated windows had been used for training | Evaluation only on held-out data |
| Lead time summed over all of a subject's seizures | Values above the 30-min horizon (54.8 min); r = 0.999 with preictal window count | Lead time per seizure |
| Window-level metrics only | No false-alarm rate; no comparison with chance | Event metrics, false alarms per hour, random-predictor test |
| Preprocessing script not in the repo; windows without timestamps | Chronology and seizure boundaries could not be verified | `src/preprocess.py`: timestamped windows with true seizure ids |
| Interictal data subsampled (~3 h of ~40 h for chb01) | False-alarm rate cannot be measured honestly | Every window kept; training subsamples on the fly |
| MSE applied to the logit | Unbounded output compared with a 0–1 target | MSE on σ(logit) |
| No baseline | No way to tell whether the graph network adds value | Logistic regression on the same features and folds |

The v2 notebooks and checkpoint are kept in `legacy/` for reference. Their reported
numbers (window AUC 0.883) were produced with the leaky split and should not be cited.

---

## 9. Project structure

```
seizure-prediction-stgnn/
├── run_all.sh               ← reproduces every result from raw EDF files
├── src/
│   ├── edf.py               ← EDF reader (numpy)
│   ├── chbmit.py            ← summary parsing, timeline, montage, window labels
│   ├── preprocess.py        ← EDF → timestamped windows + PLV
│   ├── inspect_segments.py  ← data sanity report
│   ├── data.py              ← memory-mapped per-subject access, soft labels
│   ├── splits.py            ← leave-one-patient-out, chronological splits
│   ├── model.py             ← ST-GNN, graph builder, loss
│   ├── train.py             ← training + predictions per fold
│   ├── baseline.py          ← logistic-regression baseline
│   ├── metrics.py           ← alarms, seizure-level metrics, threshold selection
│   ├── evaluate.py          ← final metrics and tables
│   ├── figures.py           ← plots
│   └── config.py
├── tests/
│   ├── test_preprocess.py   ← EDF, summaries, timeline, montage, labels
│   ├── test_eval.py         ← splits, alarms, metrics
│   ├── test_pipeline.py     ← synthetic EDF → preprocess → baseline → evaluate
│   └── make_synthetic_chbmit.py
└── legacy/                  ← v2 notebooks and checkpoint (leaky evaluation)
```

---

## 10. Limitations

- **Single dataset:** pediatric scalp EEG from one hospital; generalisation to adults or
  other recording setups is untested.
- **Offline evaluation:** the pipeline is causal per window (alarms use only past
  windows), but it has not been run as a real-time stream.
- **Broadband PLV:** synchrony in specific bands (e.g. beta/gamma) may carry more signal.
- **Temporal branch uses the channel mean**, discarding per-channel temporal detail.
- **Fixed SOP of 30 minutes:** the right horizon may differ between patients.

---

## 11. Citation

```bibtex
@mastersthesis{castro2026stgnn,
  author = {Castro Aviles, Mauricio},
  title  = {Synchrony-Driven {AI} for Seizure Prediction via
            Spatio-Temporal Graph Neural Networks},
  school = {Stevens Institute of Technology},
  type   = {M.S. Applied Artificial Intelligence},
  year   = {2026}
}
```

## License
MIT — see the LICENSE file.

*Mauricio Castro Aviles — Stevens Institute of Technology, M.S. Applied Artificial Intelligence*
