# 🧠 Synchrony-Driven Seizure Prediction via ST-GNN

> A Spatio-Temporal Graph Neural Network with soft-label supervision
> for early epileptic seizure prediction on the CHB-MIT scalp EEG database.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]()
[![PyG](https://img.shields.io/badge/PyTorch--Geometric-2.3+-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Stevens](https://img.shields.io/badge/Stevens-AAI%20Program-8C1515)]()

---
## 0. Motivation

To anticipate and predict seizures minutes before patients lose self control.

## 1. Project Overview

This project proposes a **Synchrony-Driven Spatio-Temporal Graph Neural
Network (ST-GNN)** that predicts seizures by tracking the gradual buildup
of inter-electrode synchrony in the brain's EEG signal before onset.

**What this model does differently:**
The framework outputs a **continuous risk score** from
0 to 1 that rises progressively as the brain approaches seizure onset.

> **⚠️ Evaluation update (v3).** The results originally reported for v2 were produced
> with a window-level random split, which leaks information between training and test
> data. They are kept in [Section 12](#12-v2-results-superseded) for transparency but
> should not be cited. Section 11 describes the corrected, leakage-free protocol and its
> results. See [Section 10](#10-what-changed-in-v3-and-why) for what was wrong and how it was fixed.

---

## 2. Key Features

- **Dynamic PLV Graph Construction** — converts each 5-second EEG window
  into a Phase Locking Value connectivity graph capturing inter-electrode
  synchrony.
- **Dual-Branch Architecture** — GATv2 spatial branch + temporal convolution
  branch running in parallel and fused into a single risk score.
- **Soft-Label Supervision** — continuous risk trajectory replaces binary
  labels, so the model learns risk that rises toward onset.
- **Leakage-free evaluation** — leave-one-patient-out (cross-subject) and
  forward-in-time patient-specific protocols.
- **Seizure-level metrics** — alarm smoothing, per-seizure sensitivity,
  false alarms per hour, per-seizure lead time, and a comparison against a
  random predictor with the same false-alarm rate.

---

## 3. Model Architecture

```
Multichannel EEG (23 channels, 256 Hz)
         |
    Segmentation (5-second non-overlapping windows)
         |
    ┌────┴────────────────────┐
    │                         │
PLV Graph Construction    Raw EEG Channel Mean
    │                         │
GATv2 Spatial Branch      Temporal Conv Branch
  (8-head attention)        (3x Conv1d)
  z_s ∈ R^64               z_t ∈ R^64
    │                         │
    └────────┬────────────────┘
             │
      Feature Fusion (concat → Linear → GELU → Dropout)
             │
      Risk Score r_t ∈ [0, 1]
             │
      Alarm logic (k-of-n smoothing + refractory period)
```

**GATv2 Spatial Branch:**
- Node encoder: Conv1d(1→16→32→64) + AdaptiveAvgPool + Linear → 128-dim
- GAT Layer 1: GATv2Conv(128, 64, heads=8, concat=True) → 512-dim
- GAT Layer 2: GATv2Conv(512, 64, heads=1, concat=False) → 64-dim
- GlobalMeanPool → z_s ∈ R^64

**Temporal Branch:**
- Input: mean across 23 channels → (B, 1, 1280)
- Conv1d(1→32, k=3) → ReLU → BatchNorm
- Conv1d(32→64, k=3) → ReLU → BatchNorm
- Conv1d(64→64, k=3) → ReLU → BatchNorm
- AdaptiveAvgPool → Linear(64→64) → z_t ∈ R^64

**Fusion:**
- concat(z_s, z_t) → Linear(128→64) → GELU → Dropout(0.4) → Linear(64→1)
- Sigmoid applied at inference only

**Total parameters:** 295,250 | **Model size:** ~1.2 MB

---

## 4. Soft vs Hard Labels

### Hard-Label (Baseline)
```
Interictal window → label = 0
Preictal window   → label = 1  (all preictal windows treated identically)
```
Problem: A window 30 minutes before seizure and one 10 seconds before
both get label = 1. The model cannot learn temporal risk progression.

### Soft-Label (Proposed)
```
Interictal window → risk = 0.0
Preictal window i → risk = 0.10 + 0.90 × (i / N-1)
```
Risk increases linearly from 0.10 to 1.0 across each seizure's preictal block.

**Training Loss (Joint Supervision):**
```
L = α × MSE(σ(z_t), r̃_t) + (1-α) × BCE(z_t, y_t, ω)
```
- z_t is the model's logit, σ the sigmoid
- α = 0.5 — balances trajectory learning and boundary detection
- ω = 12 — positive class weight for class imbalance
- MSE enforces correct risk ordering within the preictal block
- BCE anchors the binary interictal/preictal separation

(v2 applied the MSE term to the raw logit rather than σ(z_t); `--mse_on_logits`
reproduces that behaviour.)

---

## 5. Dataset Description

**CHB-MIT Scalp EEG Database**
- Source: PhysioNet — https://physionet.org/content/chbmit/1.0.0/
- 22 pediatric patients organized into 24 cases
- 950+ hours of continuous EEG, 198 annotated seizures
- 23 channels, 256 Hz sampling rate, 16-bit resolution

**Preprocessing:**
1. Segment into non-overlapping 5-second windows (T = 1280 samples)
2. Label windows within 30 minutes before onset → preictal
3. Exclude ictal windows and a 5-minute postictal buffer
4. Label all remaining windows → interictal
5. Compute PLV adjacency matrix using the Hilbert transform
6. Apply threshold θ = 0.3 to retain significant edges

Processed data is expected at `data/processed/<subject>/segments.npz`
(keys `X`: windows, `y`: labels) with optional cached `A_plv.npy`.

**Windows used:** 74,741 total — 68,999 interictal (92.3%), 5,742 preictal (7.7%).

**Class Imbalance Mitigation:**
- Weighted random sampler during training
- Positive class weight ω = 12 in the loss function

---

## 6. Installation

**Requirements:** Python 3.10+, an NVIDIA GPU recommended, 32 GB RAM for the full dataset.

```bash
# RTX 50-series (Blackwell) GPUs need a CUDA 12.8+ PyTorch build:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

## 7. Project Structure

```
seizure-prediction-stgnn/
├── README.md
├── requirements.txt
├── LICENSE
├── src/
│   ├── config.py            ← hyperparameters
│   ├── data.py              ← per-subject loading, seizure blocks, soft labels, PLV
│   ├── model.py             ← ST-GNN, graph builder, loss, checkpoint loading
│   ├── splits.py            ← leave-one-patient-out and chronological splits
│   ├── metrics.py           ← window and seizure-level metrics, alarm logic
│   ├── train.py             ← train + save predictions per fold
│   ├── evaluate.py          ← threshold selection on validation, final metrics
│   └── inspect_segments.py  ← checks windows are in recording order
├── tests/
│   └── test_eval.py         ← unit tests for splits and metrics
├── checkpoints/
│   ├── best_chbmit_soft.pt  ← v2 weights (trained with the leaky split)
│   └── history_soft.pt
├── chbmit_stgnn_v2.ipynb                          ← v2 exploration notebook
└── chbmit_stgnn_v1_baseline_binary_predictor.ipynb ← hard-label baseline
```

---

## 8. Usage

**Step 0 — Check that windows are stored in recording order** (event metrics depend on it):
```bash
python -m src.inspect_segments --processed_dir data/processed
```

**Step 1 — Train, leave-one-patient-out** (one model per held-out subject):
```bash
python -m src.train --processed_dir data/processed --protocol lopo --out_dir results/lopo
# quick look at a few folds:
python -m src.train --processed_dir data/processed --protocol lopo --folds chb01 chb05 --epochs 20
```

**Or train patient-specific, forward in time** (earliest seizures → train, next → validation, later → test):
```bash
python -m src.train --processed_dir data/processed --protocol chrono --out_dir results/chrono
```

**Step 2 — Evaluate:**
```bash
python -m src.evaluate --results_dir results/lopo                       # τ for ≤ 0.5 false alarms/h on validation
python -m src.evaluate --results_dir results/lopo --target_fpr 0.15     # stricter, clinically common target
```
Writes `results.md`, `results.json`, `per_subject.csv` and `per_seizure.csv`.

**Tests:**
```bash
python -m pytest tests
```

**Key hyperparameters:**
| Parameter      | Value | Description                  |
|----------------|-------|------------------------------|
| Epochs         | 60    | Training duration            |
| Learning rate  | 3e-4  | Adam optimizer               |
| Weight decay   | 1e-4  | L2 regularization            |
| α (alpha)      | 0.5   | MSE/BCE trade-off            |
| ω (pos_weight) | 12.0  | Preictal class weight        |
| Dropout        | 0.4   | Fusion layer dropout         |
| Grad clip      | 1.0   | Gradient clipping            |
| Scheduler      | Cosine| CosineAnnealingLR            |

---

## 9. Evaluation Protocol

**Splits — no test window is ever seen, or neighboured, in training:**
- *Leave-one-patient-out (cross-subject):* train on 20 subjects, choose the epoch and
  alarm threshold on 3 other subjects, test on the held-out subject. Repeated for every
  subject with at least one seizure.
- *Chronological (patient-specific):* within one subject, train on the earliest
  seizures, validate on the next one, and test on later seizures and the interictal
  data after them, with a buffer at each cut. Subjects with fewer than 3 seizures are skipped.

**From risk scores to alarms:**
- An alarm fires when at least 3 of the last 5 windows (within one continuous recording)
  exceed τ; further alarms are suppressed for 30 minutes.
- τ is chosen on validation subjects only, to maximise seizure sensitivity subject to a
  false-alarm-rate target.

**Metrics:**
- **Seizure sensitivity** — fraction of seizures with an alarm inside their 30-minute preictal period.
- **False alarms per hour** — alarms during interictal data / hours of interictal data.
- **Lead time** — per seizure, from the first alarm to onset (never summed across seizures).
- **Random-predictor comparison** — a predictor raising alarms at random with the same
  false-alarm rate predicts a seizure with probability p = 1 − exp(−FPR × SOP)
  (Schelter et al., 2006). The binomial p-value tests whether the model beats it.
- Window-level ROC-AUC is still reported, per subject.

---

## 10. What changed in v3, and why

Reviewing v2 surfaced several methodological problems:

1. **Window-level random split.** All windows from all subjects were shuffled together
   and split 70/15/15. Every subject appeared in train and test, and neighbouring
   5-second windows from the same preictal period landed on both sides. The model
   could match recordings rather than learn pre-seizure patterns, and the results were
   not cross-subject as claimed.
2. **Per-subject and lead-time analyses ran on data the model had trained on** (all of
   each subject's windows, ~70% of them in the training set).
3. **Lead time was summed across seizures.** All of a subject's preictal windows were
   treated as one block, so lead time grew with the number of seizures, which explains
   values above the 30-minute horizon (54.8 min) and the r = 0.999 correlation with
   preictal window count.
4. **No false-alarm rate or chance comparison**, the standard metrics in seizure-prediction work.
5. **MSE term applied to the logit**, comparing an unbounded value to a 0–1 target.

v3 fixes each one: subject-held-out and forward-in-time splits, threshold and model
selection on validation subjects only, per-seizure event metrics, false alarms per hour,
a random-predictor test, and the corrected loss.

---

## 11. Results (v3, leakage-free)

<!-- Paste the contents of results/lopo/results.md (and results/chrono/results.md) here. -->

*Re-evaluation in progress. Cross-patient prediction on CHB-MIT is substantially harder
than the v2 numbers suggested; results will be reported here as produced by
`src.evaluate`, including subjects where the model does not beat chance.*

---

## 12. v2 Results (superseded)

Produced with the window-level random split described in Section 10. Kept for
transparency only; these numbers overstate performance.

| Metric      | Hard-Label | Soft-Label |
|-------------|------------|------------|
| ROC-AUC     | 0.807      | 0.883      |
| Sensitivity (window) | 43.2% | 66.5% |
| Specificity (window) | 94.3% | 88.2% |
| F1 Score    | 0.413      | 0.438      |

---

## 13. Inference

```python
import torch
from torch_geometric.data import Batch
from src.data import compute_plv_matrix
from src.model import STGNN_Soft, load_checkpoint, window_to_graph

model = STGNN_Soft()
load_checkpoint(model, "results/lopo/chb01/model.pt")
model.eval()

eeg = load_eeg_window()                  # (23, 1280): 5 s at 256 Hz
graph = window_to_graph(eeg, compute_plv_matrix(eeg), label=0)
with torch.no_grad():
    risk = torch.sigmoid(model(Batch.from_data_list([graph]))).item()
```

A single window's risk is not an alarm; use `src.metrics.raise_alarms` on the stream of
risk scores so isolated spikes do not trigger warnings.

---

## 14. Limitations

- **Class imbalance** — 92.3% interictal vs 7.7% preictal.
- **Interictal data is a subset** of the full recordings, and interictal windows close
  to seizures are not excluded beyond the postictal buffer; reported false-alarm rates
  apply to the interictal data evaluated.
- **Offline evaluation only** — not validated in real-time streaming.
- **Single dataset** — CHB-MIT only; pediatric scalp EEG.
- **Broadband PLV** — frequency-specific PLV (e.g. beta/gamma) may carry more signal.
- **Temporal branch uses the channel mean**, discarding per-channel temporal detail.

---

## 15. Clinical Interpretation

Missed seizures and false alarms both carry costs: a missed seizure risks injury, while
frequent false alarms cause alarm fatigue and lead patients to ignore warnings. That is
why results are reported as a sensitivity / false-alarms-per-hour trade-off at an explicit
target rather than at a single window-level threshold.

The continuous risk score supports tiered responses (e.g. monitoring → caregiver
notification → emergency alert), with each tier's threshold chosen on validation data.

---

## 16. Reproducibility

- Seeds fixed (`--seed 42`); results may vary slightly with GPU non-determinism.
- Hardware used: NVIDIA RTX 5090, 32 GB RAM, BF16 mixed precision.
- Record exact package versions with `pip freeze > environment.txt` when reporting results.

---

## 17. Citation

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

---

## License
This project is licensed under the MIT License — see the LICENSE file for details.

---

*Mauricio Castro Aviles — Stevens Institute of Technology, May 2026*
*M.S. Applied Artificial Intelligence*
