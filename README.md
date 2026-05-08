# 🧠 Synchrony-Driven Seizure Prediction via ST-GNN

> A Spatio-Temporal Graph Neural Network with soft-label supervision
> for early epileptic seizure prediction on the CHB-MIT scalp EEG database.

[![Python](https://img.shields.io/badge/Python-3.9+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)]()
[![PyG](https://img.shields.io/badge/PyTorch--Geometric-2.3+-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Stevens](https://img.shields.io/badge/Stevens-AAI%20Program-8C1515)]()

---

## 1. Project Overview

Epileptic seizures affect over 50 million people worldwide. For patients
with drug-refractory epilepsy, unpredictability is the primary stressor —
not the seizure itself, but not knowing when it will happen.

This project proposes a **Synchrony-Driven Spatio-Temporal Graph Neural
Network (ST-GNN)** that predicts seizures by tracking the gradual buildup
of inter-electrode synchrony in the brain's EEG signal before onset.

**The core problem with existing systems:**
Binary classifiers treat every preictal window identically — a window
28 minutes before onset gets the same label as one 30 seconds before.
This ignores the continuous biological reality of seizure progression.

**What this model does differently:**
Instead of a binary alarm, it outputs a **continuous risk score** from
0 to 1 that rises progressively as the brain approaches seizure onset —
functioning as a *weather report for the brain*, not just an alarm.

**Key Results on CHB-MIT (24 subjects, cross-subject):**

| Metric          | Hard-Label Baseline | Soft-Label ST-GNN  |
|-----------------|--------------------|--------------------|
| ROC-AUC         | 0.807              | **0.883**          |
| Sensitivity     | 43.2%              | **66.5%**          |
| Specificity     | 94.3%              | 88.2%              |
| F1 Score        | 0.413              | **0.438**          |
| Mean Lead Time  | —                  | **17.4 ± 9.8 min** |
| Subjects Alerted| —                  | **23 / 24**        |

---

## 2. Key Features

- **Dynamic PLV Graph Construction** — converts each 5-second EEG window
  into a Phase Locking Value connectivity graph capturing inter-electrode
  synchrony
- **Dual-Branch Architecture** — GATv2 spatial branch + TCN temporal
  branch running in parallel and fused into a single risk score
- **Soft-Label Supervision** — continuous risk trajectory replaces binary
  labels, enabling the model to learn physiologically meaningful risk
  progression
- **Lead-Time Analysis** — first framework to report per-subject early
  warning lead times on CHB-MIT (mean 17.4 minutes)
- **Compact Model** — 295,250 parameters, ~1.2 MB, suitable for IoMT
  wearable deployment
- **Cross-Subject Evaluation** — one model trained and evaluated across
  all 24 subjects without patient-specific fine-tuning

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
GATv2 Spatial Branch      TCN Temporal Branch
  (8-head attention)        (3x Conv1d)
  z_s ∈ R^64               z_t ∈ R^64
    │                         │
    └────────┬────────────────┘
             │
      Feature Fusion (concat → Linear → GELU → Dropout)
             │
      Risk Score r_t ∈ [0, 1]
             │
      Seizure Risk Warning
```

**GATv2 Spatial Branch:**
- Node encoder: Conv1d(1→16→32→64) + AdaptiveAvgPool + Linear → 128-dim
- GAT Layer 1: GATv2Conv(128, 64, heads=8, concat=True) → 512-dim
- GAT Layer 2: GATv2Conv(512, 64, heads=1, concat=False) → 64-dim
- GlobalMeanPool → z_s ∈ R^64

**TCN Temporal Branch:**
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
Risk increases linearly from 0.10 to 1.0 across the preictal block.

**Training Loss (Joint Supervision):**
```
L = α × MSE(r̂_t, r̃_t) + (1-α) × BCE(r̂_t, y_t, ω)
```
- α = 0.5 — balances trajectory learning and boundary detection
- ω = 12 — positive class weight for class imbalance
- MSE enforces correct risk ordering within the preictal block
- BCE anchors the binary interictal/preictal separation

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
3. Exclude 5-minute postictal buffer
4. Label all remaining windows → interictal
5. Compute PLV adjacency matrix using Hilbert transform
6. Apply threshold θ = 0.3 to retain significant edges

**Dataset Statistics:**
| Split      | Total    | Interictal      | Preictal      |
|------------|----------|-----------------|---------------|
| Train      | 52,318   | 48,275          | 4,043         |
| Validation | 11,211   | 10,402          | 809           |
| Test       | 11,212   | 10,322          | 890           |
| **Total**  | **74,741** | **68,999 (92.3%)** | **5,742 (7.7%)** |

**Class Imbalance Mitigation:**
- Weighted random sampler during training
- Positive class weight ω = 12 in loss function

---

## 6. Installation

**Requirements:**
- Python 3.9+
- CUDA 11.8+ (recommended)
- 16GB RAM minimum

```bash
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse
pip install numpy scipy h5py scikit-learn matplotlib tqdm
```

**Or use requirements.txt:**
```bash
pip install -r requirements.txt
```

---

## 7. Project Structure

```
seizure-prediction-stgnn/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── src/
│   ├── model.py          ← STGNN_Soft architecture
│   ├── dataset.py        ← CHBMIT_SoftDataset
│   ├── loss.py           ← SoftSeizureLoss
│   ├── preprocess.py     ← segmentation + PLV
│   ├── build_h5.py       ← HDF5 builder with soft labels
│   ├── train.py          ← training loop
│   ├── evaluate.py       ← test evaluation + metrics
│   └── utils.py          ← PLV, graph construction helpers
│
├── checkpoints/
│   └── best_chbmit_soft.pt   ← trained weights (best val AUC)
│
├── figures/
│   ├── per_subject_auc.png
│   ├── lead_time_per_subject.png
│   ├── preictal_vs_leadtime.png
│   └── dataset_statistics.png
│
└── notebooks/
    ├── 01_preprocessing.ipynb
    ├── 02_training.ipynb
    └── 03_evaluation.ipynb
```

---

## 8. Training Instructions

**Step 1 — Preprocess raw EDF files:**
```bash
python src/preprocess.py --data_dir data/chb-mit --out_dir data/processed
```

**Step 2 — Build HDF5 datasets with soft labels:**
```bash
python src/build_h5.py --processed_dir data/processed --out_dir data/h5_soft
```

**Step 3 — Train the model:**
```bash
python src/train.py \
    --h5_dir data/h5_soft \
    --epochs 60 \
    --lr 3e-4 \
    --batch_size 64 \
    --alpha 0.5 \
    --pos_weight 12.0 \
    --save_path checkpoints/best_chbmit_soft.pt
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
| Scheduler      | Cosine| CosineAnnealingLR (T_max=60) |

**Expected training time:** ~20 minutes on NVIDIA RTX 5090

---

## 9. Evaluation Instructions

```bash
python src/evaluate.py \
    --h5_dir data/h5_soft \
    --checkpoint checkpoints/best_chbmit_soft.pt \
    --threshold 0.75
```

**Output includes:**
- ROC-AUC, Sensitivity, Specificity, F1 Score
- Confusion matrix (normalized)
- ROC curve plot
- Per-subject metrics breakdown
- Lead-time analysis per subject

---

## 10. Threshold Selection

Two thresholds are used in this work:

| Model               | Threshold | Selected by          |
|---------------------|-----------|----------------------|
| Hard-Label Baseline | τ = 0.80  | Max F1 on val set    |
| Soft-Label ST-GNN   | τ = 0.75  | Max F1 on val set    |

```python
thresholds = np.arange(0.1, 0.95, 0.05)
best_tau, best_f1 = 0.5, 0.0

for tau in thresholds:
    preds = (val_probs >= tau).astype(int)
    f1    = f1_score(val_labels, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, best_tau = f1, tau

print(f"Best threshold: {best_tau:.2f} | Val F1: {best_f1:.4f}")
```

---

## 11. Results

### Overall Performance
| Metric      | Hard-Label | Soft-Label | Gain    |
|-------------|------------|------------|---------|
| ROC-AUC     | 0.807      | **0.883**  | +7.6%   |
| Sensitivity | 43.2%      | **66.5%**  | +23.3%  |
| Specificity | 94.3%      | 88.2%      | -6.1%   |
| F1 Score    | 0.413      | **0.438**  | +2.5%   |

### Patient-Specific Average
| Metric      | Value           |
|-------------|-----------------|
| ROC-AUC     | 0.884 ± 0.080   |
| Sensitivity | 59.1% ± 25.7%   |
| Specificity | 87.8% ± 13.5%   |
| F1 Score    | 0.401 ± 0.188   |

### Lead-Time Analysis
| Metric              | Value           |
|---------------------|-----------------|
| Subjects with alert | 23 / 24         |
| Mean lead time      | 17.4 ± 9.8 min  |
| Min lead time       | 1.0 min (chb11) |
| Max lead time       | 54.8 min (chb12)|
| Pearson r           | 0.999           |

---

## 12. Inference / Usage

```python
import torch
from src.model import STGNN_Soft
from src.utils import compute_plv, window_to_graph
from torch_geometric.data import Batch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = STGNN_Soft().to(DEVICE)
ckpt  = torch.load('checkpoints/best_chbmit_soft.pt', map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()

# Prepare input (23 channels, 1280 samples = 5 seconds at 256 Hz)
eeg   = load_eeg_window()        # shape: (23, 1280)
plv   = compute_plv(eeg)         # shape: (23, 23)
graph = window_to_graph(eeg, plv)
batch = Batch.from_data_list([graph]).to(DEVICE)

# Inference
with torch.no_grad():
    risk_score = torch.sigmoid(model(batch)).item()

# Decision
THRESHOLD = 0.75
if risk_score >= THRESHOLD:
    print(f"⚠️  SEIZURE ALERT — Risk: {risk_score:.3f}")
else:
    print(f"✅  Safe — Risk: {risk_score:.3f}")
```

---

## 13. Model Files

| File | Description |
|------|-------------|
| `checkpoints/best_chbmit_soft.pt` | Best weights (val AUC=0.8881, epoch 56) |
| `src/model.py` | STGNN_Soft class definition |
| `src/loss.py` | SoftSeizureLoss — MSE + BCE |

```python
ckpt = torch.load('best_chbmit_soft.pt')
# ckpt['epoch']       → 56
# ckpt['val_auc']     → 0.8881
# ckpt['model_state'] → OrderedDict of weights
```

---

## 14. Limitations

- **Class imbalance** — 92.3% interictal vs 7.7% preictal contributes
  to low F1 (0.438) and high sensitivity variance across subjects
- **Offline evaluation only** — not validated in real-time streaming
- **Single dataset** — CHB-MIT only, no generalization validated
- **Broadband PLV** — frequency-specific filtering may improve results
- **Lead-time dependency** — strongly correlated with preictal data
  availability (Pearson r = 0.999) rather than model sensitivity alone

---

## 15. Clinical Interpretation

**Why sensitivity matters more than specificity:**

| Error Type | Clinical Impact |
|---|---|
| False Negative (missed seizure) | Patient falls, gets injured, loses consciousness |
| False Positive (false alarm) | Patient takes unnecessary precaution — nothing happens |

The soft-label model intentionally accepts lower specificity (88.2%)
in exchange for higher sensitivity (66.5%) — missing a seizure is far
more dangerous than a false alarm.

**The graded output advantage:**
The continuous risk score enables tiered clinical interventions:
- r < 0.50 — normal monitoring
- 0.50 ≤ r < 0.75 — caregiver notification
- r ≥ 0.75 — emergency alert

---

## 16. Reproducibility Notes

**Hardware:** NVIDIA RTX 5090 | 32GB RAM
**Training time:** ~20 minutes (60 epochs)
**Software:** Python 3.10 | PyTorch 2.0.1 | CUDA 11.8 | BF16 mixed precision

```python
import torch, numpy as np, random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
```

**Expected results (±0.005 due to GPU non-determinism):**
- Val AUC at epoch 56: 0.8881
- Test AUC: 0.883 | Sensitivity: 66.5%

---

## 17. License

MIT License — Copyright (c) 2025 Mauricio Castro Aviles

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software to deal in the Software without restriction,
including the rights to use, copy, modify, merge, publish, distribute,
and sell copies of the Software.

---

## 18. Citation

```bibtex
@article{castro2025stgnn,
  author  = {Castro Aviles, Mauricio},
  title   = {Synchrony-Driven {AI} for Seizure Prediction via
             Spatio-Temporal Graph Neural Networks},
  journal = {Stevens Institute of Technology,
             M.S. Applied Artificial Intelligence},
  year    = {2025},
  note    = {CHB-MIT, ROC-AUC=0.883, Lead Time=17.4 min}
}
```

---

*Mauricio Castro Aviles — Stevens Institute of Technology, May 2025*
*M.S. Applied Artificial Intelligence*
