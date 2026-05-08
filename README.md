# seizure-prediction-stgnn
Synchrony-Driven Seizure Prediction via Spatio-Temporal Graph Neural Networks
# Synchrony-Driven Seizure Prediction via ST-GNN

[![Python](https://img.shields.io/badge/Python-3.9+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

## Overview
Seizure prediction framework using Spatio-Temporal Graph Neural Networks with soft-label supervision on the CHB-MIT scalp EEG database (Version 1.5)

Results on CHB-MIT (24 subjects)
| Metric           | Value |
| ROC-AUC          | 0.883 |
| Sensitivity      | 66.5% |
| Mean Lead Time   | 17.4 ± 9.8 min |
| Subjects Alerted | 23/24 |

## Architecture ##
- **PLV Graph Construction** — dynamic inter-electrode 
  synchrony graph
- **GATv2 Spatial Branch** — 8-head attention on PLV graph
- **TCN Temporal Branch** — 3-layer Conv1d on raw EEG
- **Soft-Label Supervision** — continuous risk trajectory

## Requirements ##
```bash
pip install torch torch-geometric numpy scipy h5py scikit-learn
```

## Database ## 
CHB-MIT Scalp EEG Database — available at PhysioNet:
https://physionet.org/content/chbmit/1.0.0/
Total Size: 42.6 GB (uncompressed ZIP file).
Total Files: 664 .edf recordings.
Subjects: 22 pediatric subjects across 23 cases.
Seizures: 198 total seizures annotated within 129 files.
## Usage ##
```python
# Load trained model
import torch
model = STGNN_Soft().to(device)
ckpt  = torch.load('best_chbmit_soft.pt')
model.load_state_dict(ckpt['model_state'])
model.eval()

# Inference
with torch.no_grad():
    risk_score = torch.sigmoid(model(graph_batch))
```

## Author ##
Mauricio Castro Aviles
M.S. Applied Artificial Intelligence
Stevens Institute of Technology — May 2026
