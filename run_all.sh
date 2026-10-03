#!/usr/bin/env bash
# Reproduce every result in the README from the raw CHB-MIT EDF files.
#
#   bash run_all.sh /path/to/chb-mit          # folder containing chb01/ ... chb24/
#
# Rough time estimates on one RTX 5090: preprocessing about an hour, baseline minutes,
# ST-GNN leave-one-patient-out several hours (one model per held-out subject).
set -euo pipefail

RAW=${1:?usage: bash run_all.sh /path/to/chb-mit}
PROC=data/processed_v3

python -m pytest -q tests                                                 # 0. tests (CPU, <1 min)
python -m src.preprocess --raw_dir "$RAW" --out_dir "$PROC" --workers 4   # 1. EDF -> windows
python -m src.inspect_segments --processed_dir "$PROC"                    # 2. sanity check

for P in lopo chrono; do
  python -m src.baseline --processed_dir "$PROC" --protocol $P --out_dir results/baseline_$P   # 3. baseline
  python -m src.train    --processed_dir "$PROC" --protocol $P --out_dir results/stgnn_$P      # 4. ST-GNN
  for R in baseline_$P stgnn_$P; do
    python -m src.evaluate --results_dir results/$R                       # 5. seizure-level metrics
    python -m src.figures  --results_dir results/$R                       # 6. figures
  done
done
echo "Paste results/*/results.md and figures into the README Results section."
