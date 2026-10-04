#!/usr/bin/env bash
# Reproduce every result in the README from the raw CHB-MIT EDF files.
#
#   bash run_all.sh /path/to/chb-mit          # folder containing chb01/ ... chb24/
#
# Rough times on one RTX 5090 + 8 CPU cores: preprocessing ~15 min, each leave-one-patient-out
# ST-GNN run 1.5-3 h, context models and baselines minutes each, personalization ~20 min.
set -euo pipefail

RAW=${1:?usage: bash run_all.sh /path/to/chb-mit}
P=data/processed_v3

ev() {  # seizure-level metrics (all seizures, then lead seizures only) + figures
  python -m src.evaluate --results_dir "results/$1"
  python -m src.evaluate --results_dir "results/$1" --lead_gap_h 4 --processed_dir "$P" \
    --out_dir "results/$1/lead4h"
  python -m src.figures  --results_dir "results/$1"
}

python -m pytest -q tests                                              # tests (CPU, <1 min)
python -m src.preprocess --raw_dir "$RAW" --out_dir "$P" --workers 4   # EDF -> windows + features
python -m src.inspect_segments --processed_dir "$P"                    # sanity report

for PR in lopo chrono; do
  # baselines: logistic regression, without and with band features
  python -m src.baseline --processed_dir "$P" --protocol $PR --out_dir results/baseline_$PR
  python -m src.baseline --processed_dir "$P" --protocol $PR --features bands --out_dir results/baseline_bands_$PR
  # window encoders: ST-GNN, without and with band features
  python -m src.train --processed_dir "$P" --protocol $PR --out_dir results/stgnn_$PR
  python -m src.train --processed_dir "$P" --protocol $PR --features bands --out_dir results/stgnn_bands_$PR
  # 5-minute context models
  python -m src.context --processed_dir "$P" --protocol $PR --features bands --out_dir results/context_feat_$PR
  python -m src.context --processed_dir "$P" --source results/stgnn_$PR --out_dir results/context_stgnn_$PR
  python -m src.context --processed_dir "$P" --source results/stgnn_bands_$PR --out_dir results/context_stgnn_bands_$PR
  # context extras: time of day, seizure history, both
  for X in time:time hist:history timehist:time,history; do
    python -m src.context --processed_dir "$P" --source results/stgnn_bands_$PR \
      --extra ${X#*:} --out_dir results/context_stgnn_bands_${X%%:*}_$PR
  done
  for R in baseline baseline_bands stgnn stgnn_bands context_feat context_stgnn context_stgnn_bands \
           context_stgnn_bands_time context_stgnn_bands_hist context_stgnn_bands_timehist; do
    ev ${R}_$PR
  done
done

# augmentation (leave-one-patient-out)
python -m src.train --processed_dir "$P" --protocol lopo --features bands --augment --out_dir results/stgnn_bands_aug_lopo
python -m src.context --processed_dir "$P" --source results/stgnn_bands_aug_lopo --out_dir results/context_stgnn_bands_aug_lopo
ev stgnn_bands_aug_lopo
ev context_stgnn_bands_aug_lopo

# personalization: fine-tune the leave-one-patient-out models per patient
python -m src.personalize --processed_dir "$P" --general_encoder results/stgnn_bands_lopo \
  --general_context results/context_stgnn_bands_lopo --out_dir results/personalized
for V in general finetune_context finetune_full; do ev personalized/$V; done

# README figures
python tools/readme_figures.py export --results_dir results --out docs/figures/data.json
python tools/readme_figures.py render --data docs/figures/data.json --out_dir docs/figures
echo "Done. Results in results/*/results.md, README figures in docs/figures/."
