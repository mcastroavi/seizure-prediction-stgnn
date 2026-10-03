"""Default hyperparameters (identical to the values used in chbmit_stgnn_v2.ipynb)."""

CFG = {
    # Signal
    "fs": 256,
    "window_sec": 5.0,
    "n_time": 1280,
    # PLV graph
    "plv_threshold": 0.3,
    # Model
    "node_feat": 128,
    "gat_out": 64,
    "gat_heads": 8,
    "dropout": 0.4,
    # Training
    "epochs": 60,
    "batch_size": 128,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "alpha": 0.5,        # MSE / BCE trade-off
    "pos_weight": 12.0,  # preictal class weight in BCE
    "seed": 42,
    # Labelling
    "preictal_min": 30,  # seizure occurrence period (SOP) used for the random-predictor baseline
}
