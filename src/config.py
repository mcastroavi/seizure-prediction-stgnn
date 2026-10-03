"""Default settings for the v3 pipeline."""

CFG = {
    # Signal (set by src.preprocess)
    "fs": 128,            # Hz after resampling (band-pass is 0.5-40 Hz, so 128 Hz loses nothing)
    "window_sec": 5.0,
    "n_channels": 18,     # standard bipolar double-banana montage
    # PLV graph
    "plv_threshold": 0.3,
    # Model
    "node_feat": 128,
    "gat_out": 64,
    "gat_heads": 8,
    "dropout": 0.4,
    # Training
    "batch_size": 128,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "alpha": 0.5,         # MSE / BCE trade-off
    "seed": 42,
}
