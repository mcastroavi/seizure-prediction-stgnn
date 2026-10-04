"""Personalization: adapt the general (cross-patient) model to each patient.

    python -m src.personalize --processed_dir data/processed_v3 \\
        --general_encoder results/stgnn_bands_lopo --general_context results/context_stgnn_bands_lopo \\
        --out_dir results/personalized

For every patient with at least 3 seizures, the starting point is the leave-one-patient-out
model for which that patient was the **held-out** subject, so it has never seen the patient.
The patient's recording is split in time exactly as in the chronological protocol (first ~50%
adaptation data, next ~20% validation, last ~30% test). Three variants are produced, each in
the standard output format (``src.evaluate`` picks the alarm threshold on the validation part):

* ``general``          the general model unchanged; only the alarm threshold is set on the
                       patient's own validation data (calibration only)
* ``finetune_context`` the 5-minute context GRU is fine-tuned on the patient's early data;
                       the window encoder stays general
* ``finetune_full``    the window encoder (ST-GNN) is fine-tuned too, then the context GRU

Compare with the same patients' from-scratch patient-specific model
(``results/context_stgnn_bands_chrono``): same test data, same evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from .context import ContextGRU, Stack, embed_part, load_encoder, predict_stack, train_context
from .data import list_subjects, load_subject
from .splits import chronological_split
from .train import train_fold

VARIANTS = ("general", "finetune_context", "finetune_full")


def save_variant(out_dir, variant, subject, preds, info):
    d = os.path.join(out_dir, variant, subject)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "predictions.npz"), **preds)
    with open(os.path.join(d, "fold.json"), "w") as fh:
        json.dump({"fold": subject, "protocol": "chrono", "model": f"personalized_{variant}", **info}, fh,
                  indent=2, default=str)


def pack(parts_probs):
    out = {}
    for split, p, probs in parts_probs:
        key = f"{split}__{p.subject}"
        out.update({f"{key}__probs": probs, f"{key}__hard": p.hard, f"{key}__block": p.block,
                    f"{key}__run": p.run})
        if p.t_start is not None:
            out[f"{key}__t"] = p.t_start
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--general_encoder", required=True, help="results dir of the LOPO ST-GNN (src.train)")
    ap.add_argument("--general_context", required=True, help="results dir of the LOPO context model")
    ap.add_argument("--out_dir", default="results/personalized")
    ap.add_argument("--folds", nargs="*", default=None)
    # fine-tuning: smaller learning rates and fewer epochs than training from scratch
    ap.add_argument("--enc_lr", type=float, default=1e-4)
    ap.add_argument("--enc_epochs", type=int, default=10)
    ap.add_argument("--enc_samples", type=int, default=20000)
    ap.add_argument("--ctx_lr", type=float, default=3e-4)
    ap.add_argument("--ctx_epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if device.type == "cuda" else None
    subjects = args.folds or list_subjects(args.processed_dir)

    for s in subjects:
        enc_path = os.path.join(args.general_encoder, s, "model.pt")
        ctx_path = os.path.join(args.general_context, s, "context.pt")
        if not (os.path.exists(enc_path) and os.path.exists(ctx_path)):
            print(f"== {s}: no general model with {s} held out; skipped")
            continue
        d = load_subject(args.processed_dir, s)
        sp = chronological_split(d.block)
        if sp is None:
            print(f"== {s}: {d.n_seizures} seizures (< 3); skipped")
            continue
        t0 = time.time()
        tr, va, te = d.subset(sp["train"]), d.subset(sp["val"]), d.subset(sp["test"])
        print(f"\n== {s}: adapt {len(tr)} | val {len(va)} | test {len(te)} windows")

        enc, enc_args = load_encoder(enc_path, device)
        enc_args.num_workers = args.num_workers
        ctx_ck = torch.load(ctx_path, map_location=device)
        ca = ctx_ck["args"]
        general_ctx = ContextGRU(ctx_ck["d_in"], ca["hidden"], ca["dropout"]).to(device)
        general_ctx.load_state_dict(ctx_ck["model_state"])
        L = ca["seq_len"]
        base_info = {"general_encoder": enc_path, "general_context": ctx_path,
                     "val": [s], "test": [s], "train": [s]}

        # 1) general model, calibration only
        E = {k: embed_part(enc, p, enc_args, device, amp) for k, p in (("tr", tr), ("va", va), ("te", te))}
        S = {k: Stack([p], [E[k]], device) for k, p in (("tr", tr), ("va", va), ("te", te))}
        save_variant(args.out_dir, "general", s,
                     pack([("val", va, predict_stack(general_ctx, S["va"], L)),
                           ("test", te, predict_stack(general_ctx, S["te"], L))]), base_info)

        ctx_args = SimpleNamespace(seed=args.seed, hidden=ca["hidden"], dropout=ca["dropout"], seq_len=L,
                                   lr=args.ctx_lr, epochs=args.ctx_epochs, patience=5,
                                   samples_per_epoch=ca["samples_per_epoch"], batch_size=ca["batch_size"])

        # 2) fine-tune the context GRU only (general window encoder)
        print("  fine-tune context only")
        ctx_ft, hist_c, best_c = train_context(S["tr"], S["va"], ctx_args, device,
                                               init_state=ctx_ck["model_state"])
        save_variant(args.out_dir, "finetune_context", s,
                     pack([("val", va, predict_stack(ctx_ft, S["va"], L)),
                           ("test", te, predict_stack(ctx_ft, S["te"], L))]),
                     {**base_info, "best_val_auc": best_c, "history": hist_c})

        # 3) fine-tune the window encoder, then the context GRU on the new embeddings
        print("  fine-tune encoder")
        ft_args = SimpleNamespace(protocol="personalized", seed=args.seed, features=enc_args.features,
                                  full_graph=enc_args.full_graph, norm=enc_args.norm, augment=False,
                                  samples_per_epoch=args.enc_samples, batch_size=128,
                                  num_workers=args.num_workers, val_max_interictal=20000, alpha=0.5,
                                  pos_weight=1.0, mse_on_logits=False, lr=args.enc_lr,
                                  epochs=args.enc_epochs, patience=4)
        enc_ft, hist_e, best_e = train_fold([tr], [va], ft_args, device, amp,
                                            init_state=enc.state_dict())
        E2 = {k: embed_part(enc_ft, p, enc_args, device, amp) for k, p in (("tr", tr), ("va", va), ("te", te))}
        S2 = {k: Stack([p], [E2[k]], device) for k, p in (("tr", tr), ("va", va), ("te", te))}
        print("  fine-tune context on fine-tuned embeddings")
        ctx_ft2, hist_c2, best_c2 = train_context(S2["tr"], S2["va"], ctx_args, device,
                                                  init_state=ctx_ck["model_state"])
        save_variant(args.out_dir, "finetune_full", s,
                     pack([("val", va, predict_stack(ctx_ft2, S2["va"], L)),
                           ("test", te, predict_stack(ctx_ft2, S2["te"], L))]),
                     {**base_info, "encoder_best_val_auc": best_e, "encoder_history": hist_e,
                      "best_val_auc": best_c2, "history": hist_c2})
        print(f"  done in {time.time() - t0:.0f}s")
        del enc, enc_ft, S, S2
        torch.cuda.empty_cache() if device.type == "cuda" else None

    print("\nDone. Evaluate each variant:")
    for v in VARIANTS:
        print(f"  python -m src.evaluate --results_dir {os.path.join(args.out_dir, v)}")


if __name__ == "__main__":
    main()
