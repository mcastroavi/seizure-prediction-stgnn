"""ST-GNN (GATv2 spatial branch + temporal conv branch) with a single risk output.

Layer names are identical to ``STGNN_Soft`` in the v2 notebook, so
``legacy/checkpoints/best_chbmit_soft.pt`` still loads with ``load_checkpoint`` (it expects
the v2 input format: 23 channels, 256 Hz). The number of channels and samples per window
are read from the input, so the same class trains on the v3 format (18 channels, 128 Hz).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool

from .config import CFG


def window_to_graph(eeg: np.ndarray, plv: np.ndarray, label: int, risk: float | None = None,
                    threshold: float = CFG["plv_threshold"]) -> Data:
    """Build a channel graph whose edges are PLV values above ``threshold``.

    Falls back to each node's top-3 PLV neighbours when no edge passes the threshold
    (same behaviour as the notebook).
    """
    x = torch.as_tensor(eeg, dtype=torch.float)
    adj = torch.as_tensor(plv, dtype=torch.float)
    mask = adj > threshold
    mask.fill_diagonal_(False)
    if mask.sum() == 0:
        tmp = adj.clone()
        tmp.fill_diagonal_(0)
        _, top = tmp.topk(min(3, adj.size(0) - 1), dim=1)
        mask = torch.zeros_like(adj, dtype=torch.bool)
        mask.scatter_(1, top, True)
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    edge_attr = adj[edge_index[0], edge_index[1]].unsqueeze(1)
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
             y=torch.tensor([label], dtype=torch.long))
    if risk is not None:
        g.risk = torch.tensor([risk], dtype=torch.float)
    return g


class STGNN_Soft(nn.Module):
    def __init__(self, node_feat=CFG["node_feat"], gat_out=CFG["gat_out"],
                 gat_heads=CFG["gat_heads"], dropout=CFG["dropout"]):
        super().__init__()
        nf, gd, gh = node_feat, gat_out, gat_heads

        # Per-channel temporal encoder -> node features
        self.node_enc = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.node_proj = nn.Linear(64, nf)

        # Spatial branch: GATv2 over the PLV graph
        self.gat1 = GATv2Conv(nf, gd, heads=gh, edge_dim=1, concat=True)
        self.gat_norm1 = nn.LayerNorm(gd * gh)
        self.gat2 = GATv2Conv(gd * gh, gd, heads=1, edge_dim=1, concat=False)
        self.gat_norm2 = nn.LayerNorm(gd)

        # Temporal branch on the channel-averaged signal
        self.tcn = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),
        )
        self.tcn_proj = nn.Linear(64, gd)

        self.fusion = nn.Sequential(
            nn.Linear(gd * 2, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        n_graphs = int(batch.max().item()) + 1
        n_ch = x.size(0) // n_graphs

        xn = self.node_proj(self.node_enc(x.unsqueeze(1)).squeeze(-1))
        xn = F.elu(self.gat_norm1(self.gat1(xn, ei, ea)))
        xn = F.elu(self.gat_norm2(self.gat2(xn, ei, ea)))
        spatial = global_mean_pool(xn, batch)

        xr = x.view(n_graphs, n_ch, -1).mean(dim=1, keepdim=True)
        temporal = self.tcn_proj(self.tcn(xr).squeeze(-1))

        return self.fusion(torch.cat([spatial, temporal], dim=1))  # logits, (B, 1)


class SoftSeizureLoss(nn.Module):
    """alpha * MSE(sigmoid(logit), risk) + (1 - alpha) * weighted BCE(logit, hard label).

    Note: the notebook applied MSE to the raw logits. Here the MSE target (0..1) is compared
    with ``sigmoid(logit)`` so both terms live on the same scale. Pass ``mse_on_logits=True``
    to reproduce the original behaviour exactly.
    """

    def __init__(self, alpha=CFG["alpha"], pos_weight=1.0, mse_on_logits=False):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.mse_on_logits = mse_on_logits

    def forward(self, logits, risk, hard):
        logits = logits.view(-1)
        pred = logits if self.mse_on_logits else torch.sigmoid(logits)
        mse = F.mse_loss(pred, risk.view(-1))
        bce = F.binary_cross_entropy_with_logits(logits, hard.view(-1).float(),
                                                 pos_weight=self.pos_weight)
        return self.alpha * mse + (1 - self.alpha) * bce


def load_checkpoint(model: nn.Module, path: str, map_location="cpu") -> dict:
    """Load weights saved from the notebook (handles the ``_orig_mod.`` prefix that
    ``torch.compile`` adds)."""
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("model_state", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return ckpt
