# src/mimicvlm/models/losses.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class BCEWithLogitsConfig:
    pos_weight: Optional[torch.Tensor] = None  # shape [14]
    label_smoothing: float = 0.0  # optional
    reduction: str = "mean"


class MultiLabelBCEWithLogits(nn.Module):
    """
    Targets expected in {0,1}. If you still have {-1,0,1} anywhere, fix upstream.
    """

    def __init__(self, cfg: BCEWithLogitsConfig):
        super().__init__()
        self.cfg = cfg
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=cfg.pos_weight,
            reduction=cfg.reduction,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B,14], targets: [B,14]
        if self.cfg.label_smoothing > 0:
            eps = float(self.cfg.label_smoothing)
            targets = targets * (1.0 - eps) + 0.5 * eps
        return self.criterion(logits, targets)