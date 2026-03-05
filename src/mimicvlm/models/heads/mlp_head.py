# src/mimicvlm/models/heads/mlp_head.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 14,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU(inplace=True)

        layers = [nn.BatchNorm1d(in_dim)]
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers += [nn.Linear(in_dim, hidden_dim), act, nn.Dropout(dropout)]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), act, nn.Dropout(dropout)]
            layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)
    

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D] -> logits: [B, 14]
        return self.net(z)