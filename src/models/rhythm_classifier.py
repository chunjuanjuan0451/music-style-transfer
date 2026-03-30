from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor


class RhythmClassifier(nn.Module):
    def __init__(
        self,
        z_r_dim:    int = 64,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=z_r_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_r: Tensor, seq_len: int) -> Tensor:
        z_r_seq = z_r.unsqueeze(1).expand(-1, seq_len, -1)

        out, _ = self.gru(z_r_seq)         
        onset_prob = self.head(out).squeeze(-1)

        return onset_prob