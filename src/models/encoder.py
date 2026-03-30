from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim:  int = 89,
        hidden_dim: int = 256,
        z_p_dim:    int = 64,
        z_r_dim:    int = 64,
        num_layers: int = 2,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_p_dim    = z_p_dim
        self.z_r_dim    = z_r_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        enc_dim = hidden_dim * 2   

        self.fc_mu_p     = nn.Linear(enc_dim, z_p_dim)
        self.fc_logvar_p = nn.Linear(enc_dim, z_p_dim)
        self.fc_mu_r     = nn.Linear(enc_dim, z_r_dim)
        self.fc_logvar_r = nn.Linear(enc_dim, z_r_dim)

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        _, h_n = self.gru(x)

        h_fwd = h_n[-2]  
        h_bwd = h_n[-1]  
        h = torch.cat([h_fwd, h_bwd], dim=-1)   

        mu_p     = self.fc_mu_p(h)      
        logvar_p = self.fc_logvar_p(h)  
        mu_r     = self.fc_mu_r(h)      
        logvar_r = self.fc_logvar_r(h)  

        return mu_p, logvar_p, mu_r, logvar_r