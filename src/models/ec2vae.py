from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.encoder           import Encoder
from src.models.decoder           import Decoder
from src.models.rhythm_classifier import RhythmClassifier


@dataclass
class ForwardOutput:
    logits:      Tensor
    rhythm_pred: Tensor
    mu_p:        Tensor
    logvar_p:    Tensor
    mu_r:        Tensor
    logvar_r:    Tensor
    swap_logits:      Optional[Tensor] = None
    swap_rhythm_pred: Optional[Tensor] = None
    swap_perm:        Optional[Tensor] = None


class EC2VAE(nn.Module):
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
        self.z_p_dim = z_p_dim
        self.z_r_dim = z_r_dim

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            z_p_dim=z_p_dim,
            z_r_dim=z_r_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = Decoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            z_p_dim=z_p_dim,
            z_r_dim=z_r_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.rhythm_classifier = RhythmClassifier(
            z_r_dim=z_r_dim,
            hidden_dim=hidden_dim,
        )

    @staticmethod
    def reparameterise(mu: Tensor, logvar: Tensor) -> Tensor:
        if not torch.is_grad_enabled():
            return mu
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def encode(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        mu_p, logvar_p, mu_r, logvar_r = self.encoder(x)
        z_p = self.reparameterise(mu_p, logvar_p)
        z_r = self.reparameterise(mu_r, logvar_r)
        return z_p, z_r, mu_p, logvar_p, mu_r, logvar_r

    def decode(
        self,
        z_p:             Tensor,
        z_r:             Tensor,
        target:          Tensor,
        teacher_forcing: bool = True,
        rhythm_cond:     Optional[Tensor] = None,
    ) -> Tensor:
        return self.decoder(
            z_p,
            z_r,
            target,
            teacher_forcing=teacher_forcing,
            rhythm_cond=rhythm_cond,
        )

    def transfer(
        self,
        x_content: Tensor,
        x_style:   Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        self.eval()
        with torch.no_grad():
            z_p, _, _, _, _, _ = self.encode(x_content)
            _, z_r, _, _, _, _ = self.encode(x_style)
            seq_len = x_content.size(1)
            rhythm_cond = self.rhythm_classifier(z_r, seq_len)
            transferred = self.decoder.sample(
                z_p,
                z_r,
                seq_len,
                temperature=temperature,
                rhythm_cond=rhythm_cond,
            )
        return transferred

    def forward(self, x: Tensor, do_swap: bool = True) -> ForwardOutput:
        batch_size, seq_len, _ = x.shape

        z_p, z_r, mu_p, logvar_p, mu_r, logvar_r = self.encode(x)

        rhythm_pred = self.rhythm_classifier(z_r, seq_len)
        logits      = self.decoder(
            z_p,
            z_r,
            x,
            teacher_forcing=True,
            rhythm_cond=rhythm_pred,
        )

        swap_logits      = None
        swap_rhythm_pred = None

        if do_swap and self.training:
            perm          = torch.randperm(batch_size, device=z_r.device)
            z_r_shuffled  = z_r[perm]

            swap_rhythm_pred = self.rhythm_classifier(z_r_shuffled, seq_len)
            swap_logits      = self.decoder(
                z_p,
                z_r_shuffled,
                x,
                teacher_forcing=False,
                rhythm_cond=swap_rhythm_pred,
            )

        return ForwardOutput(
            logits=logits,
            rhythm_pred=rhythm_pred,
            mu_p=mu_p,
            logvar_p=logvar_p,
            mu_r=mu_r,
            logvar_r=logvar_r,
            swap_logits=swap_logits,
            swap_rhythm_pred=swap_rhythm_pred,
            swap_perm=perm if (do_swap and self.training) else None,
        )

    @classmethod
    def from_config(cls, config: dict) -> "EC2VAE":
        m = config["model"]
        return cls(
            input_dim=m["input_dim"],
            hidden_dim=m["hidden_dim"],
            z_p_dim=m["z_p_dim"],
            z_r_dim=m["z_r_dim"],
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)