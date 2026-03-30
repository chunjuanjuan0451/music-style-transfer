from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.ec2vae import ForwardOutput


@dataclass
class LossOutput:
    total:       Tensor
    recon:       Tensor
    kl:          Tensor
    rhythm:      Tensor
    swap_recon:  Tensor
    swap_rhythm: Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "loss/total":       self.total.item(),
            "loss/recon":       self.recon.item(),
            "loss/kl":          self.kl.item(),
            "loss/rhythm":      self.rhythm.item(),
            "loss/swap_recon":  self.swap_recon.item(),
            "loss/swap_rhythm": self.swap_rhythm.item(),
        }


def compute_loss(
    output:      ForwardOutput,
    x:           Tensor,
    rhythm_gt:   Tensor,
    beta:        float,
    gamma:       float = 1.0,
    pos_weight:  float = 5.0,
    swap_weight: float = 0.0,
    swap_recon_ratio: float = 0.3,
    swap_rhythm_gt: Optional[Tensor] = None,
) -> LossOutput:
    pw = torch.tensor(pos_weight, device=x.device)
    recon = F.binary_cross_entropy_with_logits(
        output.logits,
        x,
        pos_weight=pw,
        reduction="mean",
    )

    kl_p = _kl_gaussian(output.mu_p, output.logvar_p)   
    kl_r = _kl_gaussian(output.mu_r, output.logvar_r)
    z_p_dim = output.mu_p.size(-1)
    z_r_dim = output.mu_r.size(-1)
    kl = kl_p / z_p_dim + kl_r / z_r_dim

    rhythm = _balanced_bce_prob(
        output.rhythm_pred,
        rhythm_gt,
    )

    zero = torch.tensor(0.0, device=x.device)
    swap_recon  = zero
    swap_rhythm = zero

    if (swap_weight > 0
            and output.swap_logits is not None
            and output.swap_rhythm_pred is not None):

        if swap_recon_ratio > 0:
            swap_recon = F.binary_cross_entropy_with_logits(
                output.swap_logits,
                x,
                pos_weight=pw,
                reduction="mean",
            )

        if swap_rhythm_gt is not None:
            swap_rhythm = _balanced_bce_prob(
                output.swap_rhythm_pred,
                swap_rhythm_gt,
            )

    swap_recon_ratio = max(0.0, float(swap_recon_ratio))

    total = (
        recon
        + beta         * kl
        + gamma        * rhythm
        + swap_weight  * swap_recon_ratio * swap_recon
        + swap_weight  * gamma * swap_rhythm
    )

    return LossOutput(
        total=total,
        recon=recon,
        kl=kl,
        rhythm=rhythm,
        swap_recon=swap_recon,
        swap_rhythm=swap_rhythm,
    )

def _kl_gaussian(mu: Tensor, logvar: Tensor) -> Tensor:
    kl_per_sample = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
    return kl_per_sample.mean()


def _balanced_bce_prob(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    pred = pred.clamp(eps, 1.0 - eps)
    target = target.float()

    pos_rate = target.mean()
    pos_rate = pos_rate.clamp(min=eps, max=1.0 - eps)
    w_pos = 0.5 / pos_rate
    w_neg = 0.5 / (1.0 - pos_rate)

    loss = -(
        w_pos * target * torch.log(pred)
        + w_neg * (1.0 - target) * torch.log(1.0 - pred)
    )
    return loss.mean()