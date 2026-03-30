from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from src.models.ec2vae   import EC2VAE
from src.training.loss   import compute_loss


def pitch_class_histogram(roll: np.ndarray) -> np.ndarray:
    pitch_part = roll[:, :88]                      
    counts = np.zeros(12, dtype=np.float32)
    for pc in range(12):
        indices = np.arange(pc, 88, 12)
        counts[pc] = pitch_part[:, indices].sum()

    total = counts.sum()
    if total < 1e-8:
        return np.ones(12, dtype=np.float32) / 12.0   
    return counts / total


def pitch_class_histogram_divergence(
    roll_a: np.ndarray,
    roll_b: np.ndarray,
    eps: float = 1e-8,
) -> float:
    p = pitch_class_histogram(roll_a) + eps
    q = pitch_class_histogram(roll_b) + eps
    p /= p.sum()
    q /= q.sum()
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    return (kl_pq + kl_qp) / 2.0


def onset_sequence(roll: np.ndarray) -> np.ndarray:
    active = (roll[:, :88].sum(axis=1) > 0).astype(np.float32)
    onsets = active.copy()
    if active.shape[0] > 1:
        onsets[1:] = np.clip(active[1:] - active[:-1], 0.0, 1.0)
    return onsets


def rhythmic_density(roll: np.ndarray) -> float:
    return float(onset_sequence(roll).mean())


def rhythmic_entropy(roll: np.ndarray) -> float:
    onsets = onset_sequence(roll)
    p_on  = float(onsets.mean())
    p_off = 1.0 - p_on
    eps   = 1e-8
    h = 0.0
    for p in (p_on, p_off):
        if p > eps:
            h -= p * np.log(p + eps)
    return h


def rhythmic_density_distance(
    roll_transfer: np.ndarray,
    roll_style:    np.ndarray,
) -> float:
    return abs(rhythmic_density(roll_transfer) - rhythmic_density(roll_style))


def reconstruction_loss(
    model:   EC2VAE,
    loader:  DataLoader,
    device:  torch.device,
    config:  dict,
    n_batches: int = 50,
) -> Dict[str, float]:
    model.eval()
    t       = config["training"]
    beta    = t["beta_end"]      
    gamma   = t["gamma"]
    pos_w   = t.get("pos_weight", 5.0)

    totals  = {"total": 0.0, "recon": 0.0, "kl": 0.0, "rhythm": 0.0}
    n_seen  = 0

    with torch.no_grad():
        for i, (x, rhythm_gt, _) in enumerate(loader):
            if i >= n_batches:
                break
            x         = x.to(device)
            rhythm_gt = rhythm_gt.to(device)
            out       = model(x, do_swap=False)
            loss      = compute_loss(out, x, rhythm_gt,
                                     beta=beta, gamma=gamma, pos_weight=pos_w)
            totals["total"]  += loss.total.item()
            totals["recon"]  += loss.recon.item()
            totals["kl"]     += loss.kl.item()
            totals["rhythm"] += loss.rhythm.item()
            n_seen += 1

    return {k: v / max(n_seen, 1) for k, v in totals.items()}



def evaluate_transfer_batch(
    model:         EC2VAE,
    x_content:     Tensor,
    x_style:       Tensor,
    device:        torch.device,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        transferred = model.transfer(x_content, x_style)  

    x_content_np   = x_content.cpu().numpy()    
    x_style_np     = x_style.cpu().numpy()
    transferred_np = transferred.cpu().numpy()

    batch_size = x_content_np.shape[0]
    metrics: Dict[str, float] = {
        "pitch_kl_div":             0.0,
        "rhythm_density_content":   0.0,
        "rhythm_density_style":     0.0,
        "rhythm_density_transfer":  0.0,
        "rhythm_density_dist":      0.0,
        "rhythm_entropy_transfer":  0.0,
    }

    for i in range(batch_size):
        c = x_content_np[i]      
        s = x_style_np[i]
        t = (transferred_np[i] > 0.5).astype(np.float32)   

        metrics["pitch_kl_div"]            += pitch_class_histogram_divergence(c, t)
        metrics["rhythm_density_content"]  += rhythmic_density(c)
        metrics["rhythm_density_style"]    += rhythmic_density(s)
        metrics["rhythm_density_transfer"] += rhythmic_density(t)
        metrics["rhythm_density_dist"]     += rhythmic_density_distance(t, s)
        metrics["rhythm_entropy_transfer"] += rhythmic_entropy(t)

    return {k: v / batch_size for k, v in metrics.items()}


def save_metrics(metrics: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)