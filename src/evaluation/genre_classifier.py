from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class GenreClassifier(nn.Module):
    def __init__(
        self,
        input_dim:  int = 89,
        seq_len:    int = 32,
        n_classes:  int = 3,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1: (B, 89, 32) → (B, 64, 16)
            nn.Conv1d(input_dim,      hidden_dim,     kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 2: (B, 64, 16) → (B, 128, 8)
            nn.Conv1d(hidden_dim,     hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 3: (B, 128, 8) → (B, 256, 4)
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Global average pooling → (B, 256)
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_dim * 4, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)          
        x = self.conv_blocks(x)          
        x = self.gap(x).squeeze(-1)      
        return self.head(x)              

    def predict(self, x: Tensor) -> Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)

def train_genre_classifier(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    n_classes:    int   = 3,
    epochs:       int   = 20,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    lr_patience:  int   = 3,
    lr_factor:    float = 0.5,
    early_stop_patience: int = 8,
    early_stop_min_delta: float = 1e-3,
    input_dim:    int   = 89,
    seq_len:      int   = 32,
) -> GenreClassifier:
    model     = GenreClassifier(input_dim, seq_len, n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=1e-6,
    )
    best_acc  = 0.0
    best_state = None
    no_improve_epochs = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for x, _, label in train_loader:
            x, label = x.to(device), label.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

        # Validate
        acc = _accuracy(model, val_loader, device)
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        scheduler.step(acc)

        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  GenreClassifier epoch {epoch+1:>2}/{epochs}  "
                f"val_acc={acc:.3f}  lr={current_lr:.2e}"
            )

        if no_improve_epochs >= early_stop_patience and (best_acc - acc) > early_stop_min_delta:
            print(
                f"  Early stop classifier at epoch {epoch+1}: "
                f"no val-acc improvement for {no_improve_epochs} epochs."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"GenreClassifier trained — best val acc: {best_acc:.3f}")
    return model


def _accuracy(
    model:  GenreClassifier,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, _, label in loader:
            x, label = x.to(device), label.to(device)
            preds    = model(x).argmax(dim=1)
            correct += (preds == label).sum().item()
            total   += label.size(0)
    return correct / max(total, 1)

def compute_fooling_rate(
    classifier:  GenreClassifier,
    ec2vae:      "EC2VAE",           
    loader:      DataLoader,
    genres:      List[str],
    device:      torch.device,
    n_batches:   int = 30,
    threshold:   float = 0.5,
    cross_genre_only: bool = True,
    random_seed: Optional[int] = None,
    n_repeats: int = 1,
) -> dict:
    classifier.eval()
    ec2vae.eval()

    total = fooled = 0
    content_correct = 0
    n_same_genre_pairs = 0
    n_cross_genre_pairs = 0
    per_genre_fooled = {g: 0 for g in genres}
    per_genre_total  = {g: 0 for g in genres}

    with torch.no_grad():
        n_repeats = max(1, int(n_repeats))
        for rep in range(n_repeats):
            rep_seed = None if random_seed is None else int(random_seed) + rep
            generator = None
            if rep_seed is not None:
                generator = torch.Generator(device=device.type)
                generator.manual_seed(rep_seed)

            for i, (x, _, label) in enumerate(loader):
                if i >= n_batches:
                    break
                x, label = x.to(device), label.to(device)
                B = x.size(0)
                if B < 2:
                    continue

                content_idx = []
                style_idx = []

                for cont_i in range(B):
                    if cross_genre_only:
                        candidates = (label != label[cont_i]).nonzero(as_tuple=False).flatten()
                    else:
                        all_idx = torch.arange(B, device=label.device)
                        candidates = all_idx[all_idx != cont_i]

                    if candidates.numel() == 0:
                        continue

                    if generator is None:
                        pick = torch.randint(candidates.numel(), (1,), device=label.device)
                    else:
                        pick = torch.randint(
                            candidates.numel(),
                            (1,),
                            device=label.device,
                            generator=generator,
                        )
                    style_i = candidates[pick].item()
                    content_idx.append(cont_i)
                    style_idx.append(style_i)

                if not content_idx:
                    continue

                x_cont = x[content_idx]
                x_style = x[style_idx]
                label_cont = label[content_idx]
                label_style = label[style_idx]

                transferred = ec2vae.transfer(x_cont, x_style)
                transferred_bin = (transferred > threshold).float()
                preds = classifier(transferred_bin).argmax(dim=1)

                for j in range(len(content_idx)):
                    is_same_genre = label_cont[j].item() == label_style[j].item()
                    if is_same_genre:
                        n_same_genre_pairs += 1
                    else:
                        n_cross_genre_pairs += 1

                    style_genre = genres[label_style[j].item()]
                    per_genre_total[style_genre] += 1
                    if preds[j].item() == label_style[j].item():
                        fooled += 1
                        per_genre_fooled[style_genre] += 1
                    if preds[j].item() == label_cont[j].item():
                        content_correct += 1
                    total += 1

    fooling_rate = fooled / max(total, 1)
    content_acc  = content_correct / max(total, 1)
    per_genre    = {
        g: per_genre_fooled[g] / max(per_genre_total[g], 1)
        for g in genres
    }

    return {
        "fooling_rate":     fooling_rate,
        "content_accuracy": content_acc,
        "per_genre":        per_genre,
        "n_samples":        total,
        "cross_genre_only": cross_genre_only,
        "threshold":        threshold,
        "n_cross_genre_pairs": n_cross_genre_pairs,
        "n_same_genre_pairs":  n_same_genre_pairs,
        "random_seed": random_seed,
        "n_repeats": n_repeats,
    }


def save_classifier(model: GenreClassifier, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_classifier(
    path:      str,
    device:    torch.device,
    n_classes: int = 3,
    input_dim: int = 89,
    seq_len:   int = 32,
) -> GenreClassifier:
    model = GenreClassifier(input_dim, seq_len, n_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model