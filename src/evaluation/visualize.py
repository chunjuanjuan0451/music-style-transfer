from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor


def _plt():
    import matplotlib.pyplot as plt
    return plt

def plot_piano_roll_comparison(
    roll_content:    np.ndarray,
    roll_style:      np.ndarray,
    roll_transferred: np.ndarray,
    title:    str  = "Style Transfer Comparison",
    save_path: Optional[str] = None,
) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    panels = [
        (roll_content,     "Content Source\n(melody z_p)", "Blues"),
        (roll_style,       "Style Source\n(rhythm z_r)",   "Greens"),
        (roll_transferred, "Transferred Output",            "Oranges"),
    ]

    for ax, (roll, label, cmap) in zip(axes, panels):
        ax.imshow(
            roll[:, :88].T,
            aspect="auto", origin="lower",
            cmap=cmap, interpolation="nearest",
            vmin=0, vmax=1,
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Time step")

    axes[0].set_ylabel("Pitch (MIDI 21–108)")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.show()


def plot_tsne_latent_space(
    model:    "EC2VAE",    # noqa: F821
    loader:   "DataLoader",
    genres:   List[str],
    device:   torch.device,
    n_batches: int = 40,
    save_path: Optional[str] = None,
) -> None:
    from sklearn.manifold import TSNE

    model.eval()
    z_p_list, z_r_list, label_list = [], [], []

    with torch.no_grad():
        for i, (x, _, label) in enumerate(loader):
            if i >= n_batches:
                break
            x = x.to(device)
            z_p, z_r, *_ = model.encode(x)
            z_p_list.append(z_p.cpu().numpy())
            z_r_list.append(z_r.cpu().numpy())
            label_list.append(label.numpy())

    z_p     = np.concatenate(z_p_list, axis=0)
    z_r     = np.concatenate(z_r_list, axis=0)
    labels  = np.concatenate(label_list, axis=0)

    def _new_tsne() -> TSNE:
        common_kwargs = dict(n_components=2, random_state=42, perplexity=30)
        try:
            return TSNE(max_iter=1000, **common_kwargs)
        except TypeError:
            return TSNE(n_iter=1000, **common_kwargs)

    tsne    = _new_tsne()
    z_p_2d  = tsne.fit_transform(z_p)
    tsne    = _new_tsne()
    z_r_2d  = tsne.fit_transform(z_r)

    plt   = _plt()
    COLORS = ["#4C72B0", "#DD8452", "#55A868"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, emb, title in [
        (axes[0], z_p_2d, "z_p  (pitch/content)\nShould NOT separate by genre"),
        (axes[1], z_r_2d, "z_r  (rhythm/style)\nShould separate by genre"),
    ]:
        for i, genre in enumerate(genres):
            mask = labels == i
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=COLORS[i % len(COLORS)], label=genre,
                s=8, alpha=0.6,
            )
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, markerscale=3)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("t-SNE of Latent Space", fontsize=13)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.show()


def plot_latent_interpolation(
    model:      "EC2VAE",
    x_a:        Tensor,
    x_b:        Tensor,
    device:     torch.device,
    n_steps:    int = 8,
    save_path:  Optional[str] = None,
) -> None:
    model.eval()
    with torch.no_grad():
        z_p_a, z_r_a, *_ = model.encode(x_a.to(device))
        _,     z_r_b, *_ = model.encode(x_b.to(device))

        seq_len = x_a.size(1)
        rolls = []
        for step in range(n_steps):
            alpha = step / max(n_steps - 1, 1)
            z_r_interp = (1 - alpha) * z_r_a + alpha * z_r_b
            roll = model.decoder.sample(z_p_a, z_r_interp, seq_len)
            rolls.append(roll.squeeze(0).cpu().numpy())   # (seq_len, 89)

    plt = _plt()
    fig, axes = plt.subplots(1, n_steps, figsize=(2.5 * n_steps, 3), sharey=True)
    for i, (ax, roll) in enumerate(zip(axes, rolls)):
        alpha = i / max(n_steps - 1, 1)
        ax.imshow(
            roll[:, :88].T,
            aspect="auto", origin="lower",
            cmap="Blues", interpolation="nearest",
            vmin=0, vmax=1,
        )
        ax.set_title(f"α={alpha:.2f}", fontsize=8)
        ax.set_xlabel("t")
        ax.set_xticks([])

    axes[0].set_ylabel("Pitch")
    fig.suptitle("z_r Interpolation  (A → B)", fontsize=11)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.show()


def plot_pitch_class_histograms(
    roll_content:    np.ndarray,
    roll_transferred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    from src.evaluation.metrics import pitch_class_histogram

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]

    hist_c = pitch_class_histogram(roll_content)
    hist_t = pitch_class_histogram(roll_transferred)

    plt = _plt()
    x   = np.arange(12)
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(x - w/2, hist_c, w, label="Content source", color="#4C72B0", alpha=0.8)
    ax.bar(x + w/2, hist_t, w, label="Transferred",    color="#DD8452", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(NOTE_NAMES)
    ax.set_ylabel("Normalised count")
    ax.set_title("Pitch Class Histogram: Content vs Transferred")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.show()


def plot_rhythm_comparison(
    roll_content:    np.ndarray,
    roll_style:      np.ndarray,
    roll_transferred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Onset sequence bar charts for content / style / transferred.

    Args:
        roll_content:     (seq_len, 88+)
        roll_style:       (seq_len, 88+)
        roll_transferred: (seq_len, 88+)
        save_path:        If given, save figure.
    """
    from src.evaluation.metrics import onset_sequence

    plt = _plt()
    fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
    panels = [
        (roll_content,     "#4C72B0", "Content Source"),
        (roll_style,       "#55A868", "Style Source"),
        (roll_transferred, "#DD8452", "Transferred"),
    ]

    for ax, (roll, color, label) in zip(axes, panels):
        onsets = onset_sequence(roll)
        ax.bar(range(len(onsets)), onsets, color=color, width=0.8)
        density = float(onsets.mean())
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(-0.1, 1.2)
        ax.set_yticks([0, 1])
        ax.text(0.98, 0.7, f"density={density:.2f}",
                transform=ax.transAxes, ha="right", fontsize=8)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("Rhythm (Onset Sequence) Comparison", fontsize=11)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.show()