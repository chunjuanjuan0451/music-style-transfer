from __future__ import annotations
import numpy as np

def transpose(
    roll: np.ndarray,
    semitones: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if semitones == 0:
        return roll.copy()

    out = np.roll(roll, semitones, axis=1)   # axis=1 → pitch dimension
    if semitones > 0:
        out[:, :semitones] = 0.0             # zero low-end wrap-around
    else:
        out[:, semitones:] = 0.0             # zero high-end wrap-around
    return out


def temporal_jitter(
    roll: np.ndarray,
    rng: np.random.Generator,
    drop_p: float = 0.05,
) -> np.ndarray:
    out = roll.copy()
    seq_len = out.shape[0]

    active = out.any(axis=1)                         
    prev_active = np.concatenate([[False], active[:-1]])
    onset_steps = np.where(active & ~prev_active)[0]  # indices of onset timesteps

    if len(onset_steps) == 0:
        return out

    drop_mask = rng.random(len(onset_steps)) < drop_p
    out[onset_steps[drop_mask]] = 0.0
    return out


def augment(roll: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    semitones = int(rng.integers(-6, 6))  
    roll = transpose(roll, semitones, rng)

    if rng.random() < 0.5:
        roll = temporal_jitter(roll, rng)

    return roll