from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augment import augment as apply_augment


class MusicDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        genres: List[str],
        split: str = "train",
        augment: Optional[bool] = None,
    ) -> None:
        super().__init__()
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"
        self.split   = split
        self.genres  = genres
        self.do_augment: bool = (split == "train") if augment is None else augment

        segments_list: List[np.ndarray] = []
        labels_list:   List[np.ndarray] = []

        for label_idx, genre in enumerate(genres):
            fpath = Path(data_dir) / genre / f"{split}.npy"
            if not fpath.exists():
                raise FileNotFoundError(
                    f"Expected preprocessed file not found: {fpath}\n"
                    "Run src/data/preprocess.py locally first."
                )
            arr = np.load(str(fpath))  
            segments_list.append(arr)
            labels_list.append(
                np.full(len(arr), label_idx, dtype=np.int64)
            )

        if not segments_list:
            raise RuntimeError(
                f"No segments loaded for split='{split}' from {data_dir}.\n"
                "All genre files were either missing or empty. "
                "Check that preprocess.py ran successfully and produced non-empty .npy files."
            )

        self.segments: np.ndarray = np.concatenate(segments_list, axis=0)
        self.labels:   np.ndarray = np.concatenate(labels_list,   axis=0)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seg = self.segments[idx].copy()  
        pitch_part = seg[:, :88]       

        if self.do_augment:
            rng = np.random.default_rng(seed=42 + idx)
            pitch_part = apply_augment(pitch_part, rng)   
        rest_part = (pitch_part.sum(axis=1, keepdims=True) == 0).astype(np.float32)

        seg_out = np.concatenate([pitch_part, rest_part], axis=1)  

        active = (pitch_part.sum(axis=1) > 0).astype(np.float32)      
        rhythm_gt = active.copy()
        if active.shape[0] > 1:
            rhythm_gt[1:] = np.clip(active[1:] - active[:-1], 0.0, 1.0)

        return (
            torch.from_numpy(seg_out),          
            torch.from_numpy(rhythm_gt),            
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

    def genre_counts(self) -> dict[str, int]:
        return {
            genre: int((self.labels == i).sum())
            for i, genre in enumerate(self.genres)
        }