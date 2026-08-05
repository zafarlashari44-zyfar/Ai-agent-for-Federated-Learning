from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class BeatDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        beats: NDArray[np.float32],
        labels: NDArray[np.int64],
        *,
        transform: Callable[[NDArray[np.float32]], NDArray[np.float32]] | None = None,
    ) -> None:
        if beats.ndim != 2 or beats.shape[1] != 216:
            raise ValueError("beats must have shape (n, 216)")
        if labels.shape != (beats.shape[0],):
            raise ValueError("labels must match beat count")
        self.beats = beats
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return self.beats.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        beat = self.beats[index].copy()
        if self.transform is not None:
            beat = self.transform(beat)
        return torch.from_numpy(beat), torch.tensor(
            self.labels[index], dtype=torch.long
        )
