"""PyTorch dataset over chest X-ray samples."""

from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from mlops.data.chest_xray import LABELS, Sample


class ChestXrayDataset(Dataset):
    """Returns (image tensor, label index), with NORMAL = 0 and PNEUMONIA = 1."""

    def __init__(
        self,
        samples: list[Sample],
        data_dir: Path,
        transform: Callable[[Image.Image], torch.Tensor],
    ):
        self.samples = samples
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(self.data_dir / sample.path) as image:
            tensor = self.transform(image)
        return tensor, LABELS.index(sample.label)
