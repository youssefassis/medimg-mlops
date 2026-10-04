from pathlib import Path

import pytest
import torch
from PIL import Image

from mlops.data.chest_xray import Sample
from mlops.data.dataset import ChestXrayDataset
from mlops.data.transforms import build_transform


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_transform_gives_a_3x224x224_tensor(mode):
    image = Image.new(mode, (300, 200))

    tensor = build_transform()(image)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_dataset_returns_image_and_label_index(tmp_path):
    Image.new("L", (64, 48)).save(tmp_path / "a.jpeg")
    Image.new("RGB", (48, 64)).save(tmp_path / "b.jpeg")
    samples = [
        Sample(Path("a.jpeg"), "NORMAL", "p1"),
        Sample(Path("b.jpeg"), "PNEUMONIA", "p2"),
    ]

    dataset = ChestXrayDataset(samples, tmp_path, build_transform())

    assert len(dataset) == 2
    assert [label for _, label in dataset] == [0, 1]
    assert dataset[1][0].shape == (3, 224, 224)
