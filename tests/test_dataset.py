from pathlib import Path

from PIL import Image

from mlops.data.dataset import ImageDataset
from mlops.data.samples import Sample
from mlops.data.transforms import build_transform


def test_dataset_returns_image_and_label_index(tmp_path):
    Image.new("L", (64, 48)).save(tmp_path / "a.jpeg")
    Image.new("RGB", (48, 64)).save(tmp_path / "b.jpeg")
    samples = [
        Sample(Path("a.jpeg"), "NORMAL", "p1"),
        Sample(Path("b.jpeg"), "PNEUMONIA", "p2"),
    ]

    dataset = ImageDataset(samples, tmp_path, build_transform())

    assert len(dataset) == 2
    assert [label for _, label in dataset] == [0, 1]
    assert dataset[1][0].shape == (3, 224, 224)
