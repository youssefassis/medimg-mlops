from pathlib import Path

import numpy as np
import pytest
from pydicom import Dataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


@pytest.fixture
def make_dataset(tmp_path):
    """Creates empty image files under tmp_path/raw/chest_xray/ and returns tmp_path."""

    def make(files: list[str]) -> Path:
        for file in files:
            path = tmp_path / "raw" / "chest_xray" / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return tmp_path

    return make


@pytest.fixture
def make_dicom():
    """Returns a function building an in-memory DICOM image; extra keyword args set tags."""

    def make(pixels: np.ndarray, photometric: str = "MONOCHROME2", **tags) -> Dataset:
        dataset = Dataset()
        dataset.set_pixel_data(pixels, photometric, bits_stored=pixels.itemsize * 8)
        dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset.SOPClassUID = SecondaryCaptureImageStorage
        dataset.SOPInstanceUID = generate_uid()
        for keyword, value in tags.items():
            setattr(dataset, keyword, value)
        return dataset

    return make


@pytest.fixture
def chest_tags():
    """Header tags of a valid RSNA-like chest X-ray."""
    return {
        "Modality": "CR",
        "ViewPosition": "PA",
        "PatientSex": "F",
        "PatientAge": "51",
        "PixelSpacing": [0.143, 0.143],
    }


@pytest.fixture
def noise():
    """A 32×32 8-bit image that passes the quality checks."""
    return np.random.default_rng(0).integers(0, 256, (32, 32), dtype=np.uint8)
