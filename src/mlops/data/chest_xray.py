"""Lists the chest X-ray images with their label and patient ID."""

import re
from dataclasses import dataclass
from pathlib import Path

DATASET_DIR = Path("raw") / "chest_xray"
LABELS = ("NORMAL", "PNEUMONIA")

# person123_bacteria_45.jpeg -> person123
# IM-0115-0001.jpeg -> IM-0115, NORMAL2-IM-0115-0001.jpeg -> NORMAL2-IM-0115
PATIENT_ID_PATTERN = re.compile(r"^(person\d+)_|^((?:NORMAL2-)?IM-\d+)-")


@dataclass(frozen=True)
class Sample:
    path: Path  # relative to the data folder
    label: str
    patient_id: str


def read_patient_id(file_name: str) -> str:
    match = PATIENT_ID_PATTERN.match(file_name)
    if match is None:
        raise ValueError(f"Cannot read a patient ID from {file_name!r}")
    return match.group(1) or match.group(2)


def list_samples(data_dir: Path, split: str) -> list[Sample]:
    """Returns the images of one split (train, val or test), sorted by label then path."""
    split_dir = data_dir / DATASET_DIR / split
    samples = [
        Sample(path.relative_to(data_dir), label, read_patient_id(path.name))
        for label in LABELS
        for path in sorted((split_dir / label).glob("*.jpeg"))
    ]
    if not samples:
        raise FileNotFoundError(f"No images found in {split_dir}")
    return samples
