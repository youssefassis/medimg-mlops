"""The RSNA Pneumonia Detection Challenge: one DICOM file per patient, in data/raw/rsna/."""

import csv
from pathlib import Path

from mlops.data.ingestion import LabelledFile
from mlops.data.samples import NEGATIVE, POSITIVE

RAW_DIR = Path("raw") / "rsna"
IMAGES = "stage_2_train_images"
CLASSES = "stage_2_detailed_class_info.csv"
POSITIVE_CLASS = "Lung Opacity"


def list_files(data_dir: Path) -> list[LabelledFile]:
    """The training images: the challenge's test images have no labels."""
    return [
        LabelledFile(
            data_dir / RAW_DIR / IMAGES / f"{patient_id}.dcm", to_label(rsna_class), rsna_class
        )
        for patient_id, rsna_class in sorted(read_classes(data_dir).items())
    ]


def to_label(rsna_class: str) -> str:
    """Lung Opacity is positive. Normal and No Lung Opacity / Not Normal are both negative."""
    return POSITIVE if rsna_class == POSITIVE_CLASS else NEGATIVE


def read_classes(data_dir: Path) -> dict[str, str]:
    """Returns the RSNA class of each patient. The file has one row per box, so IDs repeat."""
    classes: dict[str, str] = {}
    with (data_dir / RAW_DIR / CLASSES).open(newline="") as file:
        for row in csv.DictReader(file):
            known = classes.setdefault(row["patientId"], row["class"])
            if known != row["class"]:
                raise ValueError(f"Patient {row['patientId']} has two classes")
    return classes
