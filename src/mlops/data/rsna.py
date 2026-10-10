"""Turns raw RSNA DICOM files into de-identified PNGs and a manifest.

De-identification works by allowlist: the manifest keeps only the fields of ManifestRow, and
every other DICOM tag is dropped. Rejected files are listed by file hash, with the reasons.
"""

import csv
import hashlib
import io
import random
from dataclasses import dataclass
from pathlib import Path

import pydicom
from PIL import Image

from mlops.config import IngestionConfig
from mlops.data.batches import write_rows
from mlops.data.chest_xray import LABELS
from mlops.data.deidentify import pseudonymize
from mlops.data.dicom import to_uint8
from mlops.data.dicom_checks import header_problems, image_problems, parse_age

RAW_DIR = Path("raw") / "rsna"
PROCESSED_DIR = Path("processed") / "rsna"
IMAGES = "stage_2_train_images"
CLASSES = "stage_2_detailed_class_info.csv"
POSITIVE_CLASS = "Lung Opacity"
# HIPAA Safe Harbor: ages above 89 are reported as one group, 90.
MAX_REPORTED_AGE = 90


@dataclass(frozen=True)
class ManifestRow:
    path: str  # relative to the data folder
    label: str
    patient_id: str  # pseudonym
    rsna_class: str
    view: str
    sex: str
    age: int
    pixel_spacing_mm: float
    source_sha256: str  # links back to the raw file, for whoever can access the raw zone


@dataclass(frozen=True)
class Rejection:
    source_sha256: str
    reasons: str


def to_label(rsna_class: str) -> str:
    """Lung Opacity is PNEUMONIA. Normal and No Lung Opacity / Not Normal are both NORMAL."""
    normal, pneumonia = LABELS
    return pneumonia if rsna_class == POSITIVE_CLASS else normal


def read_classes(data_dir: Path) -> dict[str, str]:
    """Returns the RSNA class of each patient. The file has one row per box, so IDs repeat."""
    classes: dict[str, str] = {}
    with (data_dir / RAW_DIR / CLASSES).open(newline="") as file:
        for row in csv.DictReader(file):
            known = classes.setdefault(row["patientId"], row["class"])
            if known != row["class"]:
                raise ValueError(f"Patient {row['patientId']} has two classes")
    return classes


def sample_patients(patient_ids: list[str], size: int, seed: int) -> list[str]:
    """Returns a random sample of the patients, all of them if size is larger."""
    sample = random.Random(seed).sample(sorted(patient_ids), min(size, len(patient_ids)))
    return sorted(sample)


def ingest_file(
    path: Path, rsna_class: str, data_dir: Path, config: IngestionConfig, secret: str
) -> ManifestRow | Rejection:
    """Checks one DICOM file and, if it passes, saves it as PNG."""
    content = path.read_bytes()
    source_sha256 = hashlib.sha256(content).hexdigest()
    dataset = pydicom.dcmread(io.BytesIO(content))

    if problems := header_problems(dataset, config.max_age):
        return Rejection(source_sha256, "; ".join(problems))
    image = to_uint8(dataset)
    if problems := image_problems(image, config.min_contrast, config.max_near_black):
        return Rejection(source_sha256, "; ".join(problems))

    image_path = PROCESSED_DIR / "images" / f"{source_sha256[:16]}.png"
    Image.fromarray(image).save(data_dir / image_path)
    return ManifestRow(
        path=str(image_path),
        label=to_label(rsna_class),
        patient_id=pseudonymize(dataset.PatientID, secret),
        rsna_class=rsna_class,
        view=dataset.ViewPosition,
        sex=dataset.PatientSex,
        age=min(parse_age(dataset.PatientAge), MAX_REPORTED_AGE),
        pixel_spacing_mm=round(float(dataset.PixelSpacing[0]), 3),
        source_sha256=source_sha256,
    )


def ingest(
    data_dir: Path, config: IngestionConfig, secret: str
) -> tuple[list[ManifestRow], list[Rejection]]:
    """Ingests a sample of the RSNA training images.

    Writes the PNGs, manifest.csv and rejected.csv to data_dir/processed/rsna/, replacing any
    earlier run.
    """
    classes = read_classes(data_dir)
    out_dir = data_dir / PROCESSED_DIR
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    for old_image in (out_dir / "images").glob("*.png"):
        old_image.unlink()

    manifest: list[ManifestRow] = []
    rejected: list[Rejection] = []
    for patient_id in sample_patients(list(classes), config.sample_size, config.seed):
        path = data_dir / RAW_DIR / IMAGES / f"{patient_id}.dcm"
        try:
            result = ingest_file(path, classes[patient_id], data_dir, config, secret)
        except Exception as error:
            raise RuntimeError(f"Failed to ingest {path}") from error
        (manifest if isinstance(result, ManifestRow) else rejected).append(result)

    write_rows(out_dir / "manifest.csv", ManifestRow, manifest)
    write_rows(out_dir / "rejected.csv", Rejection, rejected)
    return manifest, rejected
