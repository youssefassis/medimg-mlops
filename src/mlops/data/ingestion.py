"""Turns the DICOM files of a source (see mlops.data.sources) into de-identified PNGs and a
manifest.

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
from mlops.data.deidentify import pseudonymize
from mlops.data.dicom import to_uint8
from mlops.data.dicom_checks import header_problems, image_problems, parse_age

PROCESSED_DIR = Path("processed")
# HIPAA Safe Harbor: ages above 89 are reported as one group, 90.
MAX_REPORTED_AGE = 90


@dataclass(frozen=True)
class LabelledFile:
    """A DICOM file listed by a source."""

    path: Path
    label: str  # NEGATIVE or POSITIVE, see mlops.data.samples
    source_class: str  # the source's own class, kept for reference


@dataclass(frozen=True)
class ManifestRow:
    path: str  # relative to the data folder
    label: str
    patient_id: str  # pseudonym
    source_class: str
    view: str
    sex: str
    age: int
    pixel_spacing_mm: float
    source_sha256: str  # links back to the raw file, for whoever can access the raw zone


@dataclass(frozen=True)
class Rejection:
    source_sha256: str
    reasons: str


def by_label_and_view(row: ManifestRow) -> tuple[str, str]:
    """Splits by label and view keep the AP/PA mix of each class, so the view shortcut found in
    notebooks/explore_rsna.py weighs the same in every split."""
    return row.label, row.view


def read_manifest(data_dir: Path) -> list[ManifestRow]:
    return read_rows(data_dir / PROCESSED_DIR / "manifest.csv")


def read_rows(path: Path) -> list[ManifestRow]:
    """Reads the manifest, or a split of it such as the test set."""
    with path.open(newline="") as file:
        return [
            ManifestRow(
                **row | {"age": int(row["age"]), "pixel_spacing_mm": float(row["pixel_spacing_mm"])}
            )
            for row in csv.DictReader(file)
        ]


def sample_files(files: list[LabelledFile], size: int, seed: int) -> list[LabelledFile]:
    """Returns a random sample of the files, all of them if size is larger, sorted by path."""
    ordered = sorted(files, key=lambda file: file.path)
    return sorted(
        random.Random(seed).sample(ordered, min(size, len(ordered))), key=lambda f: f.path
    )


def ingest_file(
    file: LabelledFile, data_dir: Path, config: IngestionConfig, secret: str
) -> ManifestRow | Rejection:
    """Checks one DICOM file and, if it passes, saves it as PNG."""
    content = file.path.read_bytes()
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
        label=file.label,
        patient_id=pseudonymize(dataset.PatientID, secret),
        source_class=file.source_class,
        view=dataset.ViewPosition,
        sex=dataset.PatientSex,
        age=min(parse_age(dataset.PatientAge), MAX_REPORTED_AGE),
        pixel_spacing_mm=round(float(dataset.PixelSpacing[0]), 3),
        source_sha256=source_sha256,
    )


def ingest(
    files: list[LabelledFile], data_dir: Path, config: IngestionConfig, secret: str
) -> tuple[list[ManifestRow], list[Rejection]]:
    """Ingests a sample of the files (`config.sample_size`).

    Writes the PNGs, manifest.csv and rejected.csv to data_dir/processed/, replacing any earlier
    run.
    """
    out_dir = data_dir / PROCESSED_DIR
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    for old_image in (out_dir / "images").glob("*.png"):
        old_image.unlink()

    manifest: list[ManifestRow] = []
    rejected: list[Rejection] = []
    for file in sample_files(files, config.sample_size, config.seed):
        try:
            result = ingest_file(file, data_dir, config, secret)
        except Exception as error:
            raise RuntimeError(f"Failed to ingest {file.path}") from error
        (manifest if isinstance(result, ManifestRow) else rejected).append(result)

    write_rows(out_dir / "manifest.csv", ManifestRow, manifest)
    write_rows(out_dir / "rejected.csv", Rejection, rejected)
    return manifest, rejected
