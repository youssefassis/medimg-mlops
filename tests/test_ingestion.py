import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mlops.config import IngestionConfig
from mlops.data.deidentify import pseudonymize
from mlops.data.dicom import to_uint8
from mlops.data.ingestion import (
    PROCESSED_DIR,
    LabelledFile,
    by_label_and_view,
    ingest,
    read_manifest,
    sample_files,
)

SECRET = "test secret"
CONFIG = IngestionConfig(
    data_dir="data",
    source="test",
    sample_size=10,
    seed=0,
    max_age=120,
    min_contrast=10,
    max_near_black=0.7,
)


@pytest.fixture
def source_files(tmp_path, make_dicom, noise, chest_tags):
    """Writes a few DICOM files and returns (data_dir, their LabelledFiles, the datasets by
    patient ID)."""
    patients = {
        "positive-patient": ("PNEUMONIA", {}),
        "old-patient": ("NORMAL", {"PatientAge": "95", "ViewPosition": "AP"}),
        "male-patient": ("NORMAL", {"PatientSex": "M"}),
        "invalid-patient": ("NORMAL", {"PatientAge": "155"}),
    }
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    files, datasets = [], {}
    for patient_id, (label, tags) in patients.items():
        dataset = make_dicom(
            noise, PatientID=patient_id, PatientName=patient_id, **(chest_tags | tags)
        )
        path = raw_dir / f"{patient_id}.dcm"
        dataset.save_as(path, enforce_file_format=True)
        files.append(LabelledFile(path, label, source_class=f"class of {label}"))
        datasets[patient_id] = dataset
    return tmp_path, files, datasets


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def test_sample_files_is_seeded_and_capped():
    files = [LabelledFile(Path(f"{i}.dcm"), "NORMAL", "c") for i in range(10)]

    assert sample_files(files, 3, seed=1) == sample_files(files[::-1], 3, seed=1)
    assert len(sample_files(files, 3, seed=1)) == 3
    assert sample_files(files, 50, seed=1) == sorted(files, key=lambda file: file.path)


def test_ingest_writes_manifest_without_patient_ids(source_files):
    data_dir, files, datasets = source_files

    manifest, rejected = ingest(files, data_dir, CONFIG, SECRET)

    assert len(manifest) == 3
    rows = {row["patient_id"]: row for row in read_csv(data_dir / PROCESSED_DIR / "manifest.csv")}
    positive = rows[pseudonymize("positive-patient", SECRET)]
    assert (positive["label"], positive["source_class"]) == ("PNEUMONIA", "class of PNEUMONIA")
    assert (positive["view"], positive["sex"]) == ("PA", "F")
    assert (positive["age"], positive["pixel_spacing_mm"]) == ("51", "0.143")
    assert rows[pseudonymize("old-patient", SECRET)]["age"] == "90"  # 95, grouped

    output = data_dir / PROCESSED_DIR
    written = (output / "manifest.csv").read_text() + (output / "rejected.csv").read_text()
    file_names = " ".join(path.name for path in output.rglob("*"))
    for patient_id in datasets:
        assert patient_id not in written
        assert patient_id not in file_names


def test_ingest_saves_the_converted_pixels(source_files):
    data_dir, files, datasets = source_files

    manifest, _ = ingest(files, data_dir, CONFIG, SECRET)

    row = next(row for row in manifest if row.label == "PNEUMONIA")
    saved = np.asarray(Image.open(data_dir / row.path))
    assert np.array_equal(saved, to_uint8(datasets["positive-patient"]))


def test_ingest_lists_rejected_files_with_reasons(source_files):
    data_dir, files, _ = source_files

    _, rejected = ingest(files, data_dir, CONFIG, SECRET)

    assert [row.reasons for row in rejected] == ["age 155 is above 120"]
    assert read_csv(data_dir / PROCESSED_DIR / "rejected.csv")[0]["reasons"] == (
        "age 155 is above 120"
    )


def test_ingest_replaces_earlier_images(source_files):
    data_dir, files, _ = source_files
    stale = data_dir / PROCESSED_DIR / "images" / "stale.png"
    stale.parent.mkdir(parents=True)
    stale.touch()

    ingest(files, data_dir, CONFIG, SECRET)

    assert not stale.exists()


def test_read_manifest_reads_what_ingest_wrote(source_files):
    data_dir, files, _ = source_files

    manifest, _ = ingest(files, data_dir, CONFIG, SECRET)

    assert read_manifest(data_dir) == manifest


def test_by_label_and_view(source_files):
    data_dir, files, _ = source_files
    manifest, _ = ingest(files, data_dir, CONFIG, SECRET)

    assert {by_label_and_view(row) for row in manifest} == {
        ("PNEUMONIA", "PA"),
        ("NORMAL", "AP"),
        ("NORMAL", "PA"),
    }
