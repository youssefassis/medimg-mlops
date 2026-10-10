import csv

import numpy as np
import pytest
from PIL import Image

from mlops.config import IngestionConfig
from mlops.data.deidentify import pseudonymize
from mlops.data.dicom import to_uint8
from mlops.data.rsna import (
    PROCESSED_DIR,
    RAW_DIR,
    ingest,
    read_classes,
    sample_patients,
    to_label,
)

SECRET = "test secret"
CONFIG = IngestionConfig(
    data_dir="data", sample_size=10, seed=0, max_age=120, min_contrast=10, max_near_black=0.7
)


@pytest.fixture
def raw_rsna(tmp_path, make_dicom, noise, chest_tags):
    """Writes a tiny RSNA folder and returns (data_dir, the DICOM datasets by patient ID)."""
    patients = {
        "opacity-patient": ("Lung Opacity", {}),
        "normal-patient": ("Normal", {"PatientAge": "95", "ViewPosition": "AP"}),
        "not-normal-patient": ("No Lung Opacity / Not Normal", {"PatientSex": "M"}),
        "invalid-patient": ("Normal", {"PatientAge": "155"}),
    }
    raw_dir = tmp_path / RAW_DIR
    (raw_dir / "stage_2_train_images").mkdir(parents=True)
    datasets = {}
    rows = []
    for patient_id, (rsna_class, tags) in patients.items():
        dataset = make_dicom(
            noise, PatientID=patient_id, PatientName=patient_id, **(chest_tags | tags)
        )
        path = raw_dir / "stage_2_train_images" / f"{patient_id}.dcm"
        dataset.save_as(path, enforce_file_format=True)
        datasets[patient_id] = dataset
        rows.append((patient_id, rsna_class))
    rows.append(("opacity-patient", "Lung Opacity"))  # a second box: the ID repeats

    with (raw_dir / "stage_2_detailed_class_info.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("patientId", "class"))
        writer.writerows(rows)
    return tmp_path, datasets


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


@pytest.mark.parametrize(
    ("rsna_class", "label"),
    [
        ("Lung Opacity", "PNEUMONIA"),
        ("Normal", "NORMAL"),
        ("No Lung Opacity / Not Normal", "NORMAL"),
    ],
)
def test_to_label(rsna_class, label):
    assert to_label(rsna_class) == label


def test_read_classes_reads_each_patient_once(raw_rsna):
    data_dir, _ = raw_rsna

    classes = read_classes(data_dir)

    assert len(classes) == 4
    assert classes["opacity-patient"] == "Lung Opacity"


def test_sample_patients_is_seeded_and_capped():
    patient_ids = [f"p{i}" for i in range(10)]

    assert sample_patients(patient_ids, 3, seed=1) == sample_patients(patient_ids, 3, seed=1)
    assert len(sample_patients(patient_ids, 3, seed=1)) == 3
    assert sample_patients(patient_ids, 50, seed=1) == sorted(patient_ids)


def test_ingest_writes_manifest_without_patient_ids(raw_rsna):
    data_dir, datasets = raw_rsna

    manifest, rejected = ingest(data_dir, CONFIG, SECRET)

    assert len(manifest) == 3
    rows = {row["patient_id"]: row for row in read_csv(data_dir / PROCESSED_DIR / "manifest.csv")}
    opacity = rows[pseudonymize("opacity-patient", SECRET)]
    assert (opacity["label"], opacity["view"], opacity["sex"]) == ("PNEUMONIA", "PA", "F")
    assert (opacity["age"], opacity["pixel_spacing_mm"]) == ("51", "0.143")
    assert rows[pseudonymize("not-normal-patient", SECRET)]["label"] == "NORMAL"
    assert rows[pseudonymize("normal-patient", SECRET)]["age"] == "90"  # 95, grouped

    output = data_dir / PROCESSED_DIR
    written = (output / "manifest.csv").read_text() + (output / "rejected.csv").read_text()
    file_names = " ".join(path.name for path in output.rglob("*"))
    for patient_id in datasets:
        assert patient_id not in written
        assert patient_id not in file_names


def test_ingest_saves_the_converted_pixels(raw_rsna):
    data_dir, datasets = raw_rsna

    manifest, _ = ingest(data_dir, CONFIG, SECRET)

    row = next(row for row in manifest if row.label == "PNEUMONIA")
    saved = np.asarray(Image.open(data_dir / row.path))
    assert np.array_equal(saved, to_uint8(datasets["opacity-patient"]))


def test_ingest_lists_rejected_files_with_reasons(raw_rsna):
    data_dir, _ = raw_rsna

    _, rejected = ingest(data_dir, CONFIG, SECRET)

    assert [row.reasons for row in rejected] == ["age 155 is above 120"]
    assert read_csv(data_dir / PROCESSED_DIR / "rejected.csv")[0]["reasons"] == (
        "age 155 is above 120"
    )


def test_ingest_replaces_earlier_images(raw_rsna):
    data_dir, _ = raw_rsna
    stale = data_dir / PROCESSED_DIR / "images" / "stale.png"
    stale.parent.mkdir(parents=True)
    stale.touch()

    ingest(data_dir, CONFIG, SECRET)

    assert not stale.exists()
