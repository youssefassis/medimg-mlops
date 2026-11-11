import csv

import pytest

from mlops.data.sources import list_files
from mlops.data.sources.rsna import RAW_DIR, read_classes, to_label


@pytest.fixture
def raw_rsna(tmp_path):
    """Writes the class file of a tiny RSNA folder and returns the data folder."""
    raw_dir = tmp_path / RAW_DIR
    raw_dir.mkdir(parents=True)
    with (raw_dir / "stage_2_detailed_class_info.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("patientId", "class"))
        writer.writerows(
            [
                ("opacity-patient", "Lung Opacity"),
                ("opacity-patient", "Lung Opacity"),  # a second box: the ID repeats
                ("normal-patient", "Normal"),
            ]
        )
    return tmp_path


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
    assert read_classes(raw_rsna) == {
        "opacity-patient": "Lung Opacity",
        "normal-patient": "Normal",
    }


def test_lists_each_patient_file_with_its_label(raw_rsna):
    files = list_files("rsna", raw_rsna)

    assert [(file.path.name, file.label, file.source_class) for file in files] == [
        ("normal-patient.dcm", "NORMAL", "Normal"),
        ("opacity-patient.dcm", "PNEUMONIA", "Lung Opacity"),
    ]
    assert files[0].path.parent == raw_rsna / RAW_DIR / "stage_2_train_images"


def test_unknown_source_is_named(tmp_path):
    with pytest.raises(ValueError, match="Unknown source 'other', expected one of \\['rsna'\\]"):
        list_files("other", tmp_path)
