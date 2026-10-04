from pathlib import Path

import pytest

from mlops.data.chest_xray import Sample, list_samples, read_patient_id


@pytest.mark.parametrize(
    ("file_name", "patient_id"),
    [
        ("person1_bacteria_1.jpeg", "person1"),
        ("person447_virus_921_1.jpeg", "person447"),
        ("IM-0115-0001.jpeg", "IM-0115"),
        ("IM-0656-0001-0001.jpeg", "IM-0656"),
        ("NORMAL2-IM-0684-0001.jpeg", "NORMAL2-IM-0684"),
    ],
)
def test_read_patient_id(file_name, patient_id):
    assert read_patient_id(file_name) == patient_id


def test_read_patient_id_fails_on_unknown_name():
    with pytest.raises(ValueError, match="scan.jpeg"):
        read_patient_id("scan.jpeg")


def test_list_samples_reads_label_from_folder(make_dataset):
    data_dir = make_dataset(
        [
            "train/PNEUMONIA/person1_virus_1.jpeg",
            "train/NORMAL/IM-0001-0001.jpeg",
            "test/NORMAL/IM-0002-0001.jpeg",
        ]
    )

    assert list_samples(data_dir, "train") == [
        Sample(Path("raw/chest_xray/train/NORMAL/IM-0001-0001.jpeg"), "NORMAL", "IM-0001"),
        Sample(Path("raw/chest_xray/train/PNEUMONIA/person1_virus_1.jpeg"), "PNEUMONIA", "person1"),
    ]


def test_list_samples_fails_when_split_is_empty(make_dataset):
    data_dir = make_dataset(["train/NORMAL/IM-0001-0001.jpeg"])

    with pytest.raises(FileNotFoundError, match="val"):
        list_samples(data_dir, "val")
