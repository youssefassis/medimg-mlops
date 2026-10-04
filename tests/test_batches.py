import csv
from pathlib import Path

import pytest

from mlops.data.batches import (
    read_batch,
    split_by_patient,
    split_off_patients,
    write_batches,
)
from mlops.data.chest_xray import Sample


def make_samples(label: str, n_patients: int, images_per_patient: int = 2) -> list[Sample]:
    return [
        Sample(Path(f"{label}/p{patient}_{image}.jpeg"), label, f"{label}-p{patient}")
        for patient in range(n_patients)
        for image in range(images_per_patient)
    ]


SAMPLES = make_samples("NORMAL", 10) + make_samples("PNEUMONIA", 20)


def test_split_keeps_each_patient_in_one_batch():
    batches = split_by_patient(SAMPLES, n_batches=5, seed=0)

    seen = [{sample.patient_id for sample in batch} for batch in batches]
    for i, patients in enumerate(seen):
        for other in seen[i + 1 :]:
            assert patients.isdisjoint(other)


def test_split_uses_every_sample_once():
    batches = split_by_patient(SAMPLES, n_batches=5, seed=0)

    all_samples = [sample for batch in batches for sample in batch]
    assert sorted(all_samples, key=str) == sorted(SAMPLES, key=str)


def test_split_gives_balanced_batches_with_both_classes():
    batches = split_by_patient(SAMPLES, n_batches=5, seed=0)

    for batch in batches:
        labels = [sample.label for sample in batch]
        assert labels.count("NORMAL") == 4
        assert labels.count("PNEUMONIA") == 8


def test_split_is_reproducible_with_the_same_seed():
    assert split_by_patient(SAMPLES, 5, seed=1) == split_by_patient(SAMPLES, 5, seed=1)
    assert split_by_patient(SAMPLES, 5, seed=1) != split_by_patient(SAMPLES, 5, seed=2)


def test_split_fails_with_fewer_patients_than_batches():
    with pytest.raises(ValueError, match="NORMAL has 3 patients"):
        split_by_patient(make_samples("NORMAL", 3), n_batches=5, seed=0)


def test_split_fails_when_a_patient_has_both_labels():
    samples = [
        Sample(Path("a.jpeg"), "NORMAL", "p1"),
        Sample(Path("b.jpeg"), "PNEUMONIA", "p1"),
    ]
    with pytest.raises(ValueError, match="p1"):
        split_by_patient(samples, n_batches=1, seed=0)


def test_write_batches_writes_one_csv_per_batch(tmp_path):
    (tmp_path / "batch_9.csv").touch()
    batches = split_by_patient(SAMPLES, n_batches=2, seed=0)

    paths = write_batches(batches, tmp_path)

    assert paths == [tmp_path / "batch_1.csv", tmp_path / "batch_2.csv"]
    assert sorted(tmp_path.iterdir()) == paths
    with paths[0].open() as file:
        rows = list(csv.DictReader(file))
    assert rows == [
        {"path": str(sample.path), "label": sample.label, "patient_id": sample.patient_id}
        for sample in batches[0]
    ]


def test_split_off_patients_holds_out_a_fraction_of_each_class():
    kept, held_out = split_off_patients(SAMPLES, fraction=0.2, seed=0)

    assert {s.patient_id for s in kept}.isdisjoint({s.patient_id for s in held_out})
    assert sorted(kept + held_out, key=str) == sorted(SAMPLES, key=str)
    held_out_labels = [sample.label for sample in held_out]
    assert held_out_labels.count("NORMAL") == 2 * 2  # 2 of 10 patients, 2 images each
    assert held_out_labels.count("PNEUMONIA") == 4 * 2  # 4 of 20 patients


def test_split_off_patients_keeps_a_patient_on_each_side():
    kept, held_out = split_off_patients(make_samples("NORMAL", 2), fraction=0.01, seed=0)

    assert len(kept) == len(held_out) == 2


@pytest.mark.parametrize("fraction", [0, 1, 1.5])
def test_split_off_patients_fails_on_bad_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        split_off_patients(SAMPLES, fraction=fraction, seed=0)


def test_read_batch_reads_what_write_batches_wrote(tmp_path):
    batches = split_by_patient(SAMPLES, n_batches=2, seed=0)

    paths = write_batches(batches, tmp_path)

    assert [read_batch(path) for path in paths] == batches
