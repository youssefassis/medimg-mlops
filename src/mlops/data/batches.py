"""Splits samples into batches by patient, to simulate data arriving over time."""

import csv
import random
from collections import defaultdict
from pathlib import Path

from mlops.data.chest_xray import Sample

CSV_COLUMNS = ("path", "label", "patient_id")


def split_by_patient(samples: list[Sample], n_batches: int, seed: int) -> list[list[Sample]]:
    """Splits samples into n_batches of similar size, with each patient in one batch only.

    Each class is split on its own, so every batch keeps the class balance.
    """
    if n_batches < 1:
        raise ValueError(f"n_batches must be at least 1, got {n_batches}")

    by_label = _group_by_label_and_patient(samples)
    batches: list[list[Sample]] = [[] for _ in range(n_batches)]
    rng = random.Random(seed)
    for label, by_patient in sorted(by_label.items()):
        if len(by_patient) < n_batches:
            raise ValueError(
                f"{label} has {len(by_patient)} patients, too few for {n_batches} batches"
            )
        patient_ids = sorted(by_patient)
        rng.shuffle(patient_ids)
        label_counts = [0] * n_batches
        for patient_id in patient_ids:
            smallest = label_counts.index(min(label_counts))
            batches[smallest].extend(by_patient[patient_id])
            label_counts[smallest] += len(by_patient[patient_id])
    return [sorted(batch, key=lambda sample: sample.path) for batch in batches]


def write_batches(batches: list[list[Sample]], out_dir: Path) -> list[Path]:
    """Writes one CSV per batch (batch_1.csv, batch_2.csv, ...) and removes older ones."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("batch_*.csv"):
        old_file.unlink()

    paths = []
    for number, batch in enumerate(batches, start=1):
        path = out_dir / f"batch_{number}.csv"
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_COLUMNS)
            writer.writerows((sample.path, sample.label, sample.patient_id) for sample in batch)
        paths.append(path)
    return paths


def _group_by_label_and_patient(samples: list[Sample]) -> dict[str, dict[str, list[Sample]]]:
    by_label: dict[str, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
    patient_labels: dict[str, str] = {}
    for sample in samples:
        known_label = patient_labels.setdefault(sample.patient_id, sample.label)
        if known_label != sample.label:
            raise ValueError(f"Patient {sample.patient_id} has images labelled both ways")
        by_label[sample.label][sample.patient_id].append(sample)
    return by_label
