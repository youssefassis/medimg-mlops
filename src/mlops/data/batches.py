"""Splits samples by patient, into batches that simulate data arriving over time.

The splits work on any rows with a label and a patient_id: Samples, or the RSNA manifest rows.
"""

import csv
import random
from collections import defaultdict
from collections.abc import Callable, Hashable
from dataclasses import asdict, fields
from pathlib import Path
from typing import Protocol

from mlops.data.chest_xray import Sample


class PatientRow(Protocol):
    @property
    def label(self) -> str: ...

    @property
    def patient_id(self) -> str: ...


def drop_patients(samples: list[Sample], patient_ids: set[str]) -> list[Sample]:
    """Removes every image of the given patients, e.g. the ones also in the test set."""
    return [sample for sample in samples if sample.patient_id not in patient_ids]


def by_label(row: PatientRow) -> Hashable:
    return row.label


def split_by_patient[T: PatientRow](
    rows: list[T], n_batches: int, seed: int, stratum: Callable[[T], Hashable] = by_label
) -> list[list[T]]:
    """Splits rows into n_batches of similar size, with each patient in one batch only.

    Each stratum (by default the label) is split on its own, so every batch keeps its share.
    """
    if n_batches < 1:
        raise ValueError(f"n_batches must be at least 1, got {n_batches}")

    batches: list[list[T]] = [[] for _ in range(n_batches)]
    rng = random.Random(seed)
    for group, by_patient in sorted(_group_by_stratum_and_patient(rows, stratum).items()):
        if len(by_patient) < n_batches:
            raise ValueError(
                f"{group} has {len(by_patient)} patients, too few for {n_batches} batches"
            )
        patient_ids = sorted(by_patient)
        rng.shuffle(patient_ids)
        counts = [0] * n_batches
        for patient_id in patient_ids:
            smallest = counts.index(min(counts))
            batches[smallest].extend(by_patient[patient_id])
            counts[smallest] += len(by_patient[patient_id])
    return [sorted(batch, key=lambda row: str(row.path)) for batch in batches]


def split_off_patients[T: PatientRow](
    rows: list[T], fraction: float, seed: int, stratum: Callable[[T], Hashable] = by_label
) -> tuple[list[T], list[T]]:
    """Holds out about `fraction` of the patients of each stratum, e.g. for validation.

    Returns (kept, held_out). Each stratum keeps at least one patient on each side.
    """
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")

    kept: list[T] = []
    held_out: list[T] = []
    rng = random.Random(seed)
    for group, by_patient in sorted(_group_by_stratum_and_patient(rows, stratum).items()):
        if len(by_patient) < 2:
            raise ValueError(f"{group} has {len(by_patient)} patient, need at least 2 to split")
        patient_ids = sorted(by_patient)
        rng.shuffle(patient_ids)
        n_held_out = min(max(round(len(patient_ids) * fraction), 1), len(patient_ids) - 1)
        for i, patient_id in enumerate(patient_ids):
            (held_out if i < n_held_out else kept).extend(by_patient[patient_id])
    return kept, held_out


def write_rows(path: Path, row_class: type, rows: list) -> None:
    """Writes dataclass rows to a CSV, with one column per field."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field.name for field in fields(row_class)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_batches(batches: list[list], out_dir: Path) -> list[Path]:
    """Writes one CSV per batch (batch_1.csv, batch_2.csv, ...) and removes older ones."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("batch_*.csv"):
        old_file.unlink()

    paths = []
    for number, batch in enumerate(batches, start=1):
        path = out_dir / f"batch_{number}.csv"
        write_rows(path, type(batch[0]), batch)
        paths.append(path)
    return paths


def batch_path(data_dir: Path, number: int) -> Path:
    return data_dir / "batches" / f"batch_{number}.csv"


def read_batch(path: Path) -> list[Sample]:
    """Reads a batch. Columns other than path, label and patient_id are ignored."""
    with path.open(newline="") as file:
        return [
            Sample(Path(row["path"]), row["label"], row["patient_id"])
            for row in csv.DictReader(file)
        ]


def _group_by_stratum_and_patient[T: PatientRow](
    rows: list[T], stratum: Callable[[T], Hashable]
) -> dict[Hashable, dict[str, list[T]]]:
    groups: dict[Hashable, dict[str, list[T]]] = defaultdict(lambda: defaultdict(list))
    patient_groups: dict[str, Hashable] = {}
    for row in rows:
        group = stratum(row)
        known_group = patient_groups.setdefault(row.patient_id, group)
        if known_group != group:
            raise ValueError(
                f"Patient {row.patient_id} has images in two groups: {known_group} and {group}"
            )
        groups[group][row.patient_id].append(row)
    return groups
