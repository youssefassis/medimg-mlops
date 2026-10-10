"""Splits the ingested RSNA images into a held-out test set and patient-level batches.

Reads data/processed/rsna/manifest.csv (written by scripts/ingest_rsna.py) and writes the CSVs to
data/batches/. Every split keeps the share of each label and view.
"""

from collections import Counter
from pathlib import Path

from mlops.config import BatchesConfig, load_config
from mlops.data.batches import (
    heldout_path,
    split_by_patient,
    split_off_patients,
    write_batches,
    write_rows,
)
from mlops.data.lineage import digest
from mlops.data.rsna import ManifestRow, by_label_and_view, read_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_config(ROOT / "config" / "batches.yaml", BatchesConfig)
    data_dir = ROOT / config.data_dir
    rows = read_manifest(data_dir)

    train_rows, test_rows = split_off_patients(
        rows, config.test_fraction, config.seed, by_label_and_view
    )
    batches = split_by_patient(train_rows, config.n_batches, config.seed, by_label_and_view)
    write_rows(heldout_path(data_dir), ManifestRow, test_rows)
    paths = write_batches(batches, data_dir / "batches")

    for path, split in [(heldout_path(data_dir), test_rows), *zip(paths, batches, strict=True)]:
        labels = dict(sorted(Counter(row.label for row in split).items()))
        views = dict(sorted(Counter(row.view for row in split).items()))
        print(f"{path.relative_to(ROOT)} ({digest(path)}): {len(split)} images {labels} {views}")


if __name__ == "__main__":
    main()
