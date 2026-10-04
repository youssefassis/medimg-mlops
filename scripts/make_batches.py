"""Splits train/ and val/ into patient-level batches, written as CSVs to data/batches/."""

from collections import Counter
from pathlib import Path

from mlops.config import load_batches_config
from mlops.data.batches import split_by_patient, write_batches
from mlops.data.chest_xray import list_samples

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_batches_config(ROOT / "config" / "batches.yaml")
    data_dir = ROOT / config.data_dir
    samples = list_samples(data_dir, "train") + list_samples(data_dir, "val")

    batches = split_by_patient(samples, config.n_batches, config.seed)
    paths = write_batches(batches, data_dir / "batches")

    for path, batch in zip(paths, batches, strict=True):
        labels = Counter(sample.label for sample in batch)
        print(f"{path.relative_to(ROOT)}: {len(batch)} images {dict(sorted(labels.items()))}")


if __name__ == "__main__":
    main()
