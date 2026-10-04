"""Splits train/ and val/ into patient-level batches, written as CSVs to data/batches/.

Patients that also have images in test/ are left out, so they can't leak into training.
"""

from collections import Counter
from pathlib import Path

from mlops.config import BatchesConfig, load_config
from mlops.data.batches import drop_patients, split_by_patient, write_batches
from mlops.data.chest_xray import list_samples

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_config(ROOT / "config" / "batches.yaml", BatchesConfig)
    data_dir = ROOT / config.data_dir
    samples = list_samples(data_dir, "train") + list_samples(data_dir, "val")
    test_patients = {sample.patient_id for sample in list_samples(data_dir, "test")}
    kept = drop_patients(samples, test_patients)
    print(f"Left out {len(samples) - len(kept)} images of patients also in test/")

    batches = split_by_patient(kept, config.n_batches, config.seed)
    paths = write_batches(batches, data_dir / "batches")

    for path, batch in zip(paths, batches, strict=True):
        labels = Counter(sample.label for sample in batch)
        print(f"{path.relative_to(ROOT)}: {len(batch)} images {dict(sorted(labels.items()))}")


if __name__ == "__main__":
    main()
