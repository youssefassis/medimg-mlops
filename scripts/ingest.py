"""Turns the raw DICOM files of the source set in config/ingestion.yaml into de-identified PNGs
and a manifest, in data/processed/.

Needs DEID_SECRET, the key for patient pseudonyms (see .env.example):
    uv run --env-file .env python scripts/ingest.py
"""

import os
from collections import Counter
from pathlib import Path

from mlops.config import IngestionConfig, load_config
from mlops.data.ingestion import PROCESSED_DIR, ingest
from mlops.data.sources import list_files

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_config(ROOT / "config" / "ingestion.yaml", IngestionConfig)
    secret = os.environ.get("DEID_SECRET", "")
    if secret in ("", "change-me"):
        raise SystemExit("Set DEID_SECRET in .env first (see .env.example)")

    data_dir = ROOT / config.data_dir
    files = list_files(config.source, data_dir)
    manifest, rejected = ingest(files, data_dir, config, secret)

    labels = Counter(row.label for row in manifest)
    print(f"{len(manifest)} images in {(data_dir / PROCESSED_DIR).relative_to(ROOT)}/")
    print(f"  labels: {dict(sorted(labels.items()))}")
    print(f"  views: {dict(sorted(Counter(row.view for row in manifest).items()))}")
    print(f"{len(rejected)} rejected, see rejected.csv")
    for reason, count in Counter(row.reasons for row in rejected).most_common():
        print(f"  {count} × {reason}")


if __name__ == "__main__":
    main()
