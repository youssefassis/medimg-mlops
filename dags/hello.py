"""Checks that Airflow runs and can read the dataset mounted from data/."""

import os
from pathlib import Path

from airflow.sdk import dag, task


@dag(schedule=None, tags=["setup"])
def hello():
    @task
    def count_images() -> dict[str, int]:
        dataset_dir = Path(os.environ["DATA_DIR"]) / "raw" / "chest_xray"
        counts = {
            f"{class_dir.parent.name}/{class_dir.name}": len(list(class_dir.glob("*.jpeg")))
            for class_dir in sorted(dataset_dir.glob("*/*"))
        }
        if not counts:
            raise FileNotFoundError(f"No images found in {dataset_dir}")
        print(counts)
        return counts

    count_images()


hello()
