"""Loads the YAML files from config/."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BatchesConfig:
    data_dir: Path
    n_batches: int
    seed: int


def load_batches_config(path: Path) -> BatchesConfig:
    config = yaml.safe_load(path.read_text())
    try:
        return BatchesConfig(
            data_dir=Path(config["data_dir"]),
            n_batches=int(config["n_batches"]),
            seed=int(config["seed"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid config {path}: {error!r}") from error
