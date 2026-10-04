"""Loads the YAML files from config/ into dataclasses."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BatchesConfig:
    data_dir: str  # relative to the project root
    n_batches: int
    seed: int


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: str  # relative to the project root
    batch: int
    val_fraction: float
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    device: str
    experiment: str


@dataclass(frozen=True)
class PromotionConfig:
    model_name: str
    min_recall: float
    min_specificity: float
    max_drop: float


def load_config[T](path: Path, config_class: type[T]) -> T:
    """Fails if a key is missing or unknown."""
    values = yaml.safe_load(path.read_text())
    try:
        return config_class(**values)
    except TypeError as error:
        raise ValueError(f"Invalid config {path}: {error}") from error
