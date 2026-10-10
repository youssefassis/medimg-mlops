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
class IngestionConfig:
    data_dir: str  # relative to the project root
    sample_size: int  # how many RSNA training images to ingest; all of them if larger
    seed: int
    max_age: int  # older patients are data entry errors
    min_contrast: float  # lowest standard deviation of the pixels, on 0–255
    max_near_black: float  # highest share of near-black pixels


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
