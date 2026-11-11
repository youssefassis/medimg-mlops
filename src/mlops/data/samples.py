"""The classes to predict, set in config/task.yaml, and the image samples every step works on."""

from dataclasses import dataclass
from pathlib import Path

from mlops.config import TaskConfig, load_config

_TASK = load_config(Path(__file__).resolve().parents[3] / "config" / "task.yaml", TaskConfig)
NEGATIVE, POSITIVE = _TASK.negative, _TASK.positive
LABELS = (NEGATIVE, POSITIVE)  # in the model's output order: NEGATIVE is 0, POSITIVE is 1


@dataclass(frozen=True)
class Sample:
    path: Path  # relative to the data folder
    label: str
    patient_id: str
