"""The classes to predict, and the image samples every step works on."""

from dataclasses import dataclass
from pathlib import Path

NEGATIVE, POSITIVE = "NORMAL", "PNEUMONIA"
LABELS = (NEGATIVE, POSITIVE)  # in the model's output order: NEGATIVE is 0, POSITIVE is 1


@dataclass(frozen=True)
class Sample:
    path: Path  # relative to the data folder
    label: str
    patient_id: str
