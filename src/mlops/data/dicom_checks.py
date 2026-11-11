"""Checks a DICOM image before it enters the dataset."""

import re
from dataclasses import dataclass

import numpy as np
from pydicom import Dataset

SEXES = {"F", "M", "O"}
NEAR_BLACK = 10  # pixel values below this count as black, on 0–255

# DICOM ages are 3 digits and a unit, e.g. "051Y" or "006M".
AGE_PATTERN = re.compile(r"^(\d{1,3})([DWMY]?)$")
UNITS_PER_YEAR = {"D": 365, "W": 52, "M": 12, "Y": 1}


def parse_age(value: str) -> int:
    """Returns the age in whole years.

    Also accepts a bare number like "51", read as years: some datasets, like RSNA, write ages
    that way.
    """
    match = AGE_PATTERN.match(value.strip())
    if match is None:
        raise ValueError(f"invalid age {value!r}")
    number, unit = int(match.group(1)), match.group(2) or "Y"
    return number // UNITS_PER_YEAR[unit]


@dataclass(frozen=True)
class HeaderChecks:
    """What a header must contain, from config/ingestion.yaml."""

    modalities: list[str]
    views: list[str]
    max_age: int


def header_problems(dataset: Dataset, checks: HeaderChecks) -> list[str]:
    """Returns what is wrong with the header; empty if nothing."""
    problems = []
    if (modality := dataset.get("Modality")) not in checks.modalities:
        problems.append(f"modality {modality!r} is not one of {checks.modalities}")
    if (view := dataset.get("ViewPosition")) not in checks.views:
        problems.append(f"view {view!r} is not one of {checks.views}")
    if (sex := dataset.get("PatientSex")) not in SEXES:
        problems.append(f"unknown sex {sex!r}")
    if (channels := dataset.get("SamplesPerPixel")) != 1:
        problems.append(f"{channels} channels instead of 1")

    try:
        age = parse_age(dataset.get("PatientAge", ""))
    except ValueError as error:
        problems.append(str(error))
    else:
        if age > checks.max_age:
            problems.append(f"age {age} is above {checks.max_age}")

    spacing = dataset.get("PixelSpacing")
    if spacing is None or len(spacing) != 2 or spacing[0] <= 0 or spacing[0] != spacing[1]:
        problems.append(f"pixel spacing {spacing} is missing or not square")
    return problems


def image_problems(image: np.ndarray, min_contrast: float, max_near_black: float) -> list[str]:
    """Returns what is wrong with an 8-bit image; empty if nothing."""
    problems = []
    if (contrast := image.std()) < min_contrast:
        problems.append(f"contrast {contrast:.1f} is below {min_contrast}")
    if (near_black := (image < NEAR_BLACK).mean()) > max_near_black:
        problems.append(f"{near_black:.0%} of pixels are near black, above {max_near_black:.0%}")
    return problems
