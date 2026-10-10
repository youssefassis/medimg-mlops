"""Checks a batch before training on it."""

from pathlib import Path

from PIL import Image

from mlops.data.chest_xray import LABELS, Sample

MAX_PROBLEMS_SHOWN = 20


def validate_batch(samples: list[Sample], data_dir: Path, test_patients: set[str]) -> None:
    """Raises ValueError listing every problem found.

    Checks that every image opens, both classes are present, labels are known, and no patient
    also has images in the test set.
    """
    problems = [
        f"{sample.path}: {error}"
        for sample in samples
        if (error := _open_error(data_dir / sample.path))
    ]

    labels = {sample.label for sample in samples}
    if missing := set(LABELS) - labels:
        problems.append(f"no images for {sorted(missing)}")
    if unknown := labels - set(LABELS):
        problems.append(f"unknown labels {sorted(unknown)}")
    if shared := {sample.patient_id for sample in samples} & test_patients:
        problems.append(f"{len(shared)} patients also in the test set: {sorted(shared)[:5]}")

    if problems:
        shown = problems[:MAX_PROBLEMS_SHOWN]
        if len(problems) > MAX_PROBLEMS_SHOWN:
            shown.append(f"... and {len(problems) - MAX_PROBLEMS_SHOWN} more")
        raise ValueError("Invalid batch:\n" + "\n".join(shown))


def _open_error(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError) as error:  # PIL raises SyntaxError for some broken files
        return str(error)
    return None
