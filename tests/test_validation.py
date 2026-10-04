from pathlib import Path

import pytest
from PIL import Image

from mlops.data.chest_xray import Sample
from mlops.data.validation import validate_batch


@pytest.fixture
def data_dir(tmp_path):
    for name in ["n1.jpeg", "p1.jpeg"]:
        Image.new("L", (8, 8)).save(tmp_path / name)
    (tmp_path / "broken.jpeg").write_text("not an image")
    return tmp_path


NORMAL = Sample(Path("n1.jpeg"), "NORMAL", "n1")
PNEUMONIA = Sample(Path("p1.jpeg"), "PNEUMONIA", "p1")


def test_valid_batch_passes(data_dir):
    validate_batch([NORMAL, PNEUMONIA], data_dir, test_patients={"other"})


@pytest.mark.parametrize(
    ("samples", "test_patients", "problem"),
    [
        ([NORMAL, PNEUMONIA, Sample(Path("gone.jpeg"), "NORMAL", "n2")], set(), "gone.jpeg"),
        ([NORMAL, PNEUMONIA, Sample(Path("broken.jpeg"), "NORMAL", "n2")], set(), "broken.jpeg"),
        ([NORMAL], set(), r"no images for \['PNEUMONIA'\]"),
        ([NORMAL, PNEUMONIA, Sample(Path("n1.jpeg"), "COVID", "c1")], set(), "unknown labels"),
        ([NORMAL, PNEUMONIA], {"p1"}, "1 patients also in test"),
    ],
)
def test_invalid_batch_fails(data_dir, samples, test_patients, problem):
    with pytest.raises(ValueError, match=problem):
        validate_batch(samples, data_dir, test_patients)


def test_all_problems_are_reported(data_dir):
    samples = [NORMAL, Sample(Path("gone.jpeg"), "NORMAL", "n2")]

    with pytest.raises(ValueError, match=r"(?s)gone\.jpeg.*no images for"):
        validate_batch(samples, data_dir, test_patients=set())
