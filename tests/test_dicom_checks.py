import numpy as np
import pytest

from mlops.data.dicom_checks import header_problems, image_problems, parse_age


@pytest.mark.parametrize(
    ("value", "years"),
    [("051Y", 51), ("51", 51), ("006M", 0), ("018M", 1), ("104W", 2), ("400D", 1)],
)
def test_parse_age(value, years):
    assert parse_age(value) == years


@pytest.mark.parametrize("value", ["", "51 years", "Y051", "1234"])
def test_parse_age_fails_on_invalid_value(value):
    with pytest.raises(ValueError, match="invalid age"):
        parse_age(value)


def test_valid_header_has_no_problems(make_dicom, noise, chest_tags):
    assert header_problems(make_dicom(noise, **chest_tags), max_age=120) == []


@pytest.mark.parametrize(
    ("tags", "problem"),
    [
        ({"Modality": "CT"}, "modality 'CT'"),
        ({"ViewPosition": "LL"}, "unknown view 'LL'"),
        ({"PatientSex": ""}, "unknown sex ''"),
        ({"PatientAge": "155"}, "age 155 is above 120"),
        ({"PatientAge": "old"}, "invalid age 'old'"),
        ({"PixelSpacing": [0.1, 0.2]}, "not square"),
    ],
)
def test_header_problems(make_dicom, noise, chest_tags, tags, problem):
    dataset = make_dicom(noise, **(chest_tags | tags))

    assert [problem in found for found in header_problems(dataset, max_age=120)] == [True]


def test_missing_pixel_spacing_is_a_problem(make_dicom, noise, chest_tags):
    dataset = make_dicom(noise, **chest_tags)
    del dataset.PixelSpacing

    assert header_problems(dataset, max_age=120) == ["pixel spacing None is missing or not square"]


def test_varied_image_has_no_problems(noise):
    assert image_problems(noise, min_contrast=10, max_near_black=0.7) == []


def test_flat_image_lacks_contrast():
    image = np.full((4, 4), 128, dtype=np.uint8)

    assert image_problems(image, min_contrast=10, max_near_black=0.7) == [
        "contrast 0.0 is below 10"
    ]


def test_mostly_black_image_is_a_problem():
    image = np.zeros((10, 10), dtype=np.uint8)
    image[:2] = 255  # 80% black, with enough contrast

    assert image_problems(image, min_contrast=10, max_near_black=0.7) == [
        "80% of pixels are near black, above 70%"
    ]
