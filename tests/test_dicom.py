import numpy as np
import pytest

from mlops.data.dicom import to_uint8


def test_stretches_to_full_range(make_dicom):
    dataset = make_dicom(np.array([[10, 20], [30, 50]], dtype=np.uint8))

    assert to_uint8(dataset).tolist() == [[0, 64], [128, 255]]


def test_inverts_monochrome1(make_dicom):
    dataset = make_dicom(np.array([[0, 100]], dtype=np.uint8), photometric="MONOCHROME1")

    assert to_uint8(dataset).tolist() == [[255, 0]]


def test_applies_rescale_before_window(make_dicom):
    # Stored 1024 is 0 after rescale, so a window centred on 0 must clip around it.
    pixels = np.array([[0, 924, 1024, 1124, 4000]], dtype=np.uint16)
    dataset = make_dicom(
        pixels, RescaleSlope=1, RescaleIntercept=-1024, WindowCenter=0, WindowWidth=200
    )

    image = to_uint8(dataset)

    assert image[0, 0] == image[0, 1] == 0
    assert image[0, 3] == image[0, 4] == 255
    assert 0 < image[0, 2] < 255


def test_constant_image_is_black(make_dicom):
    dataset = make_dicom(np.full((2, 2), 7, dtype=np.uint8))

    assert to_uint8(dataset).tolist() == [[0, 0], [0, 0]]


def test_rejects_colour_images(make_dicom):
    dataset = make_dicom(np.zeros((2, 2, 3), dtype=np.uint8), photometric="RGB")

    with pytest.raises(ValueError, match="3 channels"):
        to_uint8(dataset)
