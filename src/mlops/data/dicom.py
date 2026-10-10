"""Turns DICOM pixel data into 8-bit grayscale images."""

import numpy as np
from pydicom import Dataset
from pydicom.pixels import apply_modality_lut, apply_voi_lut


def to_uint8(dataset: Dataset) -> np.ndarray:
    """Returns the image as 8-bit grayscale, with dense tissue (bone) bright.

    Steps, in the order the DICOM standard defines them:
    1. Modality LUT: stored values to real units (RescaleSlope and RescaleIntercept).
    2. VOI LUT: the window a radiologist would view the image with, if the file has one.
    3. Stretch the result to 0–255, per image.
    4. Invert MONOCHROME1 images, where low values are bright.
    """
    if dataset.SamplesPerPixel != 1:
        raise ValueError(f"Expected a grayscale image, got {dataset.SamplesPerPixel} channels")

    pixels = apply_modality_lut(dataset.pixel_array, dataset)
    pixels = apply_voi_lut(pixels, dataset).astype(np.float64)

    low, high = pixels.min(), pixels.max()
    if high == low:
        image = np.zeros(pixels.shape, dtype=np.uint8)
    else:
        image = np.round((pixels - low) / (high - low) * 255).astype(np.uint8)

    if dataset.PhotometricInterpretation == "MONOCHROME1":
        image = 255 - image
    return image
