"""Image preprocessing, shared by training and serving."""

import torch
from torchvision.transforms import v2

IMAGE_SIZE = 224
# The pretrained ResNet18 weights expect images normalized like ImageNet.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform() -> v2.Compose:
    """Turns a PIL image (grayscale or RGB) into a normalized 3×224×224 tensor."""
    return v2.Compose(
        [
            v2.Grayscale(num_output_channels=3),
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
