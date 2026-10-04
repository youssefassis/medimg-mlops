"""The classifier: ResNet18 with one output per label (NORMAL, PNEUMONIA)."""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from mlops.data.chest_xray import LABELS


def build_model(pretrained: bool = True) -> nn.Module:
    """`pretrained` loads the ImageNet weights (downloaded once, then cached by torch)."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model
