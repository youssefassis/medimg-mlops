import torch

from mlops.training.model import build_model


def test_model_outputs_one_score_per_label():
    model = build_model(pretrained=False)

    assert model(torch.zeros(4, 3, 224, 224)).shape == (4, 2)
