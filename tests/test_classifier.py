import base64
import io

import mlflow
import pytest
import torch
from pydicom import dcmwrite
from torch import nn

from mlops.training.train import log_model


class FixedScore(nn.Module):
    """Gives every image the same pneumonia probability."""

    def __init__(self, score: float):
        super().__init__()
        self.register_buffer("logits", torch.log(torch.tensor([1 - score, score])))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(1, 2, 3)).unsqueeze(1) * 0 + self.logits


@pytest.fixture
def load_classifier(tmp_path, monkeypatch):
    """Returns a function logging a classifier around FixedScore and loading it back."""
    monkeypatch.chdir(tmp_path)  # MLflow stores artifacts under the working directory
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")

    def load(score: float, threshold: float = 0.6):
        with mlflow.start_run() as run:
            log_model(FixedScore(score), threshold, max_age=120)
        return mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")

    return load


@pytest.fixture
def dicom_file(make_dicom, noise, chest_tags):
    """Returns a function encoding a DICOM file as base64; keyword args change the tags."""

    def make(**tags) -> str:
        buffer = io.BytesIO()
        dcmwrite(buffer, make_dicom(noise, **(chest_tags | tags)), enforce_file_format=True)
        return base64.b64encode(buffer.getvalue()).decode()

    return make


def test_predicts_at_the_threshold(load_classifier, dicom_file):
    (prediction,) = load_classifier(score=0.7).predict([dicom_file()])

    assert prediction.pneumonia is True
    assert prediction.score == pytest.approx(0.7)
    assert prediction.problems == []


def test_score_below_the_threshold_is_not_pneumonia(load_classifier, dicom_file):
    (prediction,) = load_classifier(score=0.5).predict([dicom_file()])

    assert prediction.pneumonia is False


def test_file_failing_the_header_checks_gets_only_problems(load_classifier, dicom_file):
    good, bad = load_classifier(score=0.7).predict([dicom_file(), dicom_file(Modality="CT")])

    assert good.pneumonia is True
    assert (bad.pneumonia, bad.score) == (None, None)
    assert bad.problems == ["modality 'CT' is not a radiograph"]


def test_input_that_is_not_dicom_gets_only_problems(load_classifier):
    (prediction,) = load_classifier(score=0.7).predict([base64.b64encode(b"no image").decode()])

    assert prediction.pneumonia is None
    assert prediction.problems[0].startswith("not a base64 DICOM file")
