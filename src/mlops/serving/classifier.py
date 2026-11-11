"""The model served by `mlflow models serve`: base64 DICOM files in, predictions out.

Logged from this file ("models from code"), with the trained network as an artifact. Each file
is checked and preprocessed exactly as in ingestion and training.
"""

import base64
import io

import mlflow
import pydicom
import torch
from mlflow.models import set_model
from PIL import Image
from pydantic import BaseModel
from pydicom.errors import InvalidDicomError

from mlops.data.chest_xray import LABELS
from mlops.data.dicom import to_uint8
from mlops.data.dicom_checks import header_problems
from mlops.data.transforms import build_transform


class Prediction(BaseModel):
    """A file with problems gets no prediction. Problems never quote the file, which holds
    patient data."""

    pneumonia: bool | None = None
    score: float | None = None  # probability of pneumonia
    problems: list[str] = []


class PneumoniaClassifier(mlflow.pyfunc.PythonModel):
    """Needs the artifact `network` (a PyTorch model logged by MLflow) and the model config
    `threshold` and `max_age`."""

    def load_context(self, context):
        self.network = mlflow.pytorch.load_model(context.artifacts["network"])
        self.threshold = context.model_config["threshold"]
        self.max_age = context.model_config["max_age"]

    def predict(self, model_input: list[str], params=None) -> list[Prediction]:
        return [self._predict_file(content) for content in model_input]

    @torch.no_grad()
    def _predict_file(self, content: str) -> Prediction:
        try:
            dataset = pydicom.dcmread(io.BytesIO(base64.b64decode(content)))
        except (ValueError, InvalidDicomError) as error:
            return Prediction(problems=[f"not a base64 DICOM file: {error}"])
        problems = header_problems(dataset, self.max_age)
        if "PixelData" not in dataset:
            problems.append("no pixel data")
        if problems:
            return Prediction(problems=problems)

        tensor = build_transform()(Image.fromarray(to_uint8(dataset))).unsqueeze(0)
        probabilities = torch.softmax(self.network(tensor), dim=1)
        score = probabilities[0, LABELS.index("PNEUMONIA")].item()
        return Prediction(pneumonia=score >= self.threshold, score=score)


set_model(PneumoniaClassifier())
