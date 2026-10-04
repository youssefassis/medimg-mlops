from pathlib import Path

import mlflow
import pytest
import torch
from PIL import Image

from mlops.config import TrainingConfig
from mlops.data.batches import write_batches
from mlops.data.chest_xray import Sample
from mlops.evaluation.evaluate import evaluate_run
from mlops.training.train import class_weights, train


def make_images(data_dir: Path, split: str, label: str, n: int) -> list[Sample]:
    folder = data_dir / "raw" / "chest_xray" / split / label
    folder.mkdir(parents=True, exist_ok=True)
    samples = []
    for i in range(n):
        patient_id = f"IM-{i:04d}" if label == "NORMAL" else f"person{i}"
        name = f"{patient_id}-0001.jpeg" if label == "NORMAL" else f"{patient_id}_virus_1.jpeg"
        Image.new("L", (32, 32), color=0 if label == "NORMAL" else 255).save(folder / name)
        samples.append(Sample((folder / name).relative_to(data_dir), label, patient_id))
    return samples


def test_class_weights_balance_the_classes():
    samples = [Sample(Path(f"{i}.jpeg"), "PNEUMONIA", f"p{i}") for i in range(3)]
    samples.append(Sample(Path("n.jpeg"), "NORMAL", "n"))

    assert class_weights(samples).tolist() == pytest.approx([2.0, 2 / 3])


def test_train_then_evaluate_log_to_mlflow(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    batch = make_images(data_dir, "train", "NORMAL", 4) + make_images(
        data_dir, "train", "PNEUMONIA", 4
    )
    write_batches([batch], data_dir / "batches")
    make_images(data_dir, "test", "NORMAL", 2)
    make_images(data_dir, "test", "PNEUMONIA", 2)
    monkeypatch.chdir(tmp_path)  # MLflow stores artifacts under the working directory
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    config = TrainingConfig(
        data_dir="data",
        batch=1,
        val_fraction=0.5,
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        seed=0,
        device="cpu",
        experiment="test",
    )

    run_id = train(config, data_dir, pretrained=False)
    test_metrics = evaluate_run(run_id, data_dir, batch_size=4)

    run = mlflow.get_run(run_id)
    assert run.data.params["n_train"] == "4"
    assert run.data.params["n_test"] == "4"
    assert run.data.metrics["test_roc_auc"] == test_metrics["test_roc_auc"]
    assert {"train_loss", "val_roc_auc", "test_recall", "test_roc_auc"} <= run.data.metrics.keys()
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    assert model(torch.zeros(3, 3, 224, 224)).shape == (3, 2)
