"""Records which data each MLflow run used, so every model points to its exact images."""

import hashlib
from pathlib import Path

import mlflow
from mlflow.data.dataset_source_registry import resolve_dataset_source
from mlflow.data.meta_dataset import MetaDataset


def digest(path: Path) -> str:
    """A short hash of the file's contents: it changes whenever any row changes.

    Rows of the RSNA manifest include the source file hash, so the digest covers which images
    are used, their labels, and their split.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def log_dataset(path: Path, context: str) -> str:
    """Logs the CSV as an input of the active MLflow run and returns its digest.

    context is how the run used it, e.g. "training" or "testing".
    """
    dataset_digest = digest(path)
    dataset = MetaDataset(
        source=resolve_dataset_source(str(path)), name=path.stem, digest=dataset_digest
    )
    mlflow.log_input(dataset, context=context)
    return dataset_digest
