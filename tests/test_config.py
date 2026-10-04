from pathlib import Path

import pytest

from mlops.config import BatchesConfig, load_batches_config


def test_load_batches_config(tmp_path):
    path = tmp_path / "batches.yaml"
    path.write_text("data_dir: data\nn_batches: 5\nseed: 42\n")

    assert load_batches_config(path) == BatchesConfig(Path("data"), n_batches=5, seed=42)


def test_load_batches_config_fails_on_missing_key(tmp_path):
    path = tmp_path / "batches.yaml"
    path.write_text("data_dir: data\nn_batches: 5\n")

    with pytest.raises(ValueError, match="seed"):
        load_batches_config(path)


def test_project_batches_config_loads():
    load_batches_config(Path(__file__).parents[1] / "config" / "batches.yaml")
