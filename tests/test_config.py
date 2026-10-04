from pathlib import Path

import pytest

from mlops.config import BatchesConfig, load_config

CONFIG_DIR = Path(__file__).parents[1] / "config"


def test_load_config(tmp_path):
    path = tmp_path / "batches.yaml"
    path.write_text("data_dir: data\nn_batches: 5\nseed: 42\n")

    assert load_config(path, BatchesConfig) == BatchesConfig("data", n_batches=5, seed=42)


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("data_dir: data\nn_batches: 5\n", "seed"),
        ("data_dir: data\nn_batches: 5\nseed: 42\nextra: 1\n", "extra"),
    ],
)
def test_load_config_fails_on_missing_or_unknown_key(tmp_path, content, error):
    path = tmp_path / "batches.yaml"
    path.write_text(content)

    with pytest.raises(ValueError, match=error):
        load_config(path, BatchesConfig)


def test_project_batches_config_loads():
    load_config(CONFIG_DIR / "batches.yaml", BatchesConfig)
