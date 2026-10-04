from pathlib import Path

import pytest


@pytest.fixture
def make_dataset(tmp_path):
    """Creates empty image files under tmp_path/raw/chest_xray/ and returns tmp_path."""

    def make(files: list[str]) -> Path:
        for file in files:
            path = tmp_path / "raw" / "chest_xray" / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return tmp_path

    return make
