"""Load pipeline settings from the repository's `config/` folder."""

from pathlib import Path

import yaml

# Resolves to <repo>/config in an editable install; Docker mounts config/ at the same place.
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_config(name: str) -> dict:
    """Return the parsed `config/<name>.yaml`; fails if the file is missing or empty."""
    config = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
    if not isinstance(config, dict):
        raise ValueError(f"{CONFIG_DIR / name}.yaml must contain a YAML mapping")
    return config
