import pytest

from medimg_mlops import settings


def test_missing_config_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CONFIG_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        settings.load_config("example")


def test_non_mapping_config_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CONFIG_DIR", tmp_path)
    (tmp_path / "example.yaml").write_text("- just a list\n")
    with pytest.raises(ValueError, match="mapping"):
        settings.load_config("example")
