import pytest

from mlops import object_storage
from mlops.data_validation import landing_batch


def test_counts_files_in_batch(monkeypatch):
    monkeypatch.setattr(
        object_storage, "list_object_keys", lambda bucket, prefix: ["a.dcm", "b.dcm"]
    )
    assert landing_batch.count_landing_files("batch_001") == 2


def test_empty_batch_fails_loudly(monkeypatch):
    monkeypatch.setattr(object_storage, "list_object_keys", lambda bucket, prefix: [])
    with pytest.raises(FileNotFoundError, match="batch_001"):
        landing_batch.count_landing_files("batch_001")
