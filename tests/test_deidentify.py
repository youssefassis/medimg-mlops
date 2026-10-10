import pytest

from mlops.data.deidentify import pseudonymize


def test_same_patient_gets_same_pseudonym():
    assert pseudonymize("patient-1", "secret") == pseudonymize("patient-1", "secret")


def test_pseudonym_depends_on_patient_and_secret():
    pseudonyms = {
        pseudonymize("patient-1", "secret"),
        pseudonymize("patient-2", "secret"),
        pseudonymize("patient-1", "other secret"),
    }

    assert len(pseudonyms) == 3


def test_pseudonym_does_not_contain_the_id():
    assert "patient-1" not in pseudonymize("patient-1", "secret")


def test_empty_secret_is_refused():
    with pytest.raises(ValueError, match="secret is empty"):
        pseudonymize("patient-1", "")
