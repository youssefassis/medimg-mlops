import mlflow

from mlops.data.lineage import digest, log_dataset


def test_digest_changes_when_a_row_changes(tmp_path):
    path = tmp_path / "batch_1.csv"
    path.write_text("path,label\na.png,NORMAL\n")
    before = digest(path)

    path.write_text("path,label\na.png,PNEUMONIA\n")

    assert digest(path) != before
    assert len(before) == 12


def test_log_dataset_adds_an_input_to_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    path = tmp_path / "test.csv"
    path.write_text("path,label\na.png,NORMAL\n")

    with mlflow.start_run() as run:
        logged_digest = log_dataset(path, context="testing")

    (dataset_input,) = mlflow.get_run(run.info.run_id).inputs.dataset_inputs
    assert (dataset_input.dataset.name, dataset_input.dataset.digest) == ("test", logged_digest)
    assert dataset_input.tags[0].value == "testing"
