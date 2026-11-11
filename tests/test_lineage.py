import mlflow

from mlops.data.lineage import digest, heldout_digest, log_dataset, run_datasets


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


def test_run_datasets_lists_each_input_with_its_use(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    for name in ("batch_1", "test"):
        (tmp_path / f"{name}.csv").write_text(f"path,label\n{name}.png,NORMAL\n")

    with mlflow.start_run() as run:
        batch_digest = log_dataset(tmp_path / "batch_1.csv", context="training")
        test_digest = log_dataset(tmp_path / "test.csv", context="testing")

    logged = mlflow.get_run(run.info.run_id)
    assert sorted(run_datasets(logged), key=lambda d: d["name"]) == [
        {"context": "training", "name": "batch_1", "digest": batch_digest},
        {"context": "testing", "name": "test", "digest": test_digest},
    ]
    assert heldout_digest(logged) == test_digest


def test_run_without_test_set_has_no_heldout_digest(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")

    with mlflow.start_run() as run:
        pass

    assert heldout_digest(mlflow.get_run(run.info.run_id)) is None
