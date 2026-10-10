# Changelog

Notable changes, one entry per step. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Clinical validation: decision threshold chosen on validation, 95% confidence intervals, metrics per subgroup, and a gate on each view and sex (#28)
- Train on RSNA: a held-out test set and batches split by patient, label and view; each run logs its datasets with a content hash (#27)
- Ingest RSNA: header and pixel checks, de-identification, PNGs with a manifest (#26)
- Notebook exploring the RSNA DICOM data: headers, labels, shortcuts, pixels and preprocessing (#25)
- Read DICOM pixel data into 8-bit grayscale images
- Model registry: each trained model is registered, and promoted to production only if it passes the gate (#14)
- Training pipeline DAG: validate a batch, train, evaluate on the test set (#13)
- Train ResNet18 on one batch and track the run in MLflow (#12)
- Split the training data into patient-level batches (#11)
- Local stack with Docker Compose: Airflow, MLflow and Postgres (#10)

### Fixed
- The `evaluate` task failed: its `run_id` argument clashed with Airflow's own (#14)
- Patients that also appear in the test set are left out of the training batches (#13)

## [0.1.0] - 2026-09-28

### Added
- Project setup: uv, ruff, pre-commit hooks and CI
