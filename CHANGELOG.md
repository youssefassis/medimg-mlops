# Changelog

Notable changes, one entry per step. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Training pipeline DAG: validate a batch, train, evaluate on the test set (#13)
- Train ResNet18 on one batch and track the run in MLflow (#12)
- Split the training data into patient-level batches (#11)
- Local stack with Docker Compose: Airflow, MLflow and Postgres (#10)

### Fixed
- Patients that also appear in the test set are left out of the training batches (#13)

## [0.1.0] - 2026-09-28

### Added
- Project setup: uv, ruff, pre-commit hooks and CI
