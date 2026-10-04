# Changelog

Notable changes, one entry per step. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Local stack with Docker Compose: Airflow 3, MLflow, and one Postgres shared by both (#10)
- `hello` DAG that counts the dataset images, to check the stack works
- `data/` folder with a README on how to get the dataset; its contents stay out of git

## [0.1.0] - 2026-09-28

### Added
- Python project managed with uv, linted and formatted with ruff
- pre-commit hooks: style, stale `uv.lock`, secrets, commit message format, no commits to main
- CI running the pre-commit hooks on every PR
