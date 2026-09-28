# mlops

Learning project for Airflow, MLflow and continual learning in medical imaging.

**Use case:** pneumonia detection on chest X-rays (RSNA dataset), retrained as new batches arrive
and deployed only if it meets the deployment criteria.

## Layout

```
config/   YAML settings (no secrets)          models/   local checkpoints (gitignored)
dags/     Airflow DAGs, orchestration only    scripts/  command-line helpers
data/     local datasets (gitignored)         src/      the mlops package, one sub-package per stage
docker/   container build files               tests/    mirrors src/
```

## Setup

```bash
uv sync                 # Python 3.12 environment
uv run pre-commit install
uv run pytest
```

## Local stack

Needs Docker with 6 GB of RAM (on macOS: `colima start --cpu 4 --memory 8`).

```bash
cp .env.example .env    # replace every "change-me"
docker compose up -d --build
```

| Service | URL |
|---|---|
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5001 |
| RustFS console | http://localhost:9001/rustfs/console/ |

To run the pipeline: upload files to `landing/batch_001/` in RustFS, then trigger the
`continual_learning` DAG in Airflow.

## Roadmap

0. Project skeleton, tooling, CI
1. Local stack (Airflow, MLflow, RustFS) and a placeholder pipeline
2. Data validation and DICOM standardization
3. Baseline model
4. Model registry and deployment
5. Continual learning: fine-tuning and measuring forgetting
6. Continual learning: reducing forgetting
7. Monitoring and drift-triggered retraining
