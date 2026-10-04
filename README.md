# mlops

Learning MLOps by building a small pipeline: pneumonia detection on chest X-rays, with Airflow
and MLflow running locally.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Docker, with at least 4 GB of RAM

## Getting started

1. Install the tools:
   ```bash
   uv sync
   uv run pre-commit install
   ```
2. Get the dataset (see [data/README.md](data/README.md)), then split it into batches:
   ```bash
   uv run python scripts/make_batches.py
   ```
3. Start the stack:
   ```bash
   cp .env.example .env    # fill in the "change-me" values
   docker compose up -d --build
   ```
4. Open Airflow at http://localhost:8080 and MLflow at http://localhost:5001.
5. Train a model: in Airflow, trigger `training_pipeline` and pick a batch (1–5). It validates
   the batch, trains on it, scores the model on `test/`, and logs the run to MLflow.
   On CPU this takes about 20 minutes.

   To run the same steps outside Airflow (faster on a Mac GPU):
   ```bash
   uv run python scripts/train.py
   ```
   Settings are in `config/training.yaml`.

Stop everything with `docker compose down`.

Run the tests with `uv run pytest`. The DAG tests need Airflow and run on their own:
```bash
uv run --isolated --no-project --with apache-airflow==3.3.2 --with pytest==9.1.1 pytest tests/dags
```

## Project structure

```
config/    settings for training and batches
dags/      Airflow DAGs
data/      the dataset and batches (not in git)
docker/    Dockerfiles and database setup
scripts/   commands to run by hand
src/mlops/ the project code
tests/     tests, run on a tiny fake dataset
```
