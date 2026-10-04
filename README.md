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
2. Get the dataset: see [data/README.md](data/README.md).
3. Start the stack:
   ```bash
   cp .env.example .env    # fill in the "change-me" values
   docker compose up -d --build
   ```
4. Open Airflow at http://localhost:8080 and MLflow at http://localhost:5001.
   Trigger the `hello` DAG to check that Airflow sees the dataset.

Stop everything with `docker compose down`.

## Project structure

```
dags/      Airflow DAGs
data/      the dataset (not in git)
docker/    Dockerfiles and database setup
```
