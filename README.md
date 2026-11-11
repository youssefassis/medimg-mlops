# medimg-mlops

A small MLOps pipeline for medical imaging, running locally: it learns to detect pneumonia on
chest X-rays, retrains when new data arrives, and only puts a model in production if it passes
quality checks.

```
DICOM files → checked and de-identified → batches → training → quality gate → production → prediction API
```

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Docker, with at least 4 GB of RAM
- An NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/), for training

## Getting started

1. Install the tools:
   ```bash
   uv sync
   uv run pre-commit install
   ```
2. Create `.env` and replace each "change-me":
   ```bash
   cp .env.example .env
   ```
3. Download the data: see [data/README.md](data/README.md).
4. Prepare the data (about 3 minutes):
   ```bash
   uv run --env-file .env python scripts/ingest.py
   uv run python scripts/make_batches.py
   ```
5. Start everything:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
   ```
6. Train: in Airflow, turn on `release_batch` and `training_pipeline`, then trigger
   `release_batch` with batch 1. Release batches 2 to 5 to simulate new data arriving: each
   release retrains the model.
7. Get a prediction, once a model is in production:
   ```bash
   FILE=data/raw/rsna/stage_2_test_images/0000a175-0e68-4ca4-b1af-167204a7e0bc.dcm
   curl -H "Content-Type: application/json" -d "{\"inputs\": [\"$(base64 -i $FILE)\"]}" \
     localhost:8000/invocations
   ```

Stop everything with `docker compose down`.

## Where to look

| What | Where |
|---|---|
| Pipeline runs | Airflow, http://localhost:8080 |
| Experiments and models | MLflow, http://localhost:5001 |
| Why a model was or wasn't promoted | MLflow → Models → `pneumonia-classifier` |
| Prediction API | http://localhost:8000 |

## Development

- Run the tests: `uv run pytest`. The DAG tests run on their own: see the `dags` job in
  [.github/workflows/ci.yml](.github/workflows/ci.yml).
- Explore the data in a notebook:
  `uv run --group notebooks marimo edit notebooks/explore_rsna.py`

## Project structure

```
config/    settings: the classes to predict, ingestion, batches, training and promotion
dags/      Airflow DAGs
data/      the dataset and batches (not in git)
docker/    Dockerfiles and database setup
notebooks/ marimo notebooks to explore the data
scripts/   commands to run by hand
src/mlops/ the project code
tests/     tests, run on a tiny fake dataset
```
