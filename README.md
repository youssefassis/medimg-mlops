# medimg-mlops

Learning MLOps for medical imaging by building a small pipeline: pneumonia detection on chest
X-rays, with Airflow and MLflow running locally.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Docker, with at least 4 GB of RAM

## Getting started

1. Install the tools:
   ```bash
   uv sync
   uv run pre-commit install
   ```
2. Create `.env` and fill in the "change-me" values:
   ```bash
   cp .env.example .env
   ```
3. Get the RSNA data (see [data/README.md](data/README.md)), then ingest it and split it into
   a test set and batches:
   ```bash
   uv run --env-file .env python scripts/ingest_rsna.py   # about 2.5 minutes
   uv run python scripts/make_batches.py
   ```
   Ingestion checks and de-identifies a sample of the DICOM files and saves them as PNGs with a
   manifest. The split holds out 20% of patients as the test set and divides the rest into
   5 batches, each with the same share of every label and view.
4. Start the stack:
   ```bash
   docker compose up -d --build
   ```
5. Open Airflow at http://localhost:8080 and MLflow at http://localhost:5001. The prediction API
   is at http://localhost:8000.
6. Train a model: in Airflow, unpause `release_batch` and `training_pipeline`, then trigger
   `release_batch` with batch 1. Each release triggers `training_pipeline`, which trains on
   every batch released so far: release batches 2 to 5 one by one to simulate new data
   arriving. It validates the batches, trains on them, scores the model on the test set, and
   logs the run to MLflow, with the batches and the test set as its datasets.
   Training picks the decision threshold on validation data: the highest one that still finds
   `min_recall` of the opacities. The test set is scored at that threshold, with 95% confidence
   intervals, and per view, sex, age group and pixel spacing (the table is in the run's
   `validation/subgroups.json`). The model is then registered as `pneumonia-classifier`, and
   gets the `production` alias if it passes the gate in `config/promotion.yaml`, overall and
   for each view and sex. The decision is recorded in the run's `report.json` (data used,
   metrics, gate, production model compared with, reasons), and summarized on the model
   version's page in MLflow (Models → `pneumonia-classifier`).
   On CPU this takes several minutes per batch.

   To run the same steps outside Airflow, on the batches up to `batch` in
   `config/training.yaml` (about 2 minutes per batch on a Mac GPU):
   ```bash
   uv run python scripts/train.py
   ```
   Settings are in `config/training.yaml`.
7. Get a prediction from the production model by sending it DICOM files, base64-encoded:
   ```bash
   FILE=data/raw/rsna/stage_2_test_images/0000a175-0e68-4ca4-b1af-167204a7e0bc.dcm
   curl -H "Content-Type: application/json" -d "{\"inputs\": [\"$(base64 -i $FILE)\"]}" \
     localhost:8000/invocations
   ```
   It returns one prediction per file, such as
   `{"pneumonia": false, "score": 0.03, "problems": []}`: the score is the probability of
   pneumonia, compared with the model's threshold. A file that fails the ingestion header checks
   gets no prediction, only its `problems`. The API is `mlflow models serve`; it starts once a
   model is in production. After promoting a new one, run `docker compose restart serving`.

Stop everything with `docker compose down`.

Run the tests with `uv run pytest`. The DAG tests need Airflow and run on their own, without
the shared fixtures in `tests/conftest.py`:
```bash
uv run --isolated --no-project --with apache-airflow==3.3.2 --with pytest==9.1.1 \
  pytest tests/dags --confcutdir=tests/dags
```

## Exploring the RSNA DICOM data

The notebook `notebooks/explore_rsna.py` looks at the RSNA pneumonia dataset: DICOM headers,
labels, shortcuts a model could learn, pixels and preprocessing options. Get the data first
(see [data/README.md](data/README.md)), then open it with [marimo](https://marimo.io):
```bash
uv run --group notebooks marimo edit notebooks/explore_rsna.py
```

## Project structure

```
config/    settings for ingestion, batches, training and promotion
dags/      Airflow DAGs
data/      the dataset and batches (not in git)
docker/    Dockerfiles and database setup
notebooks/ marimo notebooks to explore the data
scripts/   commands to run by hand
src/mlops/ the project code
tests/     tests, run on a tiny fake dataset
```
