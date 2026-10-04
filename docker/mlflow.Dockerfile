FROM ghcr.io/mlflow/mlflow:v3.16.1

# The base image has no Postgres driver.
RUN pip install --no-cache-dir psycopg2-binary==2.9.*
