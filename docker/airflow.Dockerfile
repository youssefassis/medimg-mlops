FROM apache/airflow:3.3.2-python3.12

# Install the pipeline package in editable mode; docker-compose mounts ./src over it
# so code changes are picked up without rebuilding.
COPY --chown=airflow:root pyproject.toml README.md /opt/mlops/
COPY --chown=airflow:root src /opt/mlops/src
RUN pip install --no-cache-dir "apache-airflow==3.3.2" -e /opt/mlops
