FROM apache/airflow:3.3.2-python3.12

# mlops gets its own venv, so its packages (torch, mlflow, ...) never clash with Airflow's.
# Tasks run in it with @task.external_python. src/ and config/ are mounted at runtime.
ENV UV_PROJECT_ENVIRONMENT=/opt/mlops/.venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    TORCH_HOME=/opt/mlops/torch

USER root
RUN mkdir /opt/mlops && chown airflow:0 /opt/mlops
USER airflow
WORKDIR /opt/mlops

COPY --chown=airflow:0 pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache
COPY --chown=airflow:0 src ./src
RUN uv sync --frozen --no-dev --no-cache

# Download the pretrained weights once, instead of on every training run.
RUN .venv/bin/python -c \
    "from torchvision.models import ResNet18_Weights, resnet18; resnet18(weights=ResNet18_Weights.DEFAULT)"

WORKDIR /opt/airflow
