FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv

# The served model runs code from mlops, so the image has the project and nothing else:
# no Airflow, dev or notebook tools.
ENV UV_PROJECT_ENVIRONMENT=/opt/mlops/.venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH=/opt/mlops/.venv/bin:$PATH

RUN useradd --create-home serving && mkdir /opt/mlops && chown serving /opt/mlops
USER serving
WORKDIR /opt/mlops

COPY --chown=serving pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache
COPY --chown=serving src ./src
RUN uv sync --frozen --no-dev --no-cache

# MODEL_NAME is the registered model, model_name in config/promotion.yaml.
EXPOSE 8000
CMD mlflow models serve --model-uri "models:/${MODEL_NAME}@production" \
    --env-manager local --host 0.0.0.0 --port 8000
