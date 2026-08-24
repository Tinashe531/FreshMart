# FreshMart -- one multi-stage Dockerfile for all three local-demo
# containers (pipeline / api / dashboard), instead of separate Dockerfiles
# per service. docker-compose.yml builds each service from a different
# `target:` stage of this same file.
#
# Build a single stage directly with:
#   docker build --target pipeline  -t freshmart-pipeline  -f docker/Dockerfile .
#   docker build --target api       -t freshmart-api       -f docker/Dockerfile .
#   docker build --target dashboard -t freshmart-dashboard -f docker/Dockerfile .

FROM python:3.12-slim AS base
WORKDIR /app
# System deps needed by pyarrow (parquet) across every stage -- kept in the
# shared base so it's declared once, not repeated per stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# ============================================================================
# STAGE: pipeline -- processing/validation/test environment (Module 3).
# Unchanged from the original single-purpose Dockerfile this replaces.
# ============================================================================
FROM base AS pipeline

COPY docker/requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY data/raw/ ./data/raw/

# Default command runs the full pipeline + governance chain and the test
# suite, so "docker run" against this image reproduces exactly what was
# verified during development.
CMD ["sh", "-c", "python -m pytest tests/ -v && python src/data_prep/run_pipeline.py && python src/validation/run_expectations.py && python src/governance/run_governance.py"]


# ============================================================================
# STAGE: api -- Prediction API (Module 4), serves /predict.
# Short, stable dependency list -- installed directly, no separate
# requirements file needed for 7 packages. NOT the root requirements.txt
# (which also pulls in Airflow/Great Expectations/MLflow -- unnecessary
# for serving predictions, and Airflow's strict pins risk slower/fragile
# builds for a container that doesn't need any of it).
# ============================================================================
FROM base AS api

RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" pydantic \
    pandas==3.0.2 joblib==1.5.3 xgboost==3.4.1 pyarrow

COPY src/models/ ./src/models/

EXPOSE 8000
CMD ["uvicorn", "src.models.api:app", "--host", "0.0.0.0", "--port", "8000"]


# ============================================================================
# STAGE: dashboard -- Stakeholder + Monitoring dashboard (Module 5 / Task 7).
# Reuses dashboards/requirements.txt directly -- the same file Streamlit
# Cloud reads for the actual production deployment, so this container's
# dependencies match the real live app, not a separate guess at it.
# ============================================================================
FROM base AS dashboard

COPY dashboards/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY dashboards/ ./dashboards/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501
CMD ["streamlit", "run", "dashboards/app.py", "--server.address=0.0.0.0", "--server.port=8501"]