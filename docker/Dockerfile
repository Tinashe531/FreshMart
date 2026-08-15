# FreshMart pipeline — processing/validation/test environment.
#
# Scope (deliberate): this container reproduces the environment needed to
# RUN and TEST the pipeline's processing, validation, and governance code
# (Python, Pandas, Great Expectations, Pytest, pipeline + governance
# scripts). It does not containerize the full Airflow stack (scheduler,
# webserver, metadata database) — that is a separate, heavier concern.
# Airflow orchestrates; this container is what Airflow's tasks would run
# inside of.

FROM python:3.11-slim

WORKDIR /app

# System deps needed by pyarrow (parquet) — kept minimal on purpose
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY data/raw/ ./data/raw/

# Default command runs the full pipeline + governance chain and the test
# suite, so "docker run" against this image reproduces exactly what was
# verified during development.
CMD ["sh", "-c", "python -m pytest tests/ -v && python src/data_prep/run_pipeline.py && python src/validation/run_expectations.py && python src/governance/run_governance.py"]
