"""
FreshMart data pipeline — Airflow DAG.

Airflow orchestrates the sequence and dependencies between tasks.
Python/Pandas performs the actual processing (src/data_prep/,
src/governance/) — the DAG calls that code, it does not reimplement it.

Sequence (per the agreed small-DAG design):

    ingest -> clean -> transform -> validate (Great Expectations)
        -> PASS: save parquet -> bias check + audit logging
        -> FAIL: stop + log

Schedule: designed to run ahead of each of FreshMart's three weekly
replenishment cycles (Mon/Wed/Sat), per the Module 1/2 architecture.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowFailException


def task_ingest(**context):
    from src.data_prep.ingest import ingest_all
    from src.governance.audit_log import log_event, clear_log

    clear_log()
    result = ingest_all()
    log_event(
        "RUN_STARTED",
        source="annex1-4.csv",
        transaction_records=result["counts"]["transactions"],
        item_master_records=result["counts"]["item_master"],
    )
    # Airflow XCom cannot hold a full 878K-row dataframe efficiently;
    # downstream tasks re-read from data/raw/ directly. This task's
    # purpose is the structural check + the RUN_STARTED audit event.
    return result["counts"]


def task_clean(**context):
    from src.data_prep.ingest import ingest_all
    from src.data_prep.clean import clean_transactions
    from src.governance.audit_log import log_event

    raw = ingest_all()
    cleaned = clean_transactions(raw["transactions"], raw["item_master"])
    cleaned["data"].to_parquet("data/processed/_cleaned_transactions.parquet", index=False)

    checks = cleaned["report"]["checks"]
    log_event(
        "CLEANING_COMPLETED",
        records=len(cleaned["data"]),
        duplicates_removed=checks[1]["duplicates_found"],
        unmatched_item_master_records=checks[3]["unmatched_records"],
    )


def task_transform(**context):
    import pandas as pd
    from src.data_prep.transform import build_item_day_demand
    from src.governance.audit_log import log_event

    cleaned = pd.read_parquet("data/processed/_cleaned_transactions.parquet")
    transformed = build_item_day_demand(cleaned)
    transformed["data"].to_parquet("data/processed/_candidate_item_day.parquet", index=False)

    log_event(
        "TRANSFORMATION_COMPLETED",
        item_day_records=len(transformed["data"]),
        negative_net_demand_flagged=transformed["report"]["negative_net_demand_item_days"],
    )


def task_validate_and_save(**context):
    """Quality gate: run Great Expectations against the candidate output.
    PASS -> promote to the final parquet path. FAIL -> raise, which stops
    the DAG and leaves the previous validated dataset untouched."""
    import shutil
    from src.validation.run_expectations import run_validation
    from src.governance.audit_log import log_event

    # run_expectations reads from the final path by default; point it at
    # the candidate output for this gate check.
    import src.validation.run_expectations as ge_module
    ge_module.DATA_PATH = "data/processed/_candidate_item_day.parquet"

    summary = run_validation()

    if not summary["success"]:
        log_event("VALIDATION_COMPLETED", status="FAIL",
                   evaluated=summary["statistics"].get("evaluated_expectations"),
                   successful=summary["statistics"].get("successful_expectations"))
        raise AirflowFailException(
            "Great Expectations validation failed — pipeline stopped before "
            "promoting output. Previous validated dataset left untouched."
        )

    log_event("VALIDATION_COMPLETED", status="PASS",
               evaluated=summary["statistics"].get("evaluated_expectations"),
               successful=summary["statistics"].get("successful_expectations"),
               monitored_negative_net_demand=summary["monitored_negative_net_demand_item_days"])

    shutil.copy(
        "data/processed/_candidate_item_day.parquet",
        "data/processed/freshmart_item_day.parquet",
    )
    log_event("OUTPUT_CREATED", path="data/processed/freshmart_item_day.parquet")


def task_governance(**context):
    from src.governance.run_governance import run
    run()  # logs BIAS_CHECK_COMPLETED and ANONYMIZATION_SCAN_COMPLETED internally


default_args = {
    "owner": "freshmart-analytics",
    "retries": 1,
}

with DAG(
    dag_id="freshmart_data_pipeline",
    description="FreshMart demand data pipeline: ingest -> clean -> transform -> validate -> save -> govern",
    schedule="0 18 * * SUN,TUE,FRI",  # 12+ hrs ahead of Mon/Wed/Sat replenishment cycles
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["freshmart", "module3", "data-pipeline"],
) as dag:

    ingest = PythonOperator(task_id="ingest", python_callable=task_ingest)
    clean = PythonOperator(task_id="clean", python_callable=task_clean)
    transform = PythonOperator(task_id="transform", python_callable=task_transform)
    validate_and_save = PythonOperator(task_id="validate_and_save", python_callable=task_validate_and_save)
    governance = PythonOperator(task_id="governance", python_callable=task_governance)

    ingest >> clean >> transform >> validate_and_save >> governance
