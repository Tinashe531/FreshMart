"""
Runs the governance chain against the real pipeline output:
bias check -> anonymization scan -> audit logging.

This is the second half of the connected chain described in the project
plan: freshmart_item_day.parquet feeds both Bias Detection and (via
anonymize) a PII scan, and every step is recorded in the audit log as
evidence.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_prep.ingest import ingest_all
from src.data_prep.clean import clean_transactions
from src.governance.bias_check import run_bias_check
from src.governance.anonymize import anonymize
from src.governance.audit_log import log_event

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "freshmart_item_day.parquet"


def run() -> dict:
    processed = pd.read_parquet(OUTPUT_PATH)

    # Bias check needs the raw and cleaned transaction data (same grain)
    raw = ingest_all()
    cleaned = clean_transactions(raw["transactions"], raw["item_master"])
    raw_with_category = raw["transactions"].merge(
        raw["item_master"][["Item Code", "Category Name"]],
        on="Item Code", how="left"
    )
    bias_result = run_bias_check(raw_with_category, cleaned["data"])
    log_event(
        "BIAS_CHECK_COMPLETED",
        categories_checked=bias_result["categories_checked"],
        categories_flagged=bias_result["categories_flagged"],
        threshold_pp=bias_result["threshold_pp"],
    )

    # Anonymization scan against the actual output schema
    _, removed_pii = anonymize(processed)
    log_event(
        "ANONYMIZATION_SCAN_COMPLETED",
        columns_scanned=len(processed.columns),
        pii_columns_removed=len(removed_pii),
        pii_columns=removed_pii,
    )

    return {"bias_result": bias_result, "pii_removed": removed_pii}


if __name__ == "__main__":
    import json
    result = run()
    print(json.dumps(result, indent=2, default=str))
