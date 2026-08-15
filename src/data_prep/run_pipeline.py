"""
Runs the FreshMart pipeline end to end against the real data:
ingest -> clean -> transform -> save.

This is the same code the Airflow DAG calls — the DAG orchestrates these
functions, it does not duplicate their logic. Great Expectations
validation is run as a separate step (src/validation/) against this
script's output, since GE tests data, not code.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_prep.ingest import ingest_all
from src.data_prep.clean import clean_transactions
from src.data_prep.transform import build_item_day_demand
from src.governance.audit_log import log_event, clear_log

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "freshmart_item_day.parquet"


def run(reset_log: bool = False) -> dict:
    if reset_log:
        clear_log()

    raw = ingest_all()
    log_event(
        "RUN_STARTED",
        source="annex1-4.csv",
        transaction_records=raw["counts"]["transactions"],
        item_master_records=raw["counts"]["item_master"],
    )

    cleaned = clean_transactions(raw["transactions"], raw["item_master"])
    log_event(
        "CLEANING_COMPLETED",
        records=len(cleaned["data"]),
        duplicates_removed=cleaned["report"]["checks"][1]["duplicates_found"],
        unmatched_item_master_records=cleaned["report"]["checks"][3]["unmatched_records"],
    )

    transformed = build_item_day_demand(cleaned["data"])
    log_event(
        "TRANSFORMATION_COMPLETED",
        item_day_records=len(transformed["data"]),
        negative_net_demand_flagged=transformed["report"]["negative_net_demand_item_days"],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    transformed["data"].to_parquet(OUTPUT_PATH, index=False)
    log_event(
        "OUTPUT_CREATED",
        path=str(OUTPUT_PATH),
        records=len(transformed["data"]),
    )

    return {
        "cleaned_report": cleaned["report"],
        "transform_report": transformed["report"],
        "output_path": str(OUTPUT_PATH),
        "output_records": len(transformed["data"]),
    }


if __name__ == "__main__":
    import json
    result = run(reset_log=True)
    print(json.dumps(result, indent=2, default=str))
