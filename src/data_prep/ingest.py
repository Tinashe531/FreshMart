"""
Ingestion & initial checks.

Loads the four raw FreshMart data sources and performs structural checks
(expected columns present, non-empty) before any cleaning happens.
Raw files are read but never modified.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_COLUMNS = {
    "annex1.csv": ["Item Code", "Item Name", "Category Code", "Category Name"],
    "annex2.csv": ["Date", "Time", "Item Code", "Quantity Sold (kilo)",
                   "Unit Selling Price (RMB/kg)", "Sale or Return", "Discount (Yes/No)"],
    "annex3.csv": ["Date", "Item Code", "Wholesale Price (RMB/kg)"],
    "annex4.csv": ["Item Code", "Item Name", "Loss Rate (%)"],
}


class IngestionError(Exception):
    """Raised when a raw file fails a structural check."""


def _check_columns(df: pd.DataFrame, filename: str) -> None:
    expected = EXPECTED_COLUMNS[filename]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise IngestionError(f"{filename}: missing expected columns {missing}")


def load_item_master(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "annex1.csv")
    _check_columns(df, "annex1.csv")
    if len(df) == 0:
        raise IngestionError("annex1.csv: file is empty")
    return df


def load_transactions(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "annex2.csv")
    _check_columns(df, "annex2.csv")
    if len(df) == 0:
        raise IngestionError("annex2.csv: file is empty")
    return df


def load_wholesale_prices(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "annex3.csv")
    _check_columns(df, "annex3.csv")
    if len(df) == 0:
        raise IngestionError("annex3.csv: file is empty")
    return df


def load_loss_rates(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "annex4.csv")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]  # strip BOM on first header
    _check_columns(df, "annex4.csv")
    if len(df) == 0:
        raise IngestionError("annex4.csv: file is empty")
    return df


def ingest_all(raw_dir: Path = RAW_DIR) -> dict:
    """Load all four sources and return record counts alongside the frames,
    so the counts can be logged before any processing happens."""
    item_master = load_item_master(raw_dir)
    transactions = load_transactions(raw_dir)
    wholesale = load_wholesale_prices(raw_dir)
    loss_rate = load_loss_rates(raw_dir)
    return {
        "item_master": item_master,
        "transactions": transactions,
        "wholesale": wholesale,
        "loss_rate": loss_rate,
        "counts": {
            "item_master": len(item_master),
            "transactions": len(transactions),
            "wholesale": len(wholesale),
            "loss_rate": len(loss_rate),
        },
    }


if __name__ == "__main__":
    result = ingest_all()
    for name, count in result["counts"].items():
        print(f"{name}: {count} records")
