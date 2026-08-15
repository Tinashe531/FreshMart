"""
Profile the raw FreshMart data files.

Purpose: establish reproducible, programmatic facts about the raw data
BEFORE any cleaning/transformation code is written, so that pipeline
logic is based on evidence rather than assumption (per the project's
check-and-treat principle).

Run: python src/data_prep/profile_raw_data.py
Output: prints a profile report to stdout and writes it to
        data/processed/_raw_data_profile.txt
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "_raw_data_profile.txt"

lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


def profile_annex1_item_master():
    log("=" * 70)
    log("annex1.csv — Item Master")
    log("=" * 70)
    df = pd.read_csv(RAW_DIR / "annex1.csv")
    log(f"columns: {list(df.columns)}")
    log(f"row count: {len(df)}")
    log(f"dtypes:\n{df.dtypes}")
    log(f"unique Item Code count: {df['Item Code'].nunique()}")
    log(f"duplicate Item Code rows: {df['Item Code'].duplicated().sum()}")
    log(f"missing values per column:\n{df.isnull().sum()}")
    log(f"unique category count: {df['Category Code'].nunique()}")
    log()
    return df


def profile_annex2_transactions():
    log("=" * 70)
    log("annex2.csv — Transaction data")
    log("=" * 70)
    df = pd.read_csv(RAW_DIR / "annex2.csv")
    log(f"columns: {list(df.columns)}")
    log(f"row count: {len(df)}")
    log(f"dtypes:\n{df.dtypes}")
    log(f"missing values per column:\n{df.isnull().sum()}")
    log(f"exact duplicate rows (all columns): {df.duplicated().sum()}")
    log(f"unique Item Code count: {df['Item Code'].nunique()}")
    log(f"date range: {df['Date'].min()} to {df['Date'].max()}")
    log(f"'Sale or Return' value counts:\n{df['Sale or Return'].value_counts(dropna=False)}")
    log(f"'Discount (Yes/No)' value counts:\n{df['Discount (Yes/No)'].value_counts(dropna=False)}")
    qty = df["Quantity Sold (kilo)"]
    log(f"Quantity Sold (kilo) — min: {qty.min()}, max: {qty.max()}, "
        f"negative values: {(qty < 0).sum()}, zero values: {(qty == 0).sum()}")
    price = df["Unit Selling Price (RMB/kg)"]
    log(f"Unit Selling Price — min: {price.min()}, max: {price.max()}, "
        f"negative values: {(price < 0).sum()}, zero values: {(price == 0).sum()}")
    # Check whether quantity is already signed by Sale/Return, or always positive
    sale_qty_neg = df.loc[df["Sale or Return"] == "sale", "Quantity Sold (kilo)"]
    return_qty_neg = df.loc[df["Sale or Return"] == "return", "Quantity Sold (kilo)"]
    log(f"'sale' rows with negative quantity: {(sale_qty_neg < 0).sum()} of {len(sale_qty_neg)}")
    log(f"'return' rows with negative quantity: {(return_qty_neg < 0).sum()} of {len(return_qty_neg)} "
        f"(if 0, quantity is NOT pre-signed — must derive sign from Sale or Return flag)")
    # Item codes in transactions not present in Item Master
    item_master = pd.read_csv(RAW_DIR / "annex1.csv")
    unmatched = set(df["Item Code"].unique()) - set(item_master["Item Code"].unique())
    log(f"distinct transaction Item Codes NOT found in Item Master: {len(unmatched)}")
    if unmatched:
        unmatched_txn_count = df["Item Code"].isin(unmatched).sum()
        log(f"transaction rows affected by unmatched Item Code: {unmatched_txn_count}")
    log()
    return df


def profile_annex3_wholesale():
    log("=" * 70)
    log("annex3.csv — Wholesale price history")
    log("=" * 70)
    df = pd.read_csv(RAW_DIR / "annex3.csv")
    log(f"columns: {list(df.columns)}")
    log(f"row count: {len(df)}")
    log(f"missing values per column:\n{df.isnull().sum()}")
    log(f"duplicate (Date, Item Code) rows: {df.duplicated(subset=['Date', 'Item Code']).sum()}")
    log(f"unique Item Code count: {df['Item Code'].nunique()}")
    log(f"date range: {df['Date'].min()} to {df['Date'].max()}")
    price = df["Wholesale Price (RMB/kg)"]
    log(f"Wholesale Price — min: {price.min()}, max: {price.max()}, negative: {(price < 0).sum()}")
    log()
    return df


def profile_annex4_loss_rate():
    log("=" * 70)
    log("annex4.csv — Loss rate")
    log("=" * 70)
    df = pd.read_csv(RAW_DIR / "annex4.csv")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]  # BOM on first header seen in raw file
    log(f"columns: {list(df.columns)}")
    log(f"row count: {len(df)}")
    log(f"missing values per column:\n{df.isnull().sum()}")
    log(f"duplicate Item Code rows: {df['Item Code'].duplicated().sum()}")
    loss = df["Loss Rate (%)"].astype(str).str.strip().astype(float)
    log(f"Loss Rate (%) — min: {loss.min()}, max: {loss.max()}")
    log()
    return df


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile_annex1_item_master()
    profile_annex2_transactions()
    profile_annex3_wholesale()
    profile_annex4_loss_rate()
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nProfile written to {OUT_PATH}")
