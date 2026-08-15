"""
Cleaning & preparation.

Implements the project's check-and-treat principle: each function checks
for a specific issue and only acts if the issue is actually present.
Nothing here assumes a data-quality problem exists; the profiling script
(profile_raw_data.py) established what's actually in the data.

Also implements the agreed Item Master join rule: unmatched transaction
item codes are KEPT (never dropped) and assigned category_code/category_name
= "UNKNOWN", with the unmatched count logged.
"""

import pandas as pd


def check_and_report_missing(df: pd.DataFrame, columns: list, label: str) -> dict:
    """Check specified columns for missing values. Returns a report dict;
    does not modify the dataframe. Caller decides treatment, if any."""
    report = {}
    for col in columns:
        n_missing = df[col].isnull().sum()
        report[col] = int(n_missing)
    return {"label": label, "missing_by_column": report}


def check_and_drop_exact_duplicates(df: pd.DataFrame, label: str) -> tuple:
    """Check for exact duplicate rows (all columns). Only drops if
    confirmed duplicates exist."""
    n_before = len(df)
    dup_mask = df.duplicated(keep="first")
    n_dupes = int(dup_mask.sum())
    if n_dupes > 0:
        df = df.loc[~dup_mask].copy()
    n_after = len(df)
    return df, {
        "label": label,
        "duplicates_found": n_dupes,
        "records_before": n_before,
        "records_after": n_after,
    }


def standardize_transaction_types(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize dtypes for the transaction table. Does not change values,
    only representations (e.g. parsed dates)."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df["Item Code"] = df["Item Code"].astype(str)
    df["Quantity Sold (kilo)"] = df["Quantity Sold (kilo)"].astype(float)
    df["Unit Selling Price (RMB/kg)"] = df["Unit Selling Price (RMB/kg)"].astype(float)
    df["Sale or Return"] = df["Sale or Return"].str.strip().str.lower()
    return df


def check_quantity_sign_consistency(df: pd.DataFrame) -> dict:
    """Verify the assumption the pipeline depends on: 'sale' rows are
    non-negative and 'return' rows are negative. This is a check, not a
    fix — if the assumption doesn't hold, downstream netting logic (which
    simply sums signed quantity) would be silently wrong."""
    sale_qty = df.loc[df["Sale or Return"] == "sale", "Quantity Sold (kilo)"]
    return_qty = df.loc[df["Sale or Return"] == "return", "Quantity Sold (kilo)"]
    sale_negative = int((sale_qty < 0).sum())
    return_not_negative = int((return_qty >= 0).sum())
    consistent = (sale_negative == 0) and (return_not_negative == 0)
    return {
        "assumption": "sale rows >= 0 and return rows < 0",
        "sale_rows_negative": sale_negative,
        "return_rows_not_negative": return_not_negative,
        "consistent": consistent,
    }


def join_item_master(transactions: pd.DataFrame, item_master: pd.DataFrame) -> tuple:
    """Left join transactions to the Item Master on Item Code. Unmatched
    transactions are KEPT (never dropped via inner join) and assigned
    category_code/category_name = 'UNKNOWN'. Returns the joined frame and
    a report with the unmatched count for logging."""
    item_master = item_master.copy()
    item_master["Item Code"] = item_master["Item Code"].astype(str)

    n_before = len(transactions)
    merged = transactions.merge(
        item_master[["Item Code", "Item Name", "Category Code", "Category Name"]],
        on="Item Code",
        how="left",
        suffixes=("", "_master"),
    )
    n_after = len(merged)

    unmatched_mask = merged["Category Code"].isnull()
    n_unmatched_records = int(unmatched_mask.sum())
    n_unmatched_items = int(merged.loc[unmatched_mask, "Item Code"].nunique())

    # Cast to object dtype first: Category Code is numeric in the source,
    # but must be able to hold the string "UNKNOWN" for unmatched rows
    # even when (as in the current dataset) that path isn't triggered.
    merged["Category Code"] = merged["Category Code"].astype(object)
    merged["Category Name"] = merged["Category Name"].astype(object)
    merged["Item Name"] = merged["Item Name"].astype(object)

    merged.loc[unmatched_mask, "Category Code"] = "UNKNOWN"
    merged.loc[unmatched_mask, "Category Name"] = "Unknown"
    merged.loc[unmatched_mask, "Item Name"] = merged.loc[unmatched_mask, "Item Name"].fillna("Unknown")

    report = {
        "records_before_join": n_before,
        "records_after_join": n_after,
        "record_count_preserved": n_before == n_after,
        "unmatched_records": n_unmatched_records,
        "unmatched_distinct_items": n_unmatched_items,
    }
    return merged, report


def clean_transactions(transactions_raw: pd.DataFrame, item_master_raw: pd.DataFrame) -> dict:
    """Run the full cleaning stage for the transaction table and return
    the cleaned frame plus a structured report of every check performed."""
    report = {"stage": "cleaning", "checks": []}

    missing_report = check_and_report_missing(
        transactions_raw,
        columns=["Date", "Item Code", "Quantity Sold (kilo)",
                 "Unit Selling Price (RMB/kg)", "Sale or Return"],
        label="transactions_missing_values",
    )
    report["checks"].append(missing_report)

    deduped, dup_report = check_and_drop_exact_duplicates(transactions_raw, "transactions_duplicates")
    report["checks"].append(dup_report)

    typed = standardize_transaction_types(deduped)

    sign_report = check_quantity_sign_consistency(typed)
    report["checks"].append(sign_report)
    if not sign_report["consistent"]:
        raise ValueError(
            "Quantity sign assumption violated — sale/return quantities are not "
            "consistently signed. Netting logic must be revisited before proceeding."
        )

    joined, join_report = join_item_master(typed, item_master_raw)
    report["checks"].append({"label": "item_master_join", **join_report})

    return {"data": joined, "report": report}
