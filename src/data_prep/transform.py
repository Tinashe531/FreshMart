"""
Demand transformation.

Builds the analytical output: ONE ROW = ONE ITEM ON ONE DATE.

Net daily demand = sum of signed transaction quantities per item per date
(quantity is already correctly signed in the source: sale >= 0, return < 0,
confirmed in clean.check_quantity_sign_consistency). This is a direct sum,
not a subtraction of return volume from sale volume.

Negative item-day net demand (returns exceeding sales for that item on
that day) is FLAGGED, not deleted. It is a legitimate business event
worth investigating, not a data error to silently discard.
"""

import pandas as pd


def build_item_day_demand(cleaned_transactions: pd.DataFrame) -> dict:
    """Aggregate cleaned, item-master-joined transactions into the
    item-day analytical dataset. Returns the frame and a report."""
    df = cleaned_transactions.copy()

    grouped = (
        df.groupby(["Date", "Item Code"], as_index=False)
        .agg(
            daily_demand_kg=("Quantity Sold (kilo)", "sum"),
            avg_selling_price=("Unit Selling Price (RMB/kg)", "mean"),
            transaction_count=("Quantity Sold (kilo)", "count"),
            category_code=("Category Code", "first"),
            category_name=("Category Name", "first"),
        )
    )
    grouped = grouped.rename(columns={"Date": "date", "Item Code": "item_code"})

    negative_mask = grouped["daily_demand_kg"] < 0
    grouped["negative_net_demand_flag"] = negative_mask

    n_negative = int(negative_mask.sum())

    # Enforce the output grain: exactly one row per (date, item_code)
    n_rows = len(grouped)
    n_unique_keys = grouped[["date", "item_code"]].drop_duplicates().shape[0]
    grain_ok = n_rows == n_unique_keys

    report = {
        "stage": "transformation",
        "input_transaction_rows": len(df),
        "output_item_day_rows": n_rows,
        "grain_check_one_row_per_item_date": grain_ok,
        "negative_net_demand_item_days": n_negative,
        "negative_net_demand_pct": round(100 * n_negative / n_rows, 3) if n_rows else 0.0,
        "date_range": [str(grouped["date"].min()), str(grouped["date"].max())],
        "distinct_items": int(grouped["item_code"].nunique()),
    }

    if not grain_ok:
        raise ValueError(
            f"Output grain violated: {n_rows} rows but {n_unique_keys} unique "
            f"(date, item_code) keys. There must be exactly one row per item-day."
        )

    # Column order for the final artifact
    grouped = grouped[
        ["date", "item_code", "category_code", "category_name",
         "daily_demand_kg", "avg_selling_price", "transaction_count",
         "negative_net_demand_flag"]
    ]

    return {"data": grouped, "report": report}
