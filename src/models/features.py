"""
Feature engineering for demand forecasting (Module 4, Task 2).

Leakage rule enforced throughout: every feature for item-day (item, t) may
only use information available strictly BEFORE the forecast is generated
for date t. Historical demand and price features are built with .shift(1)
(or later) relative to the row they describe. Nothing here reads today's
or any future daily_demand_kg into a feature column.

Grain: one row = one item x one date (matches Module 3 output).
"""

import pandas as pd
import numpy as np

# Replenishment cycle: Monday=0 ... Sunday=6
REPLENISHMENT_WEEKDAYS = {0, 2, 5}  # Mon, Wed, Sat


def build_features(item_day: pd.DataFrame) -> dict:
    """Build the leakage-safe feature set from the Module 3 item-day table.

    Returns {"data": features_df, "report": dict} following the repo's
    check-and-report convention.
    """
    df = item_day.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["item_code", "date"]).reset_index(drop=True)

    n_before = len(df)

    # ---- Calendar features (safe: known in advance) ----
    df["day_of_week"] = df["date"].dt.weekday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_replenishment_day"] = df["day_of_week"].isin(REPLENISHMENT_WEEKDAYS).astype(int)
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # ---- Historical demand features (leakage-safe: shifted per item) ----
    g = df.groupby("item_code", group_keys=False)["daily_demand_kg"]

    df["lag_1"] = g.shift(1)
    df["lag_7"] = g.shift(7)
    df["lag_14"] = g.shift(14)

    # Rolling stats computed on PAST values only: shift(1) before rolling
    shifted = g.shift(1)
    df["roll_mean_7"] = shifted.groupby(df["item_code"]).transform(
        lambda s: s.rolling(window=7, min_periods=3).mean()
    )
    df["roll_mean_14"] = shifted.groupby(df["item_code"]).transform(
        lambda s: s.rolling(window=14, min_periods=5).mean()
    )
    df["roll_std_7"] = shifted.groupby(df["item_code"]).transform(
        lambda s: s.rolling(window=7, min_periods=3).std()
    )

    # ---- Price features (leakage-safe: shifted per item) ----
    gp = df.groupby("item_code", group_keys=False)["avg_selling_price"]
    df["price_lag_1"] = gp.shift(1)
    df["price_roll_mean_7"] = gp.shift(1).groupby(df["item_code"]).transform(
        lambda s: s.rolling(window=7, min_periods=3).mean()
    )
    # Price change vs 7 days ago (both values from the past relative to t)
    df["price_change_pct_7"] = (
        (gp.shift(1) - gp.shift(8)) / gp.shift(8).replace(0, np.nan)
    )

    # ---- Item / category features ----
    # Volume tier: item's historical (pre-row) average daily demand, binned.
    # Computed with an expanding mean shifted by 1 so it never sees today.
    df["item_expanding_mean_demand"] = (
        g.apply(lambda s: s.shift(1).expanding(min_periods=5).mean())
        .reset_index(level=0, drop=True)
    )
    df["category_code"] = df["category_code"].astype("category")

    # ---- Drop rows where core lag features are unavailable ----
    # (Not enough history yet for that item -- cannot be used for
    # training/eval without relying on imputed/leaked values.)
    required = ["lag_1", "lag_7", "lag_14", "roll_mean_7", "roll_mean_14",
                "item_expanding_mean_demand"]
    usable_mask = df[required].notna().all(axis=1)
    n_dropped_insufficient_history = int((~usable_mask).sum())
    df_out = df.loc[usable_mask].reset_index(drop=True)

    report = {
        "stage": "feature_engineering",
        "input_rows": n_before,
        "output_rows": len(df_out),
        "dropped_insufficient_history": n_dropped_insufficient_history,
        "date_range_output": [str(df_out["date"].min()), str(df_out["date"].max())],
        "leakage_check": (
            "All historical demand/price features use .shift(1) or later "
            "relative to the target row; calendar features are known in "
            "advance; no future daily_demand_kg used in any feature."
        ),
    }

    return {"data": df_out, "report": report}


FEATURE_COLUMNS = [
    "day_of_week", "is_weekend", "is_replenishment_day", "month",
    "day_of_year", "week_of_year",
    "lag_1", "lag_7", "lag_14",
    "roll_mean_7", "roll_mean_14", "roll_std_7",
    "price_lag_1", "price_roll_mean_7", "price_change_pct_7",
    "item_expanding_mean_demand", "category_code",
]

TARGET_COLUMN = "daily_demand_kg"