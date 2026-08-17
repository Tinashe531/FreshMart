"""
Seasonal-naive baseline (Module 4, Task 3).

Forecast for item i on date t = actual demand for item i on date (t - 7).
Weekly seasonality matches FreshMart's Mon/Wed/Sat replenishment cycle.
This uses the SAME lag_7 feature already computed in features.py, so the
baseline and the ML models are evaluated on an identical row set --
required for a fair "does ML beat the baseline" comparison.
"""

import numpy as np
import pandas as pd


def seasonal_naive_predict(df: pd.DataFrame) -> pd.Series:
    """Return the seasonal-naive forecast (= lag_7) for each row.

    Assumes `lag_7` has already been computed by features.build_features.
    """
    if "lag_7" not in df.columns:
        raise ValueError("seasonal_naive_predict requires a 'lag_7' column; "
                          "run features.build_features first.")
    return df["lag_7"]


def evaluate_baseline(df: pd.DataFrame, target_col: str = "daily_demand_kg") -> dict:
    """Compute RMSE/MAE/MAPE for the seasonal-naive baseline on `df`."""
    y_true = df[target_col].to_numpy()
    y_pred = seasonal_naive_predict(df).to_numpy()

    errors = y_true - y_pred
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else float("nan")
    bias = float(np.mean(y_pred - y_true))  # positive = over-forecast on average

    return {
        "model": "seasonal_naive",
        "n_obs": int(len(df)),
        "rmse": rmse,
        "mae": mae,
        "mape_pct": mape,
        "forecast_bias": bias,
    }