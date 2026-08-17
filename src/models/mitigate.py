"""
Fairness Mitigation (Module 4, Task 6 continued).

Task 6's fairness check (explain.py) found a 40.0 percentage-point MAPE
gap between volume tiers (low_volume MAPE ~106% vs mid/high ~66%),
exceeding the 25pp mitigation threshold. This module investigates the
cause and applies a mitigation, then re-evaluates.

Root cause investigated: MAPE is scale-sensitive. Low-volume items have
small absolute daily_demand_kg values, so even a small absolute error
(e.g. 0.5kg on a 1kg-a-day item) produces a huge percentage error. The
XGBoost training objective (squared error) also implicitly prioritizes
high-magnitude items, since they contribute more to the loss -- so the
model has little incentive to fit low-volume items well.

Mitigation applied: inverse-magnitude sample weighting. Rows belonging
to low-volume items are up-weighted during training so their squared
error contributes proportionally more to the loss, pushing the model to
fit them better -- without discarding the RMSE-optimal fit on
high-volume items, which matters most for the MAPE<=20% top-10 target.

Run: python -m src.models.mitigate
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor

from src.models.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.evaluate import _prep_X, _metrics, category_breakdown, volume_tier_breakdown, top10_volume_mape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "src" / "models" / "artifacts"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
EXPERIMENT_NAME = "freshmart_demand_forecasting"

BEST_XGB_PARAMS = {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.1,
                    "subsample": 0.8, "n_jobs": -1, "tree_method": "hist"}


def compute_sample_weights(train_df: pd.DataFrame) -> np.ndarray:
    """Weight = inverse of the item's train-period average demand,
    normalized so weights average to 1.0 (keeps the effective loss scale
    comparable to the unweighted fit). Low-volume items get upweighted;
    high-volume items get downweighted -- but not to zero, since the
    top-10 volume target still has to be met."""
    item_avg = train_df.groupby("item_code")[TARGET_COLUMN].transform("mean")
    raw_weight = 1.0 / (item_avg + 1.0)  # +1 avoids exploding weights for tiny values
    weight = raw_weight / raw_weight.mean()
    # Cap to avoid a handful of near-zero-demand items dominating the loss
    weight = weight.clip(upper=weight.quantile(0.99))
    return weight.to_numpy()


def refit_with_mitigation(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = _prep_X(train_df)
    X_test = _prep_X(test_df)
    y_train = train_df[TARGET_COLUMN].to_numpy()
    y_test = test_df[TARGET_COLUMN].to_numpy()

    weights = compute_sample_weights(train_df)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost_MITIGATED_volume_reweighted"):
        for k, v in BEST_XGB_PARAMS.items():
            mlflow.log_param(k, v)
        mlflow.log_param("mitigation", "inverse_volume_sample_weighting")
        mlflow.set_tag("stage", "fairness_mitigation")

        model = XGBRegressor(**BEST_XGB_PARAMS)
        model.fit(X_train, y_train, sample_weight=weights)
        preds = model.predict(X_test)

        overall = _metrics(y_test, preds)
        mlflow.log_metrics({
            "test_rmse": overall["rmse"], "test_mae": overall["mae"],
            "test_mape_pct": overall["mape_pct"], "test_r2": overall["r2"],
            "test_bias": overall["bias"],
        })
        mlflow.xgboost.log_model(model, name="model")

    return model, preds, overall


def compare_before_after(test_df: pd.DataFrame, train_df: pd.DataFrame,
                          preds_before: np.ndarray, preds_after: np.ndarray):
    vol_before = volume_tier_breakdown(test_df, train_df, preds_before, "xgboost_before")
    vol_after = volume_tier_breakdown(test_df, train_df, preds_after, "xgboost_after_mitigation")

    gap_before = vol_before["mape_pct"].max() - vol_before["mape_pct"].min()
    gap_after = vol_after["mape_pct"].max() - vol_after["mape_pct"].min()

    cat_before = category_breakdown(test_df, preds_before, "xgboost_before")
    cat_after = category_breakdown(test_df, preds_after, "xgboost_after_mitigation")

    top10_before = top10_volume_mape(test_df, train_df, preds_before, "xgboost_before")
    top10_after = top10_volume_mape(test_df, train_df, preds_after, "xgboost_after_mitigation")

    return {
        "volume_tier_before": vol_before, "volume_tier_after": vol_after,
        "gap_before_pp": float(gap_before), "gap_after_pp": float(gap_after),
        "category_before": cat_before, "category_after": cat_after,
        "top10_mape_before": top10_before["mape_pct"], "top10_mape_after": top10_after["mape_pct"],
    }


def run_all():
    train_df = pd.read_parquet(ARTIFACTS_DIR / "train_df.parquet")
    test_df = pd.read_parquet(ARTIFACTS_DIR / "test_df.parquet")
    xgb_before = joblib.load(ARTIFACTS_DIR / "xgboost.joblib")

    X_test = _prep_X(test_df)
    preds_before = xgb_before.predict(X_test)

    print("Refitting XGBoost with inverse-volume sample weighting...")
    model_after, preds_after, overall_after = refit_with_mitigation(train_df, test_df)

    print("\n=== Overall test metrics: before vs after mitigation ===")
    overall_before = _metrics(test_df[TARGET_COLUMN].to_numpy(), preds_before)
    comp = pd.DataFrame([
        {"stage": "before_mitigation", **overall_before},
        {"stage": "after_mitigation", **overall_after},
    ])
    print(comp[["stage", "rmse", "mae", "mape_pct", "r2", "bias"]].to_string(index=False))

    result = compare_before_after(test_df, train_df, preds_before, preds_after)

    print("\n=== Volume-tier MAPE: before ===")
    print(result["volume_tier_before"].to_string(index=False))
    print("\n=== Volume-tier MAPE: after mitigation ===")
    print(result["volume_tier_after"].to_string(index=False))
    print(f"\nVolume-tier MAPE gap: {result['gap_before_pp']:.1f}pp -> {result['gap_after_pp']:.1f}pp")

    print(f"\nTop-10 volume item MAPE: {result['top10_mape_before']:.2f}% -> {result['top10_mape_after']:.2f}%  "
          f"(target: <=20%)")

    decision = (
        "gap reduced but mitigation adopted" if result["gap_after_pp"] < result["gap_before_pp"]
        else "mitigation did not reduce the gap; original model retained"
    )
    print(f"\nDecision: {decision}")

    # Persist whichever model is being carried forward, plus a clear record
    if result["gap_after_pp"] < result["gap_before_pp"]:
        joblib.dump(model_after, ARTIFACTS_DIR / "xgboost_final.joblib")
        final_choice = "mitigated"
    else:
        joblib.dump(xgb_before, ARTIFACTS_DIR / "xgboost_final.joblib")
        final_choice = "original"

    summary = {
        "gap_before_pp": result["gap_before_pp"],
        "gap_after_pp": result["gap_after_pp"],
        "top10_mape_before": result["top10_mape_before"],
        "top10_mape_after": result["top10_mape_after"],
        "overall_before": overall_before,
        "overall_after": overall_after,
        "decision": decision,
        "final_model_choice": final_choice,
    }
    with open(ARTIFACTS_DIR / "explainability" / "mitigation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nFinal model choice: {final_choice} (saved as xgboost_final.joblib)")

    return summary


if __name__ == "__main__":
    run_all()