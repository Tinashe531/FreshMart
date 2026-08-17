"""
Model development & MLflow experiment tracking (Module 4, Task 4).

Trains Random Forest and XGBoost regressors to predict daily_demand_kg.
Each candidate hyperparameter configuration is evaluated with the
rolling-origin CV folds from split.py (time-series CV, never random CV).
Every run -- including every fold -- is logged to MLflow: params, metrics,
and the fitted model artifact. The best configuration per model family is
registered as a "best of family" run; the two family-winners are then
compared against each other and against the seasonal-naive baseline in
Task 5.

Run this file directly to execute the full experiment sweep:
    python -m src.models.train
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from src.models.features import build_features, FEATURE_COLUMNS, TARGET_COLUMN
from src.models.split import chronological_split, rolling_origin_folds
from src.models.baseline import evaluate_baseline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
EXPERIMENT_NAME = "freshmart_demand_forecasting"

RF_GRID = [
    {"n_estimators": 150, "max_depth": 8, "min_samples_leaf": 5, "n_jobs": -1},
    {"n_estimators": 150, "max_depth": 12, "min_samples_leaf": 3, "n_jobs": -1},
]

XGB_GRID = [
    {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8,
     "n_jobs": -1, "tree_method": "hist"},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
     "n_jobs": -1, "tree_method": "hist"},
]


def _prep_xy(df: pd.DataFrame):
    """Encode category_code numerically (tree models need numeric input);
    everything else in FEATURE_COLUMNS is already numeric."""
    X = df[FEATURE_COLUMNS].copy()
    X["category_code"] = X["category_code"].cat.codes
    y = df[TARGET_COLUMN].to_numpy()
    return X, y


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _cv_score(model_ctor, params: dict, train_df: pd.DataFrame, folds) -> dict:
    """Fit+evaluate `params` across all rolling-origin folds. Returns mean
    RMSE/MAE across folds (the model-selection criterion)."""
    fold_rmses, fold_maes = [], []
    for train_idx, val_idx in folds:
        fold_train = train_df.loc[train_idx]
        fold_val = train_df.loc[val_idx]
        X_tr, y_tr = _prep_xy(fold_train)
        X_va, y_va = _prep_xy(fold_val)

        model = model_ctor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)

        fold_rmses.append(_rmse(y_va, preds))
        fold_maes.append(mean_absolute_error(y_va, preds))

    return {
        "cv_rmse_mean": float(np.mean(fold_rmses)),
        "cv_rmse_std": float(np.std(fold_rmses)),
        "cv_mae_mean": float(np.mean(fold_maes)),
        "n_folds": len(fold_rmses),
    }


def tune_model(model_name: str, model_ctor, grid: list, train_df: pd.DataFrame, folds) -> dict:
    """Grid search over `grid` using rolling-origin CV, logging every
    configuration to MLflow as its own run. Returns the best config."""
    best = None
    for params in grid:
        with mlflow.start_run(run_name=f"{model_name}_cv_{hash(frozenset(params.items())) % 10000}"):
            mlflow.log_param("model_family", model_name)
            for k, v in params.items():
                mlflow.log_param(k, v)

            cv_result = _cv_score(model_ctor, params, train_df, folds)
            mlflow.log_metrics({
                "cv_rmse_mean": cv_result["cv_rmse_mean"],
                "cv_rmse_std": cv_result["cv_rmse_std"],
                "cv_mae_mean": cv_result["cv_mae_mean"],
            })
            mlflow.set_tag("stage", "hyperparameter_tuning")

            if best is None or cv_result["cv_rmse_mean"] < best["cv_result"]["cv_rmse_mean"]:
                best = {"params": params, "cv_result": cv_result}

    return best


def train_final_and_log(model_name: str, model_ctor, best_params: dict,
                         train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Refit the winning config on the FULL training period, evaluate once
    on the untouched hold-out test set, log as the 'selected' MLflow run,
    and return the fitted model + test metrics."""
    X_tr, y_tr = _prep_xy(train_df)
    X_te, y_te = _prep_xy(test_df)

    with mlflow.start_run(run_name=f"{model_name}_SELECTED"):
        mlflow.log_param("model_family", model_name)
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.set_tag("stage", "final_selected_model")

        model = model_ctor(**best_params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        errors = y_te - preds
        rmse = _rmse(y_te, preds)
        mae = mean_absolute_error(y_te, preds)
        nonzero = y_te != 0
        mape = float(np.mean(np.abs(errors[nonzero] / y_te[nonzero])) * 100)
        bias = float(np.mean(preds - y_te))
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        test_metrics = {
            "test_rmse": rmse, "test_mae": mae, "test_mape_pct": mape,
            "test_r2": r2, "test_forecast_bias": bias,
        }
        mlflow.log_metrics(test_metrics)
        if model_name == "xgboost":
            mlflow.xgboost.log_model(model, name="model")
        else:
            mlflow.sklearn.log_model(model, name="model")

    return {"model": model, "test_metrics": test_metrics,
            "predictions": preds, "y_true": y_te, "test_df": test_df.reset_index(drop=True)}


def run_all():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    item_day = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "freshmart_item_day.parquet")
    feat = build_features(item_day)["data"]
    split = chronological_split(feat)
    train_df, test_df = split["train"], split["test"]
    folds = rolling_origin_folds(train_df)

    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows | CV folds: {len(folds)}")

    baseline_metrics = evaluate_baseline(test_df)
    print("Baseline:", baseline_metrics)

    results = {"baseline": baseline_metrics}

    print("\nTuning Random Forest...")
    rf_best = tune_model("random_forest", RandomForestRegressor, RF_GRID, train_df, folds)
    print("RF best params:", rf_best["params"], "CV RMSE:", rf_best["cv_result"]["cv_rmse_mean"])

    print("\nTuning XGBoost...")
    xgb_best = tune_model("xgboost", XGBRegressor, XGB_GRID, train_df, folds)
    print("XGB best params:", xgb_best["params"], "CV RMSE:", xgb_best["cv_result"]["cv_rmse_mean"])

    print("\nRefitting winners on full train, evaluating on hold-out test...")
    rf_final = train_final_and_log("random_forest", RandomForestRegressor, rf_best["params"], train_df, test_df)
    xgb_final = train_final_and_log("xgboost", XGBRegressor, xgb_best["params"], train_df, test_df)

    results["random_forest"] = {"params": rf_best["params"], "cv": rf_best["cv_result"],
                                 "test": rf_final["test_metrics"]}
    results["xgboost"] = {"params": xgb_best["params"], "cv": xgb_best["cv_result"],
                           "test": xgb_final["test_metrics"]}

    print("\n=== SUMMARY (hold-out test set) ===")
    print(f"{'model':<18}{'RMSE':>10}{'MAE':>10}{'MAPE%':>10}{'R2':>8}{'bias':>10}")
    print(f"{'seasonal_naive':<18}{baseline_metrics['rmse']:>10.3f}{baseline_metrics['mae']:>10.3f}"
          f"{baseline_metrics['mape_pct']:>10.3f}{'--':>8}{baseline_metrics['forecast_bias']:>10.3f}")
    for name, r in [("random_forest", results["random_forest"]), ("xgboost", results["xgboost"])]:
        t = r["test"]
        print(f"{name:<18}{t['test_rmse']:>10.3f}{t['test_mae']:>10.3f}{t['test_mape_pct']:>10.3f}"
              f"{t['test_r2']:>8.3f}{t['test_forecast_bias']:>10.3f}")

    # Persist artifacts to disk for downstream SHAP / fairness / FastAPI work
    artifacts_dir = PROJECT_ROOT / "src" / "models" / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    joblib.dump(rf_final["model"], artifacts_dir / "random_forest.joblib")
    joblib.dump(xgb_final["model"], artifacts_dir / "xgboost.joblib")
    train_df.to_parquet(artifacts_dir / "train_df.parquet")
    test_df.to_parquet(artifacts_dir / "test_df.parquet")
    import json
    with open(artifacts_dir / "results_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nArtifacts saved to {artifacts_dir}")

    return results, rf_final, xgb_final, train_df, test_df


if __name__ == "__main__":
    run_all()