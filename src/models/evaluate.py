"""
Performance Evaluation & Validation (Module 4, Task 5).

Loads the artifacts saved by train.py and produces:
  - Full metric comparison table (baseline vs RF vs XGBoost) on hold-out test
  - Category-level error breakdowns for BOTH ML models
  - Volume-tier error breakdowns for BOTH ML models
  - Top-10 highest-volume-item MAPE for BOTH ML models
  - Validation plots: residuals, actual-vs-predicted, model comparison
  - Leakage / temporal-separation confirmation checks

Important:
  - The final model is NOT assumed to be XGBoost.
  - Model selection is based on evidence from CV + hold-out performance.
  - Category and volume-tier results are reported for both RF and XGBoost
    before any final model is selected.
  - Leakage checks validate actual lag values against historical demand.

Run:
    python -m src.models.evaluate
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.baseline import seasonal_naive_predict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "src" / "models" / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


# ============================================================================
# DATA / MODEL PREPARATION
# ============================================================================

def _prep_X(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare model features using the same category-code convention used
    during training.

    category_code is converted to categorical codes because the tree models
    require numeric input.
    """
    X = df[FEATURE_COLUMNS].copy()
    X["category_code"] = X["category_code"].astype("category").cat.codes
    return X


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate the common regression metrics used throughout Module 4."""
    errors = y_true - y_pred

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    nonzero = y_true != 0
    if nonzero.any():
        mape = float(
            np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100
        )
    else:
        mape = float("nan")

    # Positive bias = over-forecasting on average.
    bias = float(np.mean(y_pred - y_true))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "mape_pct": mape,
        "r2": r2,
        "bias": bias,
        "n": int(len(y_true)),
    }


def load_everything():
    """Load the artifacts produced by train.py."""
    test_df = pd.read_parquet(ARTIFACTS_DIR / "test_df.parquet")
    train_df = pd.read_parquet(ARTIFACTS_DIR / "train_df.parquet")

    rf = joblib.load(ARTIFACTS_DIR / "random_forest.joblib")
    xgb = joblib.load(ARTIFACTS_DIR / "xgboost.joblib")

    return train_df, test_df, rf, xgb


# ============================================================================
# OVERALL MODEL COMPARISON
# ============================================================================

def overall_comparison(
    test_df: pd.DataFrame,
    rf,
    xgb
):
    """
    Compare the seasonal-naive baseline, Random Forest and XGBoost on the
    identical untouched hold-out test set.
    """
    y_true = test_df[TARGET_COLUMN].to_numpy()
    X_test = _prep_X(test_df)

    preds = {
        "seasonal_naive": seasonal_naive_predict(test_df).to_numpy(),
        "random_forest": rf.predict(X_test),
        "xgboost": xgb.predict(X_test),
    }

    rows = []

    for name, y_pred in preds.items():
        m = _metrics(y_true, y_pred)
        m["model"] = name
        rows.append(m)

    comparison = pd.DataFrame(rows)[
        ["model", "n", "rmse", "mae", "mape_pct", "r2", "bias"]
    ]

    return comparison, preds


# ============================================================================
# CATEGORY-LEVEL PERFORMANCE
# ============================================================================

def category_breakdown(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    model_name: str
) -> pd.DataFrame:
    """
    Calculate error metrics for each vegetable category.

    This function is intentionally reusable by explain.py and mitigate.py.
    """
    df = test_df.copy()
    df["_pred"] = y_pred

    rows = []

    for category, group in df.groupby(
        "category_name",
        observed=True
    ):
        m = _metrics(
            group[TARGET_COLUMN].to_numpy(),
            group["_pred"].to_numpy()
        )

        m["category_name"] = category
        m["model"] = model_name
        rows.append(m)

    return pd.DataFrame(rows)[
        [
            "model",
            "category_name",
            "n",
            "rmse",
            "mae",
            "mape_pct",
            "bias",
        ]
    ]


def all_category_breakdowns(
    test_df: pd.DataFrame,
    preds: dict
) -> pd.DataFrame:
    """Generate category-level results for both ML models."""
    results = []

    for model_name in ["random_forest", "xgboost"]:
        results.append(
            category_breakdown(
                test_df,
                preds[model_name],
                model_name
            )
        )

    return pd.concat(results, ignore_index=True)


# ============================================================================
# VOLUME-TIER PERFORMANCE
# ============================================================================

def volume_tier_breakdown(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    y_pred: np.ndarray,
    model_name: str
) -> pd.DataFrame:
    """
    Bin items into low/mid/high volume tiers using TRAIN-period average
    demand only.

    The test-period demand is never used to define the tiers.
    """
    item_avg = train_df.groupby("item_code")[TARGET_COLUMN].mean()

    tiers = pd.qcut(
        item_avg,
        q=3,
        labels=["low_volume", "mid_volume", "high_volume"]
    )

    tier_map = tiers.to_dict()

    df = test_df.copy()
    df["_pred"] = y_pred
    df["volume_tier"] = df["item_code"].map(tier_map)

    df = df.dropna(subset=["volume_tier"])

    rows = []

    for tier, group in df.groupby(
        "volume_tier",
        observed=True
    ):
        m = _metrics(
            group[TARGET_COLUMN].to_numpy(),
            group["_pred"].to_numpy()
        )

        m["volume_tier"] = tier
        m["model"] = model_name
        rows.append(m)

    return pd.DataFrame(rows)[
        [
            "model",
            "volume_tier",
            "n",
            "rmse",
            "mae",
            "mape_pct",
            "bias",
        ]
    ]


def all_volume_tier_breakdowns(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    preds: dict
) -> pd.DataFrame:
    """Generate volume-tier results for both ML models."""
    results = []

    for model_name in ["random_forest", "xgboost"]:
        results.append(
            volume_tier_breakdown(
                test_df,
                train_df,
                preds[model_name],
                model_name
            )
        )

    return pd.concat(results, ignore_index=True)


# ============================================================================
# TOP-10 HIGH-VOLUME ITEMS
# ============================================================================

def top10_volume_mape(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    y_pred: np.ndarray,
    model_name: str
) -> dict:
    """
    Calculate MAPE for the ten highest-volume items.

    Item volume is determined exclusively from the TRAINING period.
    This preserves temporal separation.

    The <=20% threshold is reported as the existing business acceptance
    criterion, but the result is not hidden if the model fails it.
    """
    top10_items = (
        train_df
        .groupby("item_code")[TARGET_COLUMN]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index
        .tolist()
    )

    df = test_df.copy()
    df["_pred"] = y_pred

    subset = df[df["item_code"].isin(top10_items)]

    m = _metrics(
        subset[TARGET_COLUMN].to_numpy(),
        subset["_pred"].to_numpy()
    )

    m["model"] = model_name
    m["top10_item_codes"] = top10_items
    m["meets_20pct_mape_target"] = bool(m["mape_pct"] <= 20.0)

    return m


def all_top10_results(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    preds: dict
) -> pd.DataFrame:
    """Calculate top-10 MAPE for both ML models."""
    rows = []

    for model_name in ["random_forest", "xgboost"]:
        result = top10_volume_mape(
            test_df,
            train_df,
            preds[model_name],
            model_name
        )

        rows.append(
            {
                "model": result["model"],
                "n": result["n"],
                "rmse": result["rmse"],
                "mae": result["mae"],
                "mape_pct": result["mape_pct"],
                "bias": result["bias"],
                "meets_20pct_mape_target":
                    result["meets_20pct_mape_target"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# MODEL SELECTION SUMMARY
# ============================================================================

def model_selection_summary(
    comparison: pd.DataFrame,
    top10_results: pd.DataFrame
) -> dict:
    """
    Provide a transparent evidence summary.

    RMSE is treated as the primary model-selection metric because the
    forecasting task is a continuous demand prediction problem.

    Other metrics remain part of the decision rather than being ignored.
    """
    ml_models = comparison[
        comparison["model"].isin(["random_forest", "xgboost"])
    ].copy()

    best_rmse_model = ml_models.loc[
        ml_models["rmse"].idxmin(),
        "model"
    ]

    best_mae_model = ml_models.loc[
        ml_models["mae"].idxmin(),
        "model"
    ]

    best_mape_model = ml_models.loc[
        ml_models["mape_pct"].idxmin(),
        "model"
    ]

    lowest_bias_model = ml_models.loc[
        ml_models["bias"].abs().idxmin(),
        "model"
    ]

    top10_best = top10_results.loc[
        top10_results["mape_pct"].idxmin(),
        "model"
    ]

    return {
        "primary_metric": "RMSE",
        "best_rmse_model": best_rmse_model,
        "best_mae_model": best_mae_model,
        "best_mape_model": best_mape_model,
        "lowest_absolute_bias_model": lowest_bias_model,
        "best_top10_mape_model": top10_best,
        "selection_status":
            "provisional; review category and volume-tier performance "
            "before final model selection",
    }


# ============================================================================
# PLOTS
# ============================================================================

def make_plots(
    test_df: pd.DataFrame,
    preds: dict,
    plot_model: str = "xgboost"
):
    """
    Create validation plots.

    XGBoost filenames are retained because they are part of the expected
    Module 4 artifact structure. The plots are not used to declare XGBoost
    the final model.
    """
    PLOTS_DIR.mkdir(
        exist_ok=True,
        parents=True
    )

    y_true = test_df[TARGET_COLUMN].to_numpy()

    # ------------------------------------------------------------------
    # 1. Model comparison
    # ------------------------------------------------------------------

    model_names = [
        "seasonal_naive",
        "random_forest",
        "xgboost"
    ]

    metric_specs = [
        ("rmse", "RMSE (kg)", "model_comparison_rmse.png"),
        ("mae", "MAE (kg)", "model_comparison_mae.png"),
        ("mape_pct", "MAPE (%)", "model_comparison_mape.png"),
    ]

    for metric, title, filename in metric_specs:
        values = [
            _metrics(y_true, preds[model])[metric]
            for model in model_names
        ]

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.bar(model_names, values)
        ax.set_title(f"{title} — Hold-out Test Set")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=20)

        fig.tight_layout()
        fig.savefig(
            PLOTS_DIR / filename,
            dpi=120,
            bbox_inches="tight"
        )
        plt.close(fig)

    # Preserve the expected model_comparison.png artifact.
    fig, ax = plt.subplots(figsize=(8, 5))

    rmse_values = [
        _metrics(y_true, preds[model])["rmse"]
        for model in model_names
    ]

    ax.bar(model_names, rmse_values)
    ax.set_title("Model Comparison — RMSE on Hold-out Test Set")
    ax.set_ylabel("RMSE (kg)")
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "model_comparison.png",
        dpi=120,
        bbox_inches="tight"
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Residual plot
    # ------------------------------------------------------------------

    selected_preds = preds[plot_model]
    residuals = y_true - selected_preds

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        selected_preds,
        residuals,
        alpha=0.15,
        s=8
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_xlabel("Predicted daily_demand_kg")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(
        f"{plot_model.replace('_', ' ').title()} "
        "Residual Plot — Hold-out Test Set"
    )

    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "residual_plot_xgboost.png",
        dpi=120,
        bbox_inches="tight"
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Actual vs predicted
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        y_true,
        selected_preds,
        alpha=0.15,
        s=8
    )

    max_value = max(
        y_true.max(),
        selected_preds.max()
    )

    ax.plot(
        [0, max_value],
        [0, max_value],
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel("Actual daily_demand_kg")
    ax.set_ylabel("Predicted daily_demand_kg")
    ax.set_title(
        f"{plot_model.replace('_', ' ').title()} "
        "Actual vs Predicted — Hold-out Test Set"
    )

    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "actual_vs_predicted_xgboost.png",
        dpi=120,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Plots saved to {PLOTS_DIR}")


# ============================================================================
# REAL LEAKAGE / TEMPORAL CHECKS
# ============================================================================

def leakage_checks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> dict:
    """
    Validate temporal separation and verify that selected lag features
    actually correspond to historical target values.

    The train/test artifacts together represent the complete feature dataset,
    so they are recombined chronologically for the lag validation.
    """
    checks = {}

    # ---------------------------------------------------------------
    # Temporal separation
    # ---------------------------------------------------------------

    checks["train_max_date_before_test_min_date"] = bool(
        train_df["date"].max() < test_df["date"].min()
    )

    checks["no_overlap_in_dates"] = bool(
        len(
            set(train_df["date"]).intersection(
                set(test_df["date"])
            )
        ) == 0
    )

    # ---------------------------------------------------------------
    # Target not included as a feature
    # ---------------------------------------------------------------

    checks["target_not_in_feature_columns"] = (
        TARGET_COLUMN not in FEATURE_COLUMNS
    )

    # ---------------------------------------------------------------
    # Required feature columns exist
    # ---------------------------------------------------------------

    required_lag_features = [
        "lag_1",
        "lag_7",
        "lag_14",
        "price_lag_1",
    ]

    checks["required_historical_features_present"] = all(
        feature in FEATURE_COLUMNS
        for feature in required_lag_features
    )

    # ---------------------------------------------------------------
    # Actual lag validation
    # ---------------------------------------------------------------

    combined = pd.concat(
        [train_df, test_df],
        ignore_index=True
    )

    combined["date"] = pd.to_datetime(combined["date"])

    combined = combined.sort_values(
        ["item_code", "date"]
    ).reset_index(drop=True)

    group_demand = combined.groupby(
        "item_code"
    )[TARGET_COLUMN]

    expected_lag_1 = group_demand.shift(1)
    expected_lag_7 = group_demand.shift(7)
    expected_lag_14 = group_demand.shift(14)

    lag_checks = {}

    for feature_name, expected in [
        ("lag_1", expected_lag_1),
        ("lag_7", expected_lag_7),
        ("lag_14", expected_lag_14),
    ]:
        if feature_name in combined.columns:
            observed = combined[feature_name]

            valid = (
                observed.notna()
                & expected.notna()
            )

            if valid.any():
                lag_checks[feature_name] = bool(
                    np.allclose(
                        observed[valid].to_numpy(),
                        expected[valid].to_numpy(),
                        rtol=1e-6,
                        atol=1e-6
                    )
                )
            else:
                lag_checks[feature_name] = False
        else:
            lag_checks[feature_name] = False

    checks["lag_1_matches_historical_demand"] = lag_checks["lag_1"]
    checks["lag_7_matches_historical_demand"] = lag_checks["lag_7"]
    checks["lag_14_matches_historical_demand"] = lag_checks["lag_14"]

    checks["historical_lag_features_leakage_safe"] = all(
        lag_checks.values()
    )

    return checks


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all():

    # ------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------

    train_df, test_df, rf, xgb = load_everything()

    # ------------------------------------------------------------------
    # Overall comparison
    # ------------------------------------------------------------------

    comparison, preds = overall_comparison(
        test_df,
        rf,
        xgb
    )

    print("=== Overall comparison (hold-out test) ===")
    print(
        comparison.to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Category performance for BOTH models
    # ------------------------------------------------------------------

    print(
        "\n=== Category-level breakdown "
        "(Random Forest + XGBoost) ==="
    )

    category_df = all_category_breakdowns(
        test_df,
        preds
    )

    print(
        category_df
        .sort_values(["category_name", "model"])
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Volume-tier performance for BOTH models
    # ------------------------------------------------------------------

    print(
        "\n=== Volume-tier breakdown "
        "(Random Forest + XGBoost) ==="
    )

    volume_df = all_volume_tier_breakdowns(
        test_df,
        train_df,
        preds
    )

    print(
        volume_df
        .sort_values(["volume_tier", "model"])
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Top-10 high-volume items for BOTH models
    # ------------------------------------------------------------------

    print(
        "\n=== Top-10 highest-volume items: "
        "MAPE acceptance criterion ==="
    )

    top10_df = all_top10_results(
        test_df,
        train_df,
        preds
    )

    print(
        top10_df.to_string(index=False)
    )

    print(
        "\nAcceptance criterion: top-10 volume-item MAPE <= 20%"
    )

    for _, row in top10_df.iterrows():
        print(
            f"  {row['model']}: "
            f"{row['mape_pct']:.2f}% "
            f"| Meets target: "
            f"{bool(row['meets_20pct_mape_target'])}"
        )

    # ------------------------------------------------------------------
    # Provisional model selection
    # ------------------------------------------------------------------

    selection = model_selection_summary(
        comparison,
        top10_df
    )

    print(
        "\n=== Provisional model-selection evidence ==="
    )

    for key, value in selection.items():
        print(
            f"  {key}: {value}"
        )

    # ------------------------------------------------------------------
    # Leakage / temporal checks
    # ------------------------------------------------------------------

    print(
        "\n=== Leakage / temporal separation checks ==="
    )

    leakage = leakage_checks(
        train_df,
        test_df
    )

    for key, value in leakage.items():
        print(
            f"  {key}: {value}"
        )

    # ------------------------------------------------------------------
    # Validation plots
    #
    # We retain XGBoost filenames for compatibility with the required
    # artifact structure. These plots do NOT mean XGBoost has been
    # selected as the final model.
    # ------------------------------------------------------------------

    make_plots(
        test_df,
        preds,
        plot_model="xgboost"
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    out = {
        "overall_comparison":
            comparison.to_dict(orient="records"),

        "category_breakdown":
            category_df.to_dict(orient="records"),

        "volume_tier_breakdown":
            volume_df.to_dict(orient="records"),

        "top10_volume_results":
            top10_df.to_dict(orient="records"),

        "model_selection_evidence":
            selection,

        "leakage_checks":
            leakage,
    }

    with open(
        ARTIFACTS_DIR / "evaluation_summary.json",
        "w"
    ) as f:
        json.dump(
            out,
            f,
            indent=2,
            default=str
        )

    print(
        f"\nEvaluation summary saved to "
        f"{ARTIFACTS_DIR / 'evaluation_summary.json'}"
    )

    return out


if __name__ == "__main__":
    run_all()