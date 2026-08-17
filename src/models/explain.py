"""
Explainable AI & Fairness Analysis (Module 4, Task 6).

Covers, in order:
  1. SHAP global feature importance (XGBoost, the selected model)
  2. SHAP local explanations for individual predictions
  3. LIME local explanation (cross-check against SHAP for one prediction)
  4. Counterfactual explanation ("what change flips the forecast")
  5. Fairness metrics reused from evaluate.py (category / volume-tier MAPE
     gaps) -- demographic-parity-style metrics are not applicable since
     there are no demographic attributes in this dataset (see Task 1 notes)
  6. Bias mitigation: only triggered if a meaningful gap is detected
  7. Sensitivity analysis: perturb key features, observe prediction response

Run: python -m src.models.explain
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.evaluate import _prep_X, category_breakdown, volume_tier_breakdown

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "src" / "models" / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
EXPLAIN_DIR = ARTIFACTS_DIR / "explainability"


def load():
    train_df = pd.read_parquet(ARTIFACTS_DIR / "train_df.parquet")
    test_df = pd.read_parquet(ARTIFACTS_DIR / "test_df.parquet")
    xgb = joblib.load(ARTIFACTS_DIR / "xgboost.joblib")
    return train_df, test_df, xgb


def shap_global(xgb, X_sample: pd.DataFrame):
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(X_sample)

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary_global.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return explainer, shap_values, importance


def shap_local(shap_values, X_sample: pd.DataFrame, row_idx: int, tag: str):
    fig = plt.figure(figsize=(9, 4))
    shap.plots.waterfall(shap_values[row_idx], show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"shap_local_{tag}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    row_shap = shap_values[row_idx]
    contributions = (
        pd.DataFrame({
            "feature": X_sample.columns,
            "value": X_sample.iloc[row_idx].values,
            "shap_contribution": row_shap.values,
        })
        .sort_values("shap_contribution", key=np.abs, ascending=False)
    )
    return contributions


def lime_local(xgb, X_train: pd.DataFrame, X_sample: pd.DataFrame, row_idx: int):
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        mode="regression",
        discretize_continuous=True,
    )
    exp = explainer.explain_instance(
        X_sample.iloc[row_idx].values, xgb.predict, num_features=8
    )
    return exp.as_list()


def counterfactual_search(xgb, X_sample: pd.DataFrame, row_idx: int, target_delta_pct: float = 0.30):
    """Transparent counterfactual: for each numeric feature, grid-search
    its observed 5th-95th percentile range to find how far it must move
    to change the prediction by at least `target_delta_pct`. Grid search
    (not an opaque optimizer) so the result is auditable."""
    base_row = X_sample.iloc[[row_idx]].copy()
    base_pred = float(xgb.predict(base_row)[0])

    numeric_cols = [c for c in X_sample.columns if c != "category_code"]
    results = []
    for col in numeric_cols:
        lo, hi = X_sample[col].quantile(0.05), X_sample[col].quantile(0.95)
        grid = np.linspace(lo, hi, 25)
        trial = pd.concat([base_row] * len(grid), ignore_index=True)
        trial[col] = grid
        preds = xgb.predict(trial)
        delta_pct = (preds - base_pred) / (abs(base_pred) + 1e-6)

        hit = np.where(np.abs(delta_pct) >= target_delta_pct)[0]
        if len(hit) > 0:
            first_hit = hit[np.argmin(np.abs(grid[hit] - base_row[col].values[0]))]
            results.append({
                "feature": col,
                "original_value": float(base_row[col].values[0]),
                "counterfactual_value": float(grid[first_hit]),
                "original_prediction": base_pred,
                "counterfactual_prediction": float(preds[first_hit]),
                "pct_change_in_prediction": float(delta_pct[first_hit] * 100),
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df["abs_feature_shift"] = np.abs(
            results_df["counterfactual_value"] - results_df["original_value"]
        )
        results_df = results_df.sort_values("abs_feature_shift")
    return results_df


def sensitivity_analysis(xgb, X_sample: pd.DataFrame, features_to_test=None,
                          pct_range=(-0.3, -0.15, 0, 0.15, 0.3)):
    if features_to_test is None:
        features_to_test = ["lag_1", "lag_7", "roll_mean_7", "price_lag_1", "item_expanding_mean_demand"]

    base_preds = xgb.predict(X_sample)
    rows = []
    for feat in features_to_test:
        for pct in pct_range:
            perturbed = X_sample.copy()
            perturbed[feat] = perturbed[feat] * (1 + pct)
            preds = xgb.predict(perturbed)
            avg_change_pct = float(np.mean((preds - base_preds) / (np.abs(base_preds) + 1e-6)) * 100)
            rows.append({
                "feature": feat, "input_perturbation_pct": pct * 100,
                "avg_prediction_change_pct": avg_change_pct,
                "prediction_std": float(np.std(preds)),
            })
    return pd.DataFrame(rows)


def run_all():
    EXPLAIN_DIR.mkdir(exist_ok=True, parents=True)
    train_df, test_df, xgb = load()

    X_train = _prep_X(train_df)
    X_test = _prep_X(test_df)

    print("Computing SHAP global explanations...")
    explainer, shap_values, importance = shap_global(xgb, X_test)
    print(importance.to_string(index=False))
    importance.to_csv(EXPLAIN_DIR / "shap_global_importance.csv", index=False)

    preds = xgb.predict(X_test)
    high_idx = int(np.argmax(preds))
    low_idx = int(np.argmin(preds))

    print(f"\nSHAP local explanation for row {high_idx} (high predicted demand)...")
    local_high = shap_local(shap_values, X_test, high_idx, "high_demand_example")
    print(local_high.head(6).to_string(index=False))

    print(f"\nSHAP local explanation for row {low_idx} (low predicted demand)...")
    local_low = shap_local(shap_values, X_test, low_idx, "low_demand_example")
    print(local_low.head(6).to_string(index=False))

    print(f"\nLIME cross-check for row {high_idx}...")
    lime_result = lime_local(xgb, X_train, X_test, high_idx)
    for feat, weight in lime_result:
        print(f"  {feat}: {weight:.4f}")

    print(f"\nCounterfactual search for row {high_idx} (target: +/-30% prediction change)...")
    cf_df = counterfactual_search(xgb, X_test, high_idx)
    print(cf_df.to_string(index=False))
    cf_df.to_csv(EXPLAIN_DIR / "counterfactual_example.csv", index=False)

    print("\nFairness: category-level MAPE gap check...")
    cat_df = category_breakdown(test_df, preds, "xgboost")
    mape_range = cat_df["mape_pct"].max() - cat_df["mape_pct"].min()
    print(cat_df.sort_values("mape_pct").to_string(index=False))
    print(f"MAPE range across categories: {mape_range:.1f} percentage points")

    print("\nFairness: volume-tier MAPE gap check...")
    vol_df = volume_tier_breakdown(test_df, train_df, preds, "xgboost")
    print(vol_df.to_string(index=False))
    vol_mape_range = vol_df["mape_pct"].max() - vol_df["mape_pct"].min()
    print(f"MAPE range across volume tiers: {vol_mape_range:.1f} percentage points")

    MITIGATION_THRESHOLD_PP = 25.0
    mitigation_triggered = vol_mape_range > MITIGATION_THRESHOLD_PP
    print(f"\nMitigation threshold: {MITIGATION_THRESHOLD_PP}pp | "
          f"Observed volume-tier gap: {vol_mape_range:.1f}pp | "
          f"Mitigation triggered: {mitigation_triggered}")

    print("\nSensitivity analysis...")
    sens_df = sensitivity_analysis(xgb, X_test.sample(min(500, len(X_test)), random_state=42))
    print(sens_df.to_string(index=False))
    sens_df.to_csv(EXPLAIN_DIR / "sensitivity_analysis.csv", index=False)

    summary = {
        "shap_top_features": importance.head(8).to_dict(orient="records"),
        "category_mape_range_pp": float(mape_range),
        "volume_tier_mape_range_pp": float(vol_mape_range),
        "mitigation_threshold_pp": MITIGATION_THRESHOLD_PP,
        "mitigation_triggered": bool(mitigation_triggered),
        "lime_cross_check_top_features": lime_result,
    }
    with open(EXPLAIN_DIR / "explainability_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved explainability outputs to {EXPLAIN_DIR}")

    return summary


if __name__ == "__main__":
    run_all()