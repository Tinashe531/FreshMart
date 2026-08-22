"""
Module 5 Evidence Layer
========================

Purpose: Prepare and validate the Module 4 model's outputs for presentation
in the Module 5 stakeholder dashboard and deck.

This script does NOT retrain, tune, or alter the Module 4 model. It loads
the existing Module 4 artifacts (xgboost_final.joblib, train/test splits,
SHAP outputs) and produces a Module 5 evidence package: performance metrics
(including robustness metrics MAPE doesn't provide), a properly investigated
fairness analysis, tier-stratified 95% demand ranges with coverage
validation, explainability content, and per-item forward forecasts for the
next replenishment cycle.

Layer boundary:
    Module 4 artifacts (untouched)  -->  this script  -->  module5_outputs/

Run (from the repo root):
    python src/evaluation/evidence.py
"""

import sys
import json
import warnings
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib
import shap

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent          # .../src/evaluation
PROJECT_ROOT = SCRIPT_DIR.parent.parent                # repo root (FreshMart/)
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.features import FEATURE_COLUMNS, TARGET_COLUMN

ARTIFACTS_DIR = PROJECT_ROOT / "src" / "models" / "artifacts"
OUTPUT_DIR = SCRIPT_DIR  # bundle is written next to this script, not a separate output folder

NEAR_ZERO_THRESHOLD_KG = 2.0
MITIGATION_THRESHOLD_PP = 25.0  # matches Module 4's own trigger, for consistency


# ============================================================================
# LOAD MODULE 4 ARTIFACTS (read-only)
# ============================================================================

def load_module4_artifacts():
    print("Loading Module 4 artifacts (read-only)...")
    train_df = pd.read_parquet(ARTIFACTS_DIR / "train_df.parquet")
    test_df = pd.read_parquet(ARTIFACTS_DIR / "test_df.parquet")
    model_final = joblib.load(ARTIFACTS_DIR / "xgboost_final.joblib")   # mitigated
    model_before = joblib.load(ARTIFACTS_DIR / "xgboost.joblib")        # pre-mitigation
    item_day = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "freshmart_item_day.parquet")
    item_day["date"] = pd.to_datetime(item_day["date"])

    with open(ARTIFACTS_DIR / "results_summary.json") as f:
        results_summary = json.load(f)
    with open(ARTIFACTS_DIR / "explainability" / "explainability_summary.json") as f:
        explain_summary = json.load(f)
    with open(ARTIFACTS_DIR / "explainability" / "mitigation_summary.json") as f:
        mitigation_summary = json.load(f)

    print(f"  train_df: {train_df.shape}, test_df: {test_df.shape}")
    print(f"  item_day (full history): {item_day.shape}")
    return {
        "train_df": train_df, "test_df": test_df,
        "model_final": model_final, "model_before": model_before,
        "item_day": item_day,
        "results_summary": results_summary,
        "explain_summary": explain_summary,
        "mitigation_summary": mitigation_summary,
    }


def _prep_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    X["category_code"] = X["category_code"].astype("category").cat.codes
    return X


# ============================================================================
# METRIC FUNCTIONS (MAE/RMSE/R2 preserved from Module 4; MAPE/WAPE/SMAPE/
# MedianAPE computed consistently for every breakdown that follows)
# ============================================================================

def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    abs_err = np.abs(err)

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(abs_err))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    nonzero = y_true != 0
    mape = float(np.mean(abs_err[nonzero] / np.abs(y_true[nonzero])) * 100) if nonzero.any() else float("nan")
    mdape = float(np.median(abs_err[nonzero] / np.abs(y_true[nonzero])) * 100) if nonzero.any() else float("nan")

    wape = float(abs_err.sum() / y_true.sum() * 100) if y_true.sum() != 0 else float("nan")

    denom = np.abs(y_true) + np.abs(y_pred)
    smape_mask = denom != 0
    smape = float(np.mean(2 * abs_err[smape_mask] / denom[smape_mask]) * 100) if smape_mask.any() else float("nan")

    return {
        "n": int(len(y_true)), "mae": mae, "rmse": rmse, "r2": r2,
        "mape_pct": mape, "wape_pct": wape, "smape_pct": smape, "median_ape_pct": mdape,
    }


# ============================================================================
# STEP 1: REPRODUCE MODULE 4'S OFFICIAL RESULTS (sanity check before anything else)
# ============================================================================

def reproduce_module4(data: dict) -> dict:
    print("\n=== STEP 1: Reproducing Module 4 official results ===")
    test_df = data["test_df"]
    y_true = test_df[TARGET_COLUMN].to_numpy()
    X_test = _prep_X(test_df)

    y_pred_before = data["model_before"].predict(X_test)
    y_pred_final = data["model_final"].predict(X_test)

    m_before = compute_metrics(y_true, y_pred_before)
    m_final = compute_metrics(y_true, y_pred_final)

    reported = data["results_summary"]["xgboost"]["test"]
    checks = {
        "rmse_before_matches_reported": np.isclose(m_before["rmse"], reported["test_rmse"], atol=0.01),
        "mape_before_matches_reported": np.isclose(m_before["mape_pct"], reported["test_mape_pct"], atol=0.1),
    }
    mit = data["mitigation_summary"]
    checks["rmse_after_matches_mitigation_summary"] = np.isclose(
        m_final["rmse"], mit["overall_after"]["rmse"], atol=0.01
    )

    print(f"  Pre-mitigation  RMSE={m_before['rmse']:.3f} (reported {reported['test_rmse']:.3f}) "
          f"MAPE={m_before['mape_pct']:.2f}% (reported {reported['test_mape_pct']:.2f}%)")
    print(f"  Post-mitigation RMSE={m_final['rmse']:.3f} (mitigation_summary {mit['overall_after']['rmse']:.3f})")
    print(f"  Reproduction checks: {checks}")

    if not all(checks.values()):
        raise ValueError(
            "Reproduction of Module 4 results FAILED -- do not proceed until this "
            "is resolved. Module 5 must be built on the same model/data as Module 4."
        )
    print("  ALL CHECKS PASSED -- Module 5 evidence layer is built on the correct, "
          "unmodified Module 4 model and hold-out data.")

    return {
        "y_true": y_true,
        "y_pred_before": y_pred_before,
        "y_pred_final": y_pred_final,
        "reproduction_checks": {k: bool(v) for k, v in checks.items()},
    }


# ============================================================================
# STEP 2: OVERALL + TOP-10 METRICS (MAE/RMSE/R2/MAPE preserved, WAPE/SMAPE/
# MedianAPE added as supplementary context)
# ============================================================================

def top10_items_by_train_volume(train_df: pd.DataFrame) -> list:
    """Exact Module 4 definition: top-10 items by TOTAL train-period volume."""
    return (
        train_df.groupby("item_code")[TARGET_COLUMN].sum()
        .sort_values(ascending=False).head(10).index.tolist()
    )


def overall_and_top10_metrics(data: dict, repro: dict) -> pd.DataFrame:
    print("\n=== STEP 2: Overall + Top-10 metrics (final/mitigated model) ===")
    test_df = data["test_df"].copy()
    test_df["y_pred"] = repro["y_pred_final"]

    top10_items = top10_items_by_train_volume(data["train_df"])
    top10_df = test_df[test_df["item_code"].isin(top10_items)]
    n_top10_items_present = top10_df["item_code"].nunique()

    rows = []
    overall = compute_metrics(test_df[TARGET_COLUMN], test_df["y_pred"])
    overall["scope"] = "overall"
    overall["n_items"] = test_df["item_code"].nunique()
    rows.append(overall)

    t10 = compute_metrics(top10_df[TARGET_COLUMN], top10_df["y_pred"])
    t10["scope"] = "top10_by_train_volume"
    t10["n_items"] = n_top10_items_present
    t10["n_items_defined_as_top10"] = 10
    t10["meets_20pct_mape_target"] = bool(t10["mape_pct"] <= 20.0)
    rows.append(t10)

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    if n_top10_items_present < 10:
        print(f"  NOTE: only {n_top10_items_present}/10 top-volume items have any "
              f"test-period rows -- {10 - n_top10_items_present} item(s) had no sales "
              f"recorded in the hold-out window.")
    return out


# ============================================================================
# STEP 3: VOLUME TIERS (exact Module 4 definition: 3-way qcut on TRAIN mean)
# ============================================================================

def assign_volume_tiers(train_df: pd.DataFrame) -> dict:
    """Item -> tier map. Tiers defined from TRAIN period only (no leakage)."""
    item_avg = train_df.groupby("item_code")[TARGET_COLUMN].mean()
    tiers = pd.qcut(item_avg, q=3, labels=["low_volume", "mid_volume", "high_volume"])
    return tiers.to_dict()


def volume_tier_metrics(test_df: pd.DataFrame, tier_map: dict, y_pred: np.ndarray, label: str) -> pd.DataFrame:
    df = test_df.copy()
    df["y_pred"] = y_pred
    df["volume_tier"] = df["item_code"].map(tier_map)
    df = df.dropna(subset=["volume_tier"])

    rows = []
    for tier, g in df.groupby("volume_tier", observed=True):
        m = compute_metrics(g[TARGET_COLUMN], g["y_pred"])
        m["volume_tier"] = tier
        m["model_stage"] = label
        m["pct_rows_near_zero"] = float((g[TARGET_COLUMN] < NEAR_ZERO_THRESHOLD_KG).mean() * 100)
        m["total_demand_kg"] = float(g[TARGET_COLUMN].sum())
        rows.append(m)
    return pd.DataFrame(rows)


def near_zero_breakdown(test_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    df = test_df.copy()
    df["y_pred"] = y_pred
    df["group"] = np.where(df[TARGET_COLUMN] < NEAR_ZERO_THRESHOLD_KG, "below_2kg", "at_or_above_2kg")
    rows = []
    for grp, g in df.groupby("group"):
        m = compute_metrics(g[TARGET_COLUMN], g["y_pred"])
        m["group"] = grp
        m["pct_of_all_rows"] = float(len(g) / len(df) * 100)
        rows.append(m)
    out = pd.DataFrame(rows)
    # Also give the distribution just below/above the 2kg line, to check it isn't an arbitrary cliff
    print("  Demand distribution near the 2kg threshold (to check it's not an arbitrary cutoff):")
    for lo, hi in [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 5)]:
        n = ((df[TARGET_COLUMN] >= lo) & (df[TARGET_COLUMN] < hi)).sum()
        print(f"    [{lo}, {hi}) kg: {n} rows")
    return out


def category_metrics(test_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """Performance by vegetable category -- the other half of Tab 3's
    'Forecast Performance Across Vegetables' requirement (volume tier is
    the other half, computed in fairness_and_mitigation)."""
    df = test_df.copy()
    df["y_pred"] = y_pred
    rows = []
    for cat, g in df.groupby("category_name"):
        m = compute_metrics(g[TARGET_COLUMN], g["y_pred"])
        m["category_name"] = cat
        m["pct_rows_near_zero"] = float((g[TARGET_COLUMN] < NEAR_ZERO_THRESHOLD_KG).mean() * 100)
        rows.append(m)
    out = pd.DataFrame(rows).sort_values("wape_pct")
    return out


def build_baseline_comparison(data: dict) -> pd.DataFrame:
    """XGBoost (final, mitigated) vs the seasonal-naive baseline Module 4
    already computed and saved -- reused here, not recalculated."""
    b = data["results_summary"]["baseline"]
    x = data["results_summary"]["xgboost"]["test"]
    rows = [
        {"model": "seasonal_naive_baseline", "mae": b["mae"], "rmse": b["rmse"],
         "mape_pct": b["mape_pct"], "r2": None},
        {"model": "xgboost_pre_mitigation", "mae": x["test_mae"], "rmse": x["test_rmse"],
         "mape_pct": x["test_mape_pct"], "r2": x["test_r2"]},
    ]
    return pd.DataFrame(rows)


def fairness_and_mitigation(data: dict, repro: dict) -> dict:
    print("\n=== STEP 3: Volume-tier fairness + mitigation comparison ===")
    train_df, test_df = data["train_df"], data["test_df"]
    tier_map = assign_volume_tiers(train_df)

    tier_before = volume_tier_metrics(test_df, tier_map, repro["y_pred_before"], "before_mitigation")
    tier_after = volume_tier_metrics(test_df, tier_map, repro["y_pred_final"], "after_mitigation")
    tier_all = pd.concat([tier_before, tier_after], ignore_index=True)
    print(tier_all.to_string(index=False))

    def gap(df, metric):
        return float(df[metric].max() - df[metric].min())

    gaps = []
    for stage, df in [("before_mitigation", tier_before), ("after_mitigation", tier_after)]:
        for metric in ["mape_pct", "wape_pct", "smape_pct", "median_ape_pct"]:
            gaps.append({"stage": stage, "metric": metric, "gap_pp": gap(df, metric)})
    gaps_df = pd.DataFrame(gaps)
    print("\n  Gap (max tier - min tier) by metric and mitigation stage:")
    print(gaps_df.to_string(index=False))

    # Explicit before/after comparison table, per the plan
    comparison_rows = []
    for tier in ["low_volume", "mid_volume", "high_volume"]:
        b = tier_before[tier_before["volume_tier"] == tier].iloc[0]
        a = tier_after[tier_after["volume_tier"] == tier].iloc[0]
        comparison_rows.append({
            "volume_tier": tier,
            "wape_before": b["wape_pct"], "wape_after": a["wape_pct"], "wape_change": a["wape_pct"] - b["wape_pct"],
            "mape_before": b["mape_pct"], "mape_after": a["mape_pct"], "mape_change": a["mape_pct"] - b["mape_pct"],
            "mae_before": b["mae"], "mae_after": a["mae"], "mae_change": a["mae"] - b["mae"],
        })
    mitigation_effect = pd.DataFrame(comparison_rows)
    print("\n  Mitigation effect by tier:")
    print(mitigation_effect.to_string(index=False))

    overall_before = compute_metrics(test_df[TARGET_COLUMN], repro["y_pred_before"])
    overall_after = compute_metrics(test_df[TARGET_COLUMN], repro["y_pred_final"])
    print(f"\n  Overall WAPE: before={overall_before['wape_pct']:.2f}%  after={overall_after['wape_pct']:.2f}%")
    print(f"  Overall MAE:  before={overall_before['mae']:.3f}  after={overall_after['mae']:.3f}")

    print("\n  Near-zero demand breakdown (final/mitigated model):")
    nz = near_zero_breakdown(test_df, repro["y_pred_final"])
    print(nz.to_string(index=False))

    return {
        "tier_metrics": tier_all, "tier_gaps": gaps_df,
        "mitigation_effect": mitigation_effect, "near_zero": nz,
        "overall_before": overall_before, "overall_after": overall_after,
        "tier_map": tier_map,
    }


# ============================================================================
# STEP 4: 95% DEMAND RANGES (tier-stratified empirical residual quantiles)
# with honest coverage validation on the SAME hold-out set used to calibrate
# ============================================================================

def build_prediction_ranges(data: dict, repro: dict, fairness: dict) -> dict:
    print("\n=== STEP 4: 95% demand ranges (tier-stratified residual quantiles) ===")
    test_df = data["test_df"].copy()
    test_df["y_pred"] = repro["y_pred_final"]
    test_df["resid"] = test_df[TARGET_COLUMN] - test_df["y_pred"]
    test_df["volume_tier"] = test_df["item_code"].map(fairness["tier_map"])
    test_df = test_df.dropna(subset=["volume_tier"])

    tier_bounds = {}
    coverage_rows = []
    for tier, g in test_df.groupby("volume_tier", observed=True):
        q_lo, q_hi = np.quantile(g["resid"], [0.025, 0.975])
        tier_bounds[tier] = {"resid_q_lo": float(q_lo), "resid_q_hi": float(q_hi)}

        lo = (g["y_pred"] + q_lo).clip(lower=0)
        hi = g["y_pred"] + q_hi
        covered = ((g[TARGET_COLUMN] >= lo) & (g[TARGET_COLUMN] <= hi)).mean() * 100
        width = (hi - lo)
        width_pct_per_row = (width / g["y_pred"].replace(0, np.nan)) * 100

        coverage_rows.append({
            "volume_tier": tier, "n": len(g),
            "resid_q_lo": float(q_lo), "resid_q_hi": float(q_hi),
            "mean_range_width_kg": float(width.mean()),
            "mean_width_pct_of_forecast": float(width_pct_per_row.mean()),
            "median_width_pct_of_forecast": float(width_pct_per_row.median()),
            "empirical_coverage_pct": float(covered),
            "target_coverage_pct": 95.0,
        })

    coverage_df = pd.DataFrame(coverage_rows)
    print(coverage_df.to_string(index=False))
    print("\n  NOTE: mean width-pct-of-forecast is inflated by a small number of rows "
          "with unusually small day-level forecasts (same near-zero-denominator issue "
          "as MAPE) -- median_width_pct_of_forecast is the more representative figure "
          "and is what should be used in the dashboard/deck.")
    print("\n  NOTE: coverage is validated on the SAME hold-out period used to "
          "calibrate the residual quantiles. This confirms internal consistency, "
          "not generalization to genuinely unseen future data -- stated explicitly "
          "in Model Limitations.")

    return {"tier_bounds": tier_bounds, "coverage": coverage_df}


# ============================================================================
# STEP 5: EXPLAINABILITY (reuse Module 4's saved global SHAP; compute local
# SHAP only for the specific example forecasts we choose in Step 6)
# ============================================================================

def load_global_shap(data: dict) -> pd.DataFrame:
    print("\n=== STEP 5: Explainability (reusing Module 4's saved global SHAP) ===")
    path = ARTIFACTS_DIR / "explainability" / "shap_global_importance.csv"
    global_shap = pd.read_csv(path)
    print(global_shap.head(8).to_string(index=False))
    return global_shap


FEATURE_PLAIN_LANGUAGE = {
    "lag_1": "demand from the previous recorded day",
    "lag_7": "demand from a week earlier",
    "lag_14": "demand from two weeks earlier",
    "roll_mean_7": "average demand over the last 7 recorded days",
    "roll_mean_14": "average demand over the last 14 recorded days",
    "roll_std_7": "how much recent demand has been fluctuating",
    "item_expanding_mean_demand": "this item's typical average demand historically",
    "day_of_week": "the day of the week",
    "is_weekend": "whether it falls on a weekend",
    "is_replenishment_day": "whether it's a scheduled replenishment day",
    "month": "the time of year (month)",
    "day_of_year": "seasonal timing within the year",
    "week_of_year": "seasonal timing within the year",
    "price_lag_1": "the most recent selling price",
    "price_roll_mean_7": "the recent average selling price",
    "price_change_pct_7": "how much the price has recently changed",
    "category_code": "the vegetable's category",
}


def _plain_language_shap_description(top_positive: list, top_negative: list) -> str:
    """Shared by both the precomputed example forecasts (Step 7) and the
    live per-item explanation used in the Explorer tab (Tab 2), so wording
    stays identical whether the driver came from a batch run or a live call."""
    def describe(drivers, direction):
        if not drivers:
            return f"No strong {direction} factors were identified."
        parts = [FEATURE_PLAIN_LANGUAGE.get(d["feature"], d["feature"]) for d in drivers]
        return f"The forecast is being pushed {direction} mainly by " + ", ".join(parts) + "."
    return describe(top_positive, "up") + " " + describe(top_negative, "down")


def local_shap_for_rows(model, X_full_train, rows: pd.DataFrame) -> list:
    """Compute local SHAP for a small set of specific rows (our chosen examples)."""
    explainer = shap.TreeExplainer(model)
    X_rows = rows[FEATURE_COLUMNS].copy()
    X_rows["category_code"] = X_rows["category_code"].astype("category").cat.codes
    sv = explainer.shap_values(X_rows)

    results = []
    for i in range(len(X_rows)):
        contribs = list(zip(FEATURE_COLUMNS, sv[i]))
        contribs.sort(key=lambda t: -abs(t[1]))
        top_positive = [(f, v) for f, v in contribs if v > 0][:3]
        top_negative = [(f, v) for f, v in contribs if v < 0][:3]
        results.append({
            "top_positive_drivers": [{"feature": f, "shap_value": float(v)} for f, v in top_positive],
            "top_negative_drivers": [{"feature": f, "shap_value": float(v)} for f, v in top_negative],
        })
    return results


# ============================================================================
# STEP 6: FORWARD FORECASTS -- next replenishment cycle, per item
# (row-based lag/rolling features, matching Module 4's exact construction,
#  built from each item's own most recent recorded history)
# ============================================================================

from src.models.features import REPLENISHMENT_WEEKDAYS

CATEGORY_CODE_MAP = None  # built once, reused everywhere for safety


def build_category_code_map(item_day: pd.DataFrame) -> dict:
    codes = sorted(item_day["category_code"].unique())
    return {raw: i for i, raw in enumerate(codes)}


def next_replenishment_date(last_date: pd.Timestamp) -> pd.Timestamp:
    d = last_date + pd.Timedelta(days=1)
    for _ in range(14):
        if d.weekday() in REPLENISHMENT_WEEKDAYS:
            return d
        d += pd.Timedelta(days=1)
    raise ValueError("No replenishment day found within 14 days -- check REPLENISHMENT_WEEKDAYS")


def build_forward_forecast_row(item_hist: pd.DataFrame, cat_map: dict) -> dict | None:
    """item_hist: one item's full history, sorted by date ascending.
    Returns a feature dict for the next replenishment cycle, or None if the
    item doesn't have enough history to build required lag/rolling features
    (mirrors the min-history rule Module 4 applied during training)."""
    if len(item_hist) < 14:
        return None

    last_row = item_hist.iloc[-1]
    last_date = last_row["date"]
    forecast_date = next_replenishment_date(last_date)

    demand = item_hist["daily_demand_kg"].values
    price = item_hist["avg_selling_price"].values

    lag_1, lag_7, lag_14 = demand[-1], demand[-7], demand[-14]
    roll_mean_7 = demand[-7:].mean()
    roll_mean_14 = demand[-14:].mean()
    roll_std_7 = demand[-7:].std(ddof=0) if len(demand) >= 7 else np.nan
    price_lag_1 = price[-1]
    price_roll_mean_7 = price[-7:].mean()
    price_8back = price[-8] if len(price) >= 8 else np.nan
    price_change_pct_7 = (price_lag_1 - price_8back) / price_8back if price_8back not in (0, np.nan) else np.nan
    item_expanding_mean_demand = demand.mean()

    feats = {
        "item_code": last_row["item_code"],
        "category_name": last_row["category_name"],
        "category_code_raw": last_row["category_code"],
        "category_code": cat_map[last_row["category_code"]],
        "last_known_date": last_date,
        "forecast_date": forecast_date,
        "n_history_rows": len(item_hist),
        "day_of_week": forecast_date.weekday(),
        "is_weekend": int(forecast_date.weekday() in (5, 6)),
        "is_replenishment_day": 1,
        "month": forecast_date.month,
        "day_of_year": forecast_date.dayofyear,
        "week_of_year": int(forecast_date.isocalendar().week),
        "lag_1": lag_1, "lag_7": lag_7, "lag_14": lag_14,
        "roll_mean_7": roll_mean_7, "roll_mean_14": roll_mean_14, "roll_std_7": roll_std_7,
        "price_lag_1": price_lag_1, "price_roll_mean_7": price_roll_mean_7,
        "price_change_pct_7": price_change_pct_7,
        "item_expanding_mean_demand": item_expanding_mean_demand,
    }
    return feats


def build_all_forward_forecasts(data: dict, model, tier_map: dict, tier_bounds: dict) -> pd.DataFrame:
    print("\n=== STEP 6: Forward forecasts (next replenishment cycle, per item) ===")
    item_day = data["item_day"].sort_values(["item_code", "date"])
    cat_map = build_category_code_map(item_day)
    global_max_date = item_day["date"].max()

    all_items = item_day["item_code"].unique()
    rows = []
    for item_code, g in item_day.groupby("item_code"):
        feats = build_forward_forecast_row(g, cat_map)
        last_row = g.iloc[-1]
        if feats is None:
            # Still include the item so the selector/UI can list every item and
            # show an honest "no forecast available" status, rather than silently
            # omitting 62 items from the dashboard entirely.
            rows.append({
                "item_code": item_code,
                "category_name": last_row["category_name"],
                "last_known_date": last_row["date"],
                "n_history_rows": len(g),
                "forecast_date": pd.NaT,
                "forecast_status": "No forecast available",
            })
            continue
        rows.append(feats)

    fwd = pd.DataFrame(rows)

    has_forecast = fwd["forecast_date"].notna()
    X_fwd = fwd.loc[has_forecast, FEATURE_COLUMNS].copy()
    fwd.loc[has_forecast, "forecast_demand_kg"] = model.predict(X_fwd)
    fwd["volume_tier"] = fwd["item_code"].map(tier_map)

    # --- Currentness status (Fix #3) ---
    # Reference point is the dataset's own most recent recorded date (our
    # stand-in for "today" in this offline demo), NOT the real calendar date.
    fwd["days_since_last_activity"] = (global_max_date - fwd["last_known_date"]).dt.days
    fwd.loc[~has_forecast, "days_since_last_activity"] = (global_max_date - fwd.loc[~has_forecast, "last_known_date"]).dt.days

    def status(row):
        if pd.isna(row["forecast_date"]):
            return "No forecast available"
        return "Current forecast" if row["days_since_last_activity"] <= 90 else "Older activity \u2014 treat with caution"

    fwd["forecast_status"] = fwd.apply(status, axis=1)

    # Apply tier-stratified 95% range (only where a forecast exists)
    def apply_range(row):
        if pd.isna(row["forecast_demand_kg"]):
            return pd.Series({"range_lower_kg": np.nan, "range_upper_kg": np.nan})
        tb = tier_bounds.get(row["volume_tier"])
        if tb is None:
            return pd.Series({"range_lower_kg": np.nan, "range_upper_kg": np.nan})
        lo = max(row["forecast_demand_kg"] + tb["resid_q_lo"], 0)
        hi = row["forecast_demand_kg"] + tb["resid_q_hi"]
        return pd.Series({"range_lower_kg": lo, "range_upper_kg": hi})

    fwd = pd.concat([fwd, fwd.apply(apply_range, axis=1)], axis=1)
    fwd["range_width_kg"] = fwd["range_upper_kg"] - fwd["range_lower_kg"]

    # --- Unambiguous "recent demand" naming (Fix #4) ---
    # previous_recorded_day_kg = lag_1 (single most recent recorded day)
    # recent_demand_avg_kg     = roll_mean_7 (7-recorded-day average) -- this
    #   is the definition used for "recent demand" and the change-vs-recent
    #   calculation throughout the app. lag_1 is shown separately, labeled
    #   "previous recorded day", never also called "recent demand".
    fwd["previous_recorded_day_kg"] = fwd.get("lag_1")
    fwd["recent_demand_avg_kg"] = fwd.get("roll_mean_7")
    fwd["change_vs_recent_demand_avg_pct"] = (
        (fwd["forecast_demand_kg"] - fwd["recent_demand_avg_kg"]) / fwd["recent_demand_avg_kg"].replace(0, np.nan) * 100
    )

    fwd["display_label"] = fwd["category_name"] + " \u2014 Item " + fwd["item_code"].astype(str).str[-6:]

    n_forecastable = has_forecast.sum()
    n_current = (fwd["forecast_status"] == "Current forecast").sum()
    n_older = (fwd["forecast_status"].str.startswith("Older", na=False)).sum()
    n_none = (fwd["forecast_status"] == "No forecast available").sum()
    print(f"  {len(fwd)} items total: {n_forecastable} forecastable "
          f"({n_current} 'Current forecast', {n_older} 'Older activity'), "
          f"{n_none} 'No forecast available' (insufficient history)")
    print(fwd[["item_code", "category_name", "forecast_status", "days_since_last_activity",
                "forecast_demand_kg", "range_lower_kg", "range_upper_kg"]].head(5).to_string(index=False))

    return fwd


# ============================================================================
# STEP 7: EXAMPLE FORECASTS -- one per volume tier, chosen by rule (not by
# how flattering they look), with local SHAP + plain-language explanation
# ============================================================================

def pick_example_items(forward: pd.DataFrame) -> pd.DataFrame:
    """One example per tier: the item closest to its tier's MEDIAN forecast
    value, and with a forecast_date reasonably close to the dataset's most
    recent date (so the example looks like a genuinely current forecast,
    not a stale one from an item that stopped selling years ago)."""
    global_max_date = forward["forecast_date"].max()
    has_forecast = forward["forecast_demand_kg"].notna()
    recent = forward[has_forecast & (forward["last_known_date"] >= global_max_date - pd.Timedelta(days=180))]

    examples = []
    for tier in ["low_volume", "mid_volume", "high_volume"]:
        pool = recent[recent["volume_tier"] == tier]
        if pool.empty:
            pool = forward[has_forecast & (forward["volume_tier"] == tier)]  # fall back if none recent
        median_val = pool["forecast_demand_kg"].median()
        idx = (pool["forecast_demand_kg"] - median_val).abs().idxmin()
        examples.append(pool.loc[idx])
    return pd.DataFrame(examples)


def build_examples_with_shap(data: dict, forward: pd.DataFrame) -> list:
    print("\n=== STEP 7: Example forecasts (one per tier, median-representative) ===")
    examples_df = pick_example_items(forward)
    print(examples_df[["item_code", "category_name", "volume_tier", "forecast_date",
                        "forecast_demand_kg", "range_lower_kg", "range_upper_kg"]].to_string(index=False))

    local = local_shap_for_rows(data["model_final"], None, examples_df)

    results = []
    for (_, row), shap_info in zip(examples_df.iterrows(), local):
        pos = shap_info["top_positive_drivers"]
        neg = shap_info["top_negative_drivers"]
        plain_explanation = _plain_language_shap_description(pos, neg)

        width_pct = (row["range_upper_kg"] - row["range_lower_kg"]) / row["forecast_demand_kg"] * 100 if row["forecast_demand_kg"] else np.nan

        results.append({
            "item_code": row["item_code"],
            "display_label": row["display_label"],
            "category_name": row["category_name"],
            "volume_tier": row["volume_tier"],
            "forecast_date": str(row["forecast_date"]),
            "forecast_demand_kg": round(float(row["forecast_demand_kg"]), 2),
            "previous_recorded_day_kg": round(float(row["previous_recorded_day_kg"]), 2),
            "recent_demand_avg_kg": round(float(row["recent_demand_avg_kg"]), 2),
            "range_lower_kg": round(float(row["range_lower_kg"]), 2),
            "range_upper_kg": round(float(row["range_upper_kg"]), 2),
            "range_width_pct_of_forecast": round(float(width_pct), 1) if pd.notna(width_pct) else None,
            "top_positive_drivers": pos,
            "top_negative_drivers": neg,
            "plain_language_explanation": plain_explanation,
            "uncertainty_note": (
                f"This forecast's 95% range spans about {width_pct:.0f}% of the point forecast itself -- "
                f"treat it as a guide, not a precise number."
            ) if pd.notna(width_pct) else "",
        })

    return results


# ============================================================================
# STEP 8: WHAT-IF REFERENCE -- full feature documentation + a business-
# interpretable subset safe to expose as interactive inputs
# ============================================================================

WHAT_IF_INTERPRETABLE_FEATURES = ["lag_1", "roll_mean_7", "price_lag_1", "day_of_week"]


@lru_cache(maxsize=1)
def _load_model_for_live_predictions():
    """Lightweight, cached model load for live What-If calls -- deliberately
    separate from load_module4_artifacts(), which pulls in the full
    train/test/item_day data and is far too heavy to call on every user
    interaction in the app."""
    return joblib.load(ARTIFACTS_DIR / "xgboost_final.joblib")


def load_baseline_row(forward_forecasts: pd.DataFrame, item_code) -> dict:
    """Pull one item's baseline feature vector out of the precomputed
    forward_forecasts table -- it already contains every FEATURE_COLUMNS
    value, since that's exactly what generated the stored forecast."""
    row = forward_forecasts[forward_forecasts["item_code"] == item_code]
    if row.empty:
        raise ValueError(f"No baseline row found for item_code={item_code}")
    row = row.iloc[0]
    if pd.isna(row.get("forecast_demand_kg")):
        raise ValueError(
            f"item_code={item_code} has forecast_status='{row.get('forecast_status')}' -- "
            f"What-If analysis requires a valid baseline forecast, which this item doesn't have "
            f"(insufficient history)."
        )
    return row[FEATURE_COLUMNS].to_dict()


def _apply_whatif_overrides(baseline: dict, overrides: dict) -> dict:
    bad_keys = set(overrides) - set(WHAT_IF_INTERPRETABLE_FEATURES)
    if bad_keys:
        raise ValueError(
            f"These inputs are not exposed for What-If manipulation: {sorted(bad_keys)}. "
            f"Only {WHAT_IF_INTERPRETABLE_FEATURES} may be changed by the user."
        )
    scenario = dict(baseline)
    scenario.update(overrides)
    # Keep dependent calendar features consistent if day_of_week changes.
    if "day_of_week" in overrides:
        dow = int(overrides["day_of_week"])
        scenario["is_weekend"] = int(dow in (5, 6))
        scenario["is_replenishment_day"] = int(dow in REPLENISHMENT_WEEKDAYS)
    return scenario


def _predict_one(model, feature_dict: dict) -> float:
    X = pd.DataFrame([feature_dict])[FEATURE_COLUMNS].copy()
    # category_code is already the pre-encoded numeric value from the fixed
    # category map built in build_category_code_map -- use it directly
    # rather than re-deriving cat.codes on a single-row frame (which would
    # always collapse to code 0).
    X["category_code"] = feature_dict["category_code"]
    return float(model.predict(X)[0])


def run_what_if(baseline: dict, overrides: dict) -> dict:
    """LIVE call -- the one part of Module 5 that queries the Module 4 model
    dynamically instead of reading from the precomputed bundle. Everything
    else in this file is static and precomputed; this function is what the
    Streamlit app imports directly for the What-If tab.

    baseline: dict of FEATURE_COLUMNS values for one item (from
    load_baseline_row). overrides: dict of {feature_name: new_value} for
    any subset of WHAT_IF_INTERPRETABLE_FEATURES.
    """
    model = _load_model_for_live_predictions()
    baseline_pred = _predict_one(model, baseline)
    scenario = _apply_whatif_overrides(baseline, overrides)
    scenario_pred = _predict_one(model, scenario)

    change_kg = scenario_pred - baseline_pred
    change_pct = (change_kg / baseline_pred * 100) if baseline_pred else float("nan")

    return {
        "baseline_forecast_kg": round(baseline_pred, 2),
        "scenario_forecast_kg": round(scenario_pred, 2),
        "change_kg": round(change_kg, 2),
        "change_pct": round(change_pct, 1) if not np.isnan(change_pct) else None,
        "overrides_applied": overrides,
        "is_scenario_not_observed_forecast": True,
    }


@lru_cache(maxsize=1)
def _load_shap_explainer_for_live_use():
    """Cached TreeExplainer for live per-item explanations in the Explorer
    tab. Separate from local_shap_for_rows' batch explainer since this one
    needs to persist across many single-item calls without rebuilding."""
    model = _load_model_for_live_predictions()
    return shap.TreeExplainer(model)


def explain_forecast_live(baseline: dict) -> dict:
    """LIVE call (like run_what_if) -- computes local SHAP for one item's
    baseline feature vector on demand. Only 3 example items got this
    precomputed in Step 7; every other item in the Explorer tab needs this
    function called live, using the same cached model/explainer."""
    explainer = _load_shap_explainer_for_live_use()
    X_row = pd.DataFrame([baseline])[FEATURE_COLUMNS].copy()
    X_row["category_code"] = baseline["category_code"]
    sv = explainer.shap_values(X_row)[0]

    contribs = list(zip(FEATURE_COLUMNS, sv))
    contribs.sort(key=lambda t: -abs(t[1]))
    top_positive = [{"feature": f, "shap_value": float(v)} for f, v in contribs if v > 0][:3]
    top_negative = [{"feature": f, "shap_value": float(v)} for f, v in contribs if v < 0][:3]

    return {
        "top_positive_drivers": top_positive,
        "top_negative_drivers": top_negative,
        "plain_language_explanation": _plain_language_shap_description(top_positive, top_negative),
    }


def build_what_if_reference(data: dict) -> dict:
    print("\n=== STEP 8: What-If reference (business-interpretable input subset) ===")
    train_df = data["train_df"]
    feature_doc = []
    for f in FEATURE_COLUMNS:
        if f == "category_code":
            continue
        series = train_df[f].dropna()
        feature_doc.append({
            "feature": f,
            "plain_language": FEATURE_PLAIN_LANGUAGE.get(f, f),
            "business_interpretable_what_if_input": f in WHAT_IF_INTERPRETABLE_FEATURES,
            "p5": float(series.quantile(0.05)),
            "p50_default": float(series.quantile(0.50)),
            "p95": float(series.quantile(0.95)),
        })
    doc_df = pd.DataFrame(feature_doc)
    print(doc_df[doc_df["business_interpretable_what_if_input"]].to_string(index=False))
    return {"full_feature_documentation": doc_df}


# ============================================================================
# STEP 9: DASHBOARD / BUSINESS HIGHLIGHTS (KPI card content)
# ============================================================================

def build_highlights(forward: pd.DataFrame) -> dict:
    print("\n=== STEP 9: Dashboard highlights (KPI cards) ===")
    # Reuse the canonical forecast_status field (single source of truth for
    # currentness, also used in the Explorer tab) rather than recomputing a
    # separate recency rule here.
    current = forward[forward["forecast_status"] == "Current forecast"]
    print(f"  {len(current)}/{len(forward)} items have status 'Current forecast' -- "
          f"KPI highlights use this subset only, to avoid surfacing a 'top' item "
          f"whose last real sale was years earlier.")

    highest_demand = current.loc[current["forecast_demand_kg"].idxmax()]
    largest_increase = current.loc[current["change_vs_recent_demand_avg_pct"].idxmax()]
    largest_decrease = current.loc[current["change_vs_recent_demand_avg_pct"].idxmin()]

    highlights = {
        "highest_forecast_demand": {
            "display_label": highest_demand["display_label"],
            "value_kg": round(float(highest_demand["forecast_demand_kg"]), 1),
        },
        "largest_expected_increase": {
            "display_label": largest_increase["display_label"],
            "value_pct": round(float(largest_increase["change_vs_recent_demand_avg_pct"]), 1),
        },
        "largest_expected_decrease": {
            "display_label": largest_decrease["display_label"],
            "value_pct": round(float(largest_decrease["change_vs_recent_demand_avg_pct"]), 1),
        },
    }
    print(json.dumps(highlights, indent=2))
    return highlights


# ============================================================================
# STEP 10: HISTORICAL + FORECAST SERIES (for trend charts)
# ============================================================================

def build_trend_series(data: dict, forward: pd.DataFrame) -> pd.DataFrame:
    print("\n=== STEP 10: Historical + forecast series (for trend charts) ===")
    hist = data["item_day"][["item_code", "date", "daily_demand_kg"]].copy()
    hist["series_type"] = "historical"
    hist = hist.rename(columns={"date": "date", "daily_demand_kg": "demand_kg"})

    fwd_points = forward[["item_code", "forecast_date", "forecast_demand_kg",
                           "range_lower_kg", "range_upper_kg"]].copy()
    fwd_points = fwd_points.rename(columns={"forecast_date": "date", "forecast_demand_kg": "demand_kg"})
    fwd_points["series_type"] = "forecast"

    combined = pd.concat([hist, fwd_points], ignore_index=True, sort=False)
    print(f"  Combined series: {len(hist)} historical rows + {len(fwd_points)} forecast points "
          f"across {combined['item_code'].nunique()} items")
    return combined


# ============================================================================
# STEP 11: NARRATIVE CONTENT (drafted from the evidence above -- DRAFT for
# human review, not final copy; every number here is traceable to a CSV)
# ============================================================================

def build_narrative_content(data, repro, fairness, ranges, forward, highlights) -> dict:
    print("\n=== STEP 11: Narrative content (draft -- for review) ===")

    overall = compute_metrics(data["test_df"][TARGET_COLUMN], repro["y_pred_final"])
    top10 = compute_metrics(
        data["test_df"][data["test_df"]["item_code"].isin(top10_items_by_train_volume(data["train_df"]))][TARGET_COLUMN],
        pd.Series(repro["y_pred_final"], index=data["test_df"].index)[
            data["test_df"]["item_code"].isin(top10_items_by_train_volume(data["train_df"]))
        ],
    )

    gaps = fairness["tier_gaps"]
    mape_gap_after = gaps[(gaps["stage"] == "after_mitigation") & (gaps["metric"] == "mape_pct")]["gap_pp"].iloc[0]
    wape_gap_after = gaps[(gaps["stage"] == "after_mitigation") & (gaps["metric"] == "wape_pct")]["gap_pp"].iloc[0]
    low_vol_wape_change = fairness["mitigation_effect"].loc[
        fairness["mitigation_effect"]["volume_tier"] == "low_volume", "wape_change"
    ].iloc[0]

    narrative = {
        "labels": {
            "range_term": "95% Demand Range",
            "range_term_never_use": ["95% CI", "95% Confidence Interval", "95% Prediction Interval"],
            "recent_demand_term": "Recent demand (7-day average)",
            "previous_day_term": "Previous recorded day",
            "current_status_term": "Current forecast",
            "stale_status_term": "Older activity \u2014 treat with caution",
            "no_forecast_status_term": "No forecast available",
        },
        "plain_language_model_explanation": (
            "The forecast estimates how much of each vegetable will be needed for the next "
            "delivery day. It mainly looks at how much has been selling recently -- demand "
            "from the previous day and the average over the last 1-2 weeks -- along with this "
            "item's typical historical demand, the day of the week, and recent price."
        ),
        "plain_language_shap_explanation": (
            "Across all forecasts, recent demand patterns matter most: how much was sold in "
            "the last 7 days is the single biggest factor, followed by demand from the day "
            "before. The day of the week also has a noticeable effect, reflecting the store's "
            "Monday/Wednesday/Saturday replenishment cycle."
        ),
        "fairness_summary": (
            f"Forecast accuracy is not the same for every vegetable. Under the model's official "
            f"accuracy measure (MAPE), high-volume vegetables were forecast far more accurately "
            f"than low-volume ones -- a gap of {mape_gap_after:.1f} "
            f"percentage points. Part of this gap is a known mathematical property of this "
            f"accuracy measure, which becomes unstable when actual demand is very close to zero "
            f"(true for most low-volume item-days). Using a volume-weighted measure less "
            f"sensitive to this issue, the gap is smaller ({wape_gap_after:.1f} "
            f"points) but still real: low-volume vegetables are genuinely harder to forecast "
            f"because their demand is intermittent."
        ),
        "bias_mitigation_explanation": (
            "A mitigation step (reweighting low-volume observations during training) was applied "
            "to reduce this gap. It produced a small, measurable improvement -- roughly a "
            f"{abs(low_vol_wape_change):.1f} "
            "percentage-point reduction in low-volume error under a volume-weighted measure -- "
            "but did not resolve the underlying gap. Low-volume forecasting accuracy remains a "
            "genuine limitation of the current model."
        ),
        "model_limitations": [
            f"The original acceptance target (\u226420% MAPE for the top-10 highest-volume items) was not met: "
            f"the mitigated model achieves {top10['mape_pct']:.1f}% MAPE on that group.",
            "MAPE (the metric used for the acceptance target) becomes unstable when actual demand "
            "is close to zero, which affects a meaningful share of item-days -- a volume-weighted "
            f"measure (WAPE) shows {top10['wape_pct']:.1f}% for the same top-10 group, a more "
            "representative picture of typical-day performance.",
            "Forecast accuracy varies meaningfully by item volume; low-volume vegetables remain "
            "the hardest to forecast even after bias mitigation.",
            "The 95% demand range is a supplementary, empirically-calibrated visualization added "
            "in this module -- it is not a native output of the Module 4 XGBoost model, and its "
            "coverage was validated on the same hold-out period used to calibrate it, not on "
            "genuinely unseen future data.",
            "For many items, especially lower-volume ones, the 95% range is wide relative to the "
            "point forecast -- often wider than the forecast itself -- and should be read as a "
            "signal of real uncertainty, not a precise bound.",
            f"Forward forecasts are generated per item from each item's own most recent recorded "
            f"activity; {184} of 246 items had enough history to forecast, and for items with no "
            f"recent activity the 'next replenishment cycle' forecast may reflect a date well in "
            f"the past relative to the most recent data in the set.",
            "Lag and rolling-average features are based on each item's previous recorded sales "
            "entries, not strict calendar-day lags -- for items with gaps in their sales history, "
            "a '7-day average' may span more or fewer than 7 calendar days.",
            "Forecasts are decision support only. The model does not make procurement decisions "
            "and does not account for promotions, weather, local events, or supply disruptions.",
        ],
        "transparency_statement": (
            "This forecast is a decision-support estimate, not a guarantee of future demand. It "
            "is built from historical sales patterns and does not know about promotions, weather, "
            "local events, or supply issues. The Procurement Manager retains full responsibility "
            "for the final order decision and should apply operational judgment alongside this tool."
        ),
        "business_implications": (
            "For high-volume, frequently-sold vegetables, the model gives a reasonably useful "
            "starting estimate with a moderate range. For low-volume or intermittently-sold "
            "vegetables, forecasts should be treated as a rough guide only, and procurement "
            "decisions for those items should continue to rely heavily on manager judgment and "
            "local knowledge."
        ),
        "uncertainty_explanation": (
            "Alongside each forecast, a 95% Demand Range shows a band that historical forecast "
            "errors fell within, for items and forecast levels like this one, when checked against "
            "the same hold-out period used to build this range. A wider range means less certainty. "
            "This range is a visualization aid built on top of the model's historical errors -- it "
            "does not come from the forecasting model itself, and it has not been tested against "
            "genuinely new, unseen future data."
        ),
    }

    for key, val in narrative.items():
        if isinstance(val, str):
            preview = val[:100]
        elif isinstance(val, dict):
            preview = str(val)[:100]
        else:
            preview = str(val[0])[:100]
        print(f"  [{key}] {preview}...")

    return narrative


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Everything below is computed and printed to console for review/
    # crosscheck. Exactly ONE file is written to disk: the consolidated
    # bundle at the end. No intermediate CSVs/JSONs are saved -- this is
    # the deliberate scope boundary for Module 5's extra analysis layer.
    # ------------------------------------------------------------------
    data = load_module4_artifacts()
    repro = reproduce_module4(data)
    overall_top10 = overall_and_top10_metrics(data, repro)
    fairness = fairness_and_mitigation(data, repro)
    cat_metrics = category_metrics(data["test_df"], repro["y_pred_final"])
    print("\n=== Category-level performance (final model) ===")
    print(cat_metrics.to_string(index=False))

    baseline_comparison = build_baseline_comparison(data)
    print("\n=== Baseline comparison (seasonal-naive vs XGBoost) ===")
    print(baseline_comparison.to_string(index=False))

    lime_cross_check = data["explain_summary"].get("lime_cross_check_top_features", [])
    print(f"\n=== LIME cross-check (Module 4, single representative row) ===\n  {lime_cross_check}")

    ranges = build_prediction_ranges(data, repro, fairness)
    global_shap = load_global_shap(data)
    forward = build_all_forward_forecasts(data, data["model_final"], fairness["tier_map"], ranges["tier_bounds"])
    examples = build_examples_with_shap(data, forward)
    what_if_doc = build_what_if_reference(data)
    highlights = build_highlights(forward)
    trend_series = build_trend_series(data, forward)
    narrative = build_narrative_content(data, repro, fairness, ranges, forward, highlights)

    # ------------------------------------------------------------------
    # Live What-If smoke test -- proves the run_what_if() function (the one
    # part of this file that queries the model live, not from precomputed
    # data) actually works, in the same run, without a separate script.
    # ------------------------------------------------------------------
    print("\n=== What-If smoke test (live model call, not precomputed) ===")
    sample_item = forward[forward["forecast_status"] == "Current forecast"].iloc[0]["item_code"]
    baseline_row = load_baseline_row(forward, sample_item)
    wi_result = run_what_if(baseline_row, overrides={"lag_1": baseline_row["lag_1"] * 1.5})
    print(f"  item_code={sample_item}: baseline={wi_result['baseline_forecast_kg']}kg -> "
          f"scenario (lag_1 x1.5)={wi_result['scenario_forecast_kg']}kg "
          f"({wi_result['change_pct']:+}%)")
    try:
        run_what_if(baseline_row, overrides={"item_expanding_mean_demand": 999})
        print("  WARNING: guardrail failed to reject a non-interpretable override")
    except ValueError:
        print("  Guardrail correctly rejected a non-interpretable override")

    print("\n=== Live per-item SHAP smoke test (Explorer tab dependency) ===")
    live_explain = explain_forecast_live(baseline_row)
    print(f"  item_code={sample_item}: {live_explain['plain_language_explanation']}")

    # ------------------------------------------------------------------
    # Summary block, printed only (not saved separately -- it's folded
    # into the bundle below under bundle["summary"]).
    # ------------------------------------------------------------------
    summary = {
        "reproduction_checks_passed": repro["reproduction_checks"],
        "module4_official_metrics": data["results_summary"]["xgboost"]["test"],
        "module4_official_fairness_gap_pp": data["mitigation_summary"]["gap_before_pp"],
        "module4_official_fairness_gap_after_mitigation_pp": data["mitigation_summary"]["gap_after_pp"],
        "overall_metrics_final_model": overall_top10.iloc[0].to_dict(),
        "top10_metrics_final_model": overall_top10.iloc[1].to_dict(),
        "fairness_gaps_by_metric": fairness["tier_gaps"].to_dict(orient="records"),
        "mitigation_effect_by_tier": fairness["mitigation_effect"].to_dict(orient="records"),
        "range_coverage_by_tier": ranges["coverage"].to_dict(orient="records"),
        "n_items_total": len(forward),
        "n_items_current_forecast": int((forward["forecast_status"] == "Current forecast").sum()),
        "n_items_older_activity": int(forward["forecast_status"].str.startswith("Older", na=False).sum()),
        "n_items_no_forecast": int((forward["forecast_status"] == "No forecast available").sum()),
        "dashboard_highlights": highlights,
        "known_caveats_for_human_review": [
            "Fairness gap reduction from mitigation is modest under robust metrics (WAPE) despite "
            "looking large under MAPE -- see bundle['mitigation_comparison'] before writing the "
            "deck's bias-mitigation slide.",
            "95% Demand Range coverage is validated in-sample (same hold-out set used to calibrate "
            "it) -- state this explicitly, do not present as validated generalization.",
            "Range width is very large relative to forecast size (median ~200-270% of forecast "
            "across all tiers) -- this is a genuine finding, not a bug; do not hide it.",
            "62 of 246 items have forecast_status='No forecast available' (insufficient history) -- "
            "still listed in bundle['forward_forecasts'] so the item selector can show them honestly.",
            "Some items' forecast_status is 'Older activity' -- their last recorded sale may be "
            "years before the dataset's overall end date. Explorer tab must show forecast_status "
            "and last_known_date plainly, never just the point forecast.",
            "No vegetable names exist in the source data -- display_label uses category + item "
            "code; this is disclosed, not invented.",
            "Never label the 95% Demand Range as a confidence interval or statistical prediction "
            "interval anywhere in the app -- see bundle['narrative_content']['labels'].",
        ],
    }

    # ------------------------------------------------------------------
    # The ONE file this script writes. Python's analog to an RDS file:
    # one joblib-serialized object holding every dataframe/dict this
    # evidence layer produced, PLUS the live What-If function is available
    # by importing run_what_if()/load_baseline_row() from this same file --
    # no second script, no separate outputs.
    # ------------------------------------------------------------------
    bundle = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "reproduction_checks": repro["reproduction_checks"],
        "module4_official": {
            "metrics": data["results_summary"]["xgboost"]["test"],
            "fairness_gap_before_pp": data["mitigation_summary"]["gap_before_pp"],
            "fairness_gap_after_pp": data["mitigation_summary"]["gap_after_pp"],
        },
        "overall_top10_metrics": overall_top10,
        "volume_tier_metrics": fairness["tier_metrics"],
        "category_metrics": cat_metrics,
        "baseline_comparison": baseline_comparison,
        "lime_cross_check_top_features": lime_cross_check,
        "fairness_gaps": fairness["tier_gaps"],
        "mitigation_comparison": fairness["mitigation_effect"],
        "near_zero_breakdown": fairness["near_zero"],
        "range_coverage": ranges["coverage"],
        "range_tier_bounds": ranges["tier_bounds"],
        "global_shap": global_shap,
        "forward_forecasts": forward,
        "example_forecasts": examples,
        "what_if_full_feature_documentation": what_if_doc["full_feature_documentation"],
        "what_if_interpretable_features": WHAT_IF_INTERPRETABLE_FEATURES,
        "dashboard_highlights": highlights,
        "trend_series": trend_series,
        "narrative_content": narrative,
        "summary": summary,
    }

    OUTPUT_PATH = OUTPUT_DIR / "evidence_bundle.pkl"
    joblib.dump(bundle, OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("MODULE 5 EVIDENCE LAYER COMPLETE -- ONE FILE WRITTEN")
    print("=" * 70)
    print(f"  {OUTPUT_PATH}")
    print("\nFrom the Streamlit app (run from the repo root), import like:")
    print("  from src.evaluation.evidence import run_what_if, load_baseline_row  # live calls")
    print("  bundle = joblib.load('src/evaluation/evidence_bundle.pkl')  # everything else")