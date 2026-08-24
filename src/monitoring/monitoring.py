"""
FreshMart Monitoring Dashboard -- Task 7 evidence layer.
==========================================================

CRITICAL FRAMING -- read before touching anything below:

This project has not yet accumulated real post-deployment production data
(the Streamlit app went live days before this file was written). Every
number in this file is a BACKTESTED SIMULATION built from already-known
historical data, reframed as a monitoring timeline. It is NOT live
production monitoring. MONITORING_DISCLAIMER below must be surfaced
verbatim wherever any of this module's results are displayed.

Two different reference concepts are used here, deliberately named
differently so they are never conflated in the app or the report:

  "baseline"            -- the already-established Module 4/5 test-period
                            evaluation metrics (category_metrics /
                            volume_tier_metrics, already in the Module 5
                            bundle). Reused directly, never recomputed --
                            per Task 7's explicit instruction not to invent
                            new baseline values.

  "reference distribution" -- the TRAINING period's feature distributions.
                            Drift is conventionally measured against what
                            the model was trained on, which is a different
                            question from "is current performance worse
                            than the established baseline." Using a
                            different term for each avoids the exact
                            terminology confusion flagged before writing
                            any of this.

  "monitoring period"   -- the real test-period dates (Apr-Jun 2023),
                            split into 3 real calendar months, each
                            treated as a successive monitoring window
                            arriving after deployment. Not invented data --
                            real historical outcomes, re-sequenced.

Grain: category and volume tier ONLY, never individual items -- per
Module 4/5's own finding that item-level series are too sparse/intermittent
for stable statistics.

Design mirrors src/evaluation/evidence.py: reuse what already exists
(load_module4_artifacts, compute_metrics), compute only what's genuinely
missing, print everything to console for review, save ONE consolidated
file. No new dependencies -- PSI is implemented by hand (numpy/pandas
only), not pulled in from an extra drift-detection library.

Run (from the repo root):
    python src/monitoring/monitoring.py
"""

import sys
import json
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent            # .../src/monitoring
PROJECT_ROOT = SCRIPT_DIR.parent.parent                  # repo root
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evidence import (
    load_module4_artifacts, compute_metrics, _prep_X, assign_volume_tiers,
    ARTIFACTS_DIR, TARGET_COLUMN, FEATURE_COLUMNS,
)

OUTPUT_PATH = SCRIPT_DIR / "monitoring_bundle.pkl"

MONITORING_DISCLAIMER = (
    "Backtested monitoring simulation. This project has not yet accumulated "
    "real post-deployment production data. The figures below are computed "
    "from already-known historical outcomes (the model's test period, "
    "April-June 2023), re-sequenced as three successive monthly monitoring "
    "windows to demonstrate how the dashboard would behave once real "
    "production data accumulates. This is not live production monitoring."
)

# Drift is checked on these 3 features only -- the top global SHAP drivers
# from Module 4/5, a defensible, documented selection rather than checking
# all 17 features (which would also fragment sample sizes further at
# category grain).
DRIFT_FEATURES = ["roll_mean_7", "lag_1", "price_lag_1"]

# ----------------------------------------------------------------------
# THRESHOLDS -- documented here, once, and surfaced in the bundle so the
# app and the report can display the exact same numbers. Not buried in
# scattered magic numbers through the code.
# ----------------------------------------------------------------------
THRESHOLDS = {
    "performance": {
        "metric": "WAPE, relative change vs. established baseline",
        "normal_max_pct": 10.0,
        "warning_max_pct": 20.0,
        "note": "Above 20% relative deterioration in WAPE vs. the established "
                "test-period baseline is Critical; 10-20% is Warning; below "
                "10% is Normal. Proposed in the Task 7 brief; not re-derived "
                "from scratch, but sanity-checked against real category-level "
                "figures below before being treated as final.",
    },
    "drift": {
        "metric": "Population Stability Index (PSI), monitoring month vs. training-period reference distribution",
        "normal_max": 0.10,
        "warning_max": 0.25,
        "note": "PSI < 0.10 = no significant distribution shift (Normal). "
                "0.10-0.25 = moderate shift (Warning). > 0.25 = significant "
                "shift (Critical). This is a standard, widely-cited PSI "
                "banding convention, not an arbitrary threshold invented for "
                "this project.",
    },
    "fairness": {
        "metric": "High-vs-low volume tier WAPE gap, percentage points above the established baseline gap (19.4pp, Module 4/5)",
        "normal_max_pp": 5.0,
        "warning_max_pp": 10.0,
        "note": "Gap within 5pp of the established baseline gap is Normal. "
                "5-10pp wider is Warning. More than 10pp wider than the "
                "established gap is Critical.",
    },
}


def status_from_thresholds(value, normal_max, warning_max) -> str:
    if value <= normal_max:
        return "Normal"
    elif value <= warning_max:
        return "Warning"
    return "Critical"


def population_stability_index(reference: np.ndarray, monitored: np.ndarray, bins: int = 10) -> float:
    """Standard PSI, computed by hand -- no new dependency for this.
    Bin edges are defined on the REFERENCE (training) distribution, then
    both distributions are binned against those same fixed edges."""
    reference = reference[~np.isnan(reference)]
    monitored = monitored[~np.isnan(monitored)]
    if len(reference) < 10 or len(monitored) < 10:
        return np.nan  # not enough data for a meaningful PSI at this grain

    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)  # guard against degenerate (constant) features
    if len(edges) < 3:
        return np.nan

    ref_counts, _ = np.histogram(reference, bins=edges)
    mon_counts, _ = np.histogram(monitored, bins=edges)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-4, None)
    mon_pct = np.clip(mon_counts / max(mon_counts.sum(), 1), 1e-4, None)

    psi = np.sum((mon_pct - ref_pct) * np.log(mon_pct / ref_pct))
    return float(psi)


print("Monitoring module -- Part 1 loaded (setup, disclaimer, thresholds, PSI function).")


# ============================================================================
# STEP 1: MONITORING-PERIOD CONSTRUCTION
# Real calendar months within the real test period. Not invented dates.
# ============================================================================

def build_monitoring_periods(data: dict) -> dict:
    print("\n=== STEP 1: Monitoring-period construction ===")
    test_df = data["test_df"].copy()
    train_df = data["train_df"].copy()
    test_df["date"] = pd.to_datetime(test_df["date"])
    test_df["monitoring_month"] = test_df["date"].dt.to_period("M").astype(str)

    months = sorted(test_df["monitoring_month"].unique())
    print(f"  Monitoring months (real calendar months within the test period): {months}")
    for m in months:
        n = (test_df["monitoring_month"] == m).sum()
        print(f"    {m}: {n} rows")

    X_test = _prep_X(test_df)
    model = data["model_final"]
    test_df["y_pred"] = model.predict(X_test)

    tier_map = assign_volume_tiers(train_df)
    test_df["volume_tier"] = test_df["item_code"].map(tier_map)
    train_df["volume_tier"] = train_df["item_code"].map(tier_map)

    return {"test_df": test_df, "train_df": train_df, "months": months, "tier_map": tier_map}


# ============================================================================
# STEP 2: PERFORMANCE MONITORING
# Baseline = already-established Module 5 category_metrics/volume_tier_metrics
# (reused, NOT recomputed). Each monitoring month compared against it.
# ============================================================================

def performance_monitoring(periods: dict, baseline_bundle: dict) -> dict:
    print("\n=== STEP 2: Performance monitoring (vs. established baseline) ===")
    test_df, months = periods["test_df"], periods["months"]

    baseline_cat = baseline_bundle["category_metrics"].set_index("category_name")["wape_pct"].to_dict()
    baseline_tier = baseline_bundle["volume_tier_metrics"]
    baseline_tier_after = baseline_tier[baseline_tier["model_stage"] == "after_mitigation"].set_index("volume_tier")["wape_pct"].to_dict()

    th = THRESHOLDS["performance"]

    cat_rows = []
    for month in months:
        sub = test_df[test_df["monitoring_month"] == month]
        for cat, g in sub.groupby("category_name"):
            m = compute_metrics(g[TARGET_COLUMN], g["y_pred"])
            base = baseline_cat.get(cat, np.nan)
            rel_change = (m["wape_pct"] - base) / base * 100 if base else np.nan
            status = status_from_thresholds(rel_change, th["normal_max_pct"], th["warning_max_pct"]) if not np.isnan(rel_change) else "Insufficient data"
            cat_rows.append({
                "month": month, "category_name": cat, "n": m["n"],
                "wape_pct": m["wape_pct"], "mae": m["mae"], "rmse": m["rmse"],
                "baseline_wape_pct": base, "relative_change_pct": rel_change, "status": status,
            })
    cat_df = pd.DataFrame(cat_rows)

    tier_rows = []
    for month in months:
        sub = test_df[test_df["monitoring_month"] == month]
        for tier, g in sub.groupby("volume_tier", observed=True):
            m = compute_metrics(g[TARGET_COLUMN], g["y_pred"])
            base = baseline_tier_after.get(tier, np.nan)
            rel_change = (m["wape_pct"] - base) / base * 100 if base else np.nan
            status = status_from_thresholds(rel_change, th["normal_max_pct"], th["warning_max_pct"]) if not np.isnan(rel_change) else "Insufficient data"
            tier_rows.append({
                "month": month, "volume_tier": tier, "n": m["n"],
                "wape_pct": m["wape_pct"], "mae": m["mae"], "rmse": m["rmse"],
                "baseline_wape_pct": base, "relative_change_pct": rel_change, "status": status,
            })
    tier_df = pd.DataFrame(tier_rows)

    print(tier_df.to_string(index=False))
    print()
    print("  Status counts (category-level, across all 3 months):")
    print(cat_df["status"].value_counts().to_string())

    return {"by_category": cat_df, "by_tier": tier_df}


# ============================================================================
# STEP 3: DRIFT MONITORING
# Reference = TRAINING period distribution (what the model learned from).
# Deliberately NOT the same concept as the performance "baseline" above.
# ============================================================================

def drift_monitoring(periods: dict) -> pd.DataFrame:
    print("\n=== STEP 3: Drift monitoring (vs. training-period reference distribution) ===")
    print("  NOTE: drift is computed at VOLUME TIER grain only, not category. An earlier "
          "category-level version produced implausibly extreme PSI values -- diagnosed as an "
          "item-composition artifact (categories pool items of very different typical demand "
          "scale; different items are active in different months, which alone creates large "
          "apparent distribution shifts unrelated to genuine drift). Volume tiers are "
          "constructed to be scale-homogeneous by design, which avoids this confound.")
    train_df, test_df, months = periods["train_df"], periods["test_df"], periods["months"]
    th = THRESHOLDS["drift"]

    rows = []
    for feature in DRIFT_FEATURES:
        for tier in ["low_volume", "mid_volume", "high_volume"]:
            ref = train_df.loc[train_df["volume_tier"] == tier, feature].to_numpy()
            for month in months:
                mon = test_df.loc[(test_df["volume_tier"] == tier) & (test_df["monitoring_month"] == month), feature].to_numpy()
                psi = population_stability_index(ref, mon)
                if np.isnan(psi):
                    status = "Insufficient data"
                else:
                    status = status_from_thresholds(psi, th["normal_max"], th["warning_max"])
                rows.append({
                    "feature": feature, "volume_tier": tier, "month": month,
                    "n_reference": len(ref), "n_monitoring": len(mon),
                    "psi": psi, "status": status,
                })

    drift_df = pd.DataFrame(rows)
    print(f"\n  Computed PSI for {len(DRIFT_FEATURES)} features x 3 tiers x {len(months)} months = {len(drift_df)} rows")
    print(drift_df.to_string(index=False))
    print("\n  Status counts:")
    print(drift_df["status"].value_counts().to_string())
    return drift_df


# ============================================================================
# STEP 4: FAIRNESS MONITORING
# Tracks the high-vs-low volume tier WAPE gap per month against the
# established baseline gap (19.4pp, from Module 4/5's own mitigation work).
# ============================================================================

def fairness_monitoring(perf: dict) -> pd.DataFrame:
    print("\n=== STEP 4: Fairness monitoring (volume-tier WAPE gap vs. established baseline gap) ===")
    th = THRESHOLDS["fairness"]
    baseline_gap_pp = 19.4  # established in Module 4/5 (WAPE gap, after mitigation) -- reused, not recomputed

    rows = []
    for month, g in perf["by_tier"].groupby("month"):
        gap = g["wape_pct"].max() - g["wape_pct"].min()
        change_pp = gap - baseline_gap_pp
        status = status_from_thresholds(change_pp, th["normal_max_pp"], th["warning_max_pp"])
        worst_tier = g.loc[g["wape_pct"].idxmax(), "volume_tier"]
        rows.append({
            "month": month, "tier_wape_gap_pp": gap, "baseline_gap_pp": baseline_gap_pp,
            "change_vs_baseline_pp": change_pp, "worst_performing_tier": worst_tier, "status": status,
        })
    fair_df = pd.DataFrame(rows)
    print(fair_df.to_string(index=False))
    return fair_df


# ============================================================================
# STEP 5: INCIDENT RESPONSE & DECOMMISSIONING -- documentation, not code.
# Matches the tone already used in the Module 5 dashboard's Model
# Limitations / Transparency sections: descriptive status, not the
# dashboard telling anyone what to do.
# ============================================================================

INCIDENT_RESPONSE = {
    "Normal": {
        "meaning": "Performance, drift, and fairness are all within established thresholds.",
        "action": "Continue routine monitoring. No action required.",
    },
    "Warning": {
        "meaning": "A metric has moved outside the Normal band but not into Critical.",
        "action": "Investigate the specific category/tier and feature flagged. Check for a "
                   "plausible cause (seasonal shift, data quality issue, genuine demand change) "
                   "before concluding the model itself has degraded. Document findings.",
    },
    "Critical": {
        "meaning": "A metric has moved beyond the Warning threshold.",
        "action": "Investigate immediately. Bring findings to the Data & AI Governance Committee "
                   "(Module 1) rather than acting unilaterally. Consider corrective action: "
                   "recalibration, retraining, or reverting to a previous validated model. "
                   "The model remains decision support throughout -- the Procurement Manager's "
                   "judgement is not suspended while an incident is investigated.",
    },
}

DECOMMISSIONING_PLAN = {
    "triggers": [
        "Sustained Critical performance status across multiple consecutive monitoring periods.",
        "Persistent, unresolved category-level fairness gap beyond the established threshold.",
        "Significant, unresolved drift that corrective action does not address.",
        "Replacement by a better-validated model (see Task 7 brief's model-improvement principle: "
        "a new model must demonstrate meaningful improvement without degrading fairness or "
        "data-quality characteristics before it replaces the current one).",
        "Material change in business requirements (e.g., store expansion, new product categories) "
        "that the current model's scope does not cover.",
    ],
    "process": [
        "Identify the retirement condition and document the evidence.",
        "Governance review (Data & AI Governance Committee) -- decision is not made unilaterally.",
        "Stop production use of the model.",
        "Preserve required audit records: model artifacts, evaluation history, override logs.",
        "Document the decommissioning decision and rationale.",
        "Archive the model; transition to its replacement or to a manual/human decision process.",
    ],
}


if __name__ == "__main__":
    data = load_module4_artifacts()
    bundle_path = PROJECT_ROOT / "src" / "evaluation" / "evidence_bundle.pkl"
    if not bundle_path.exists():
        bundle_path = PROJECT_ROOT / "src" / "evaluation" / "module5_evidence_bundle.pkl"
    bundle = joblib.load(bundle_path)

    periods = build_monitoring_periods(data)
    perf = performance_monitoring(periods, bundle)
    drift = drift_monitoring(periods)
    fairness = fairness_monitoring(perf)

    # ------------------------------------------------------------------
    # Overall status rollup -- one number for a top-level KPI card.
    # Worst status across all three monitoring dimensions, most recent month.
    # ------------------------------------------------------------------
    latest_month = periods["months"][-1]
    status_rank = {"Normal": 0, "Warning": 1, "Critical": 2, "Insufficient data": 0}

    latest_perf_status = perf["by_tier"][perf["by_tier"]["month"] == latest_month]["status"]
    latest_drift_status = drift[drift["month"] == latest_month]["status"]
    latest_fair_status = fairness[fairness["month"] == latest_month]["status"]

    worst_rank = max(
        latest_perf_status.map(status_rank).max(),
        latest_drift_status.map(status_rank).max(),
        latest_fair_status.map(status_rank).max(),
    )
    overall_status = {v: k for k, v in status_rank.items() if v == worst_rank and k != "Insufficient data"}
    overall_status = ["Normal", "Warning", "Critical"][worst_rank]

    print(f"\n=== OVERALL STATUS (most recent monitoring month, {latest_month}): {overall_status} ===")

    narrative = {
        "disclaimer": MONITORING_DISCLAIMER,
        "performance_explanation": (
            "Each monitoring month's WAPE is compared against the model's established "
            "test-period baseline (the same evaluation already reported in Module 4/5). "
            "A relative increase of more than 20% is Critical; 10-20% is Warning."
        ),
        "drift_explanation": (
            "Drift compares each monitoring month's feature distribution (roll_mean_7, lag_1, "
            "price_lag_1 -- the model's top drivers) against the training-period distribution, "
            "using the Population Stability Index (PSI), at volume-tier grain. Category-level "
            "drift was tested and discarded: it produced misleadingly extreme values driven by "
            "which items happened to be active that month, not genuine behavioural change."
        ),
        "fairness_explanation": (
            "Tracks whether the gap between the best- and worst-performing volume tiers is "
            "widening beyond the already-established baseline gap (19.4 percentage points, "
            "from Module 4/5's fairness work) -- not whether the gap exists at all, since we "
            "already know it does and that mitigation only partly closed it."
        ),
        "overall_status_explanation": (
            f"Reflects the worst status across performance, drift, and fairness for the most "
            f"recent monitoring month ({latest_month}). A single Warning or Critical anywhere "
            f"drives the overall status, even if other areas are Normal."
        ),
    }

    monitoring_bundle = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "disclaimer": MONITORING_DISCLAIMER,
        "reference_period": {"start": str(periods["train_df"]["date"].min()), "end": str(periods["train_df"]["date"].max())},
        "monitoring_period": {"months": periods["months"]},
        "thresholds": THRESHOLDS,
        "performance_by_category": perf["by_category"],
        "performance_by_tier": perf["by_tier"],
        "drift": drift,
        "fairness": fairness,
        "overall_status": overall_status,
        "latest_month": latest_month,
        "incident_response": INCIDENT_RESPONSE,
        "decommissioning_plan": DECOMMISSIONING_PLAN,
        "narrative": narrative,
    }

    joblib.dump(monitoring_bundle, OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("MONITORING MODULE COMPLETE -- ONE FILE WRITTEN")
    print("=" * 70)
    print(f"  {OUTPUT_PATH}")
    print(f"  Overall status ({latest_month}): {overall_status}")