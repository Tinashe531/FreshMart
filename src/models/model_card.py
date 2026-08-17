"""
Model Card generator (Module 4 deliverable).

Pulls numbers directly from the JSON artifacts saved by:
    train.py
    evaluate.py
    explain.py
    mitigate.py

Run:
    python -m src.models.model_card

Output:
    src/models/artifacts/MODEL_CARD.md
"""

import sys
import json
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "src" / "models" / "artifacts"
EXPLAIN_DIR = ARTIFACTS_DIR / "explainability"


def _load(path):
    with open(path, "r") as f:
        return json.load(f)


def build_model_card() -> str:

    # ------------------------------------------------------------------
    # Load existing artifacts
    # ------------------------------------------------------------------
    results = _load(ARTIFACTS_DIR / "results_summary.json")
    evaluation = _load(ARTIFACTS_DIR / "evaluation_summary.json")
    mitigation = _load(EXPLAIN_DIR / "mitigation_summary.json")
    explainability = _load(EXPLAIN_DIR / "explainability_summary.json")

    # ------------------------------------------------------------------
    # Final model information
    # ------------------------------------------------------------------
    final_choice = mitigation["final_model_choice"]

    before = mitigation["overall_before"]
    after = mitigation["overall_after"]

    top10_before = mitigation["top10_mape_before"]
    top10_after = mitigation["top10_mape_after"]

    # ------------------------------------------------------------------
    # Filter the evaluation artifact to XGBoost diagnostics
    #
    # These results were generated BEFORE fairness mitigation.
    # ------------------------------------------------------------------
    xgb_category_rows = [
        row
        for row in evaluation["category_breakdown"]
        if row["model"] == "xgboost"
    ]

    xgb_volume_rows = [
        row
        for row in evaluation["volume_tier_breakdown"]
        if row["model"] == "xgboost"
    ]

    cat_rows = "\n".join(
        f"| {c['category_name']} | {c['n']} | "
        f"{c['mape_pct']:.1f}% | {c['rmse']:.2f} |"
        for c in xgb_category_rows
    )

    vol_rows = "\n".join(
        f"| {v['volume_tier']} | {v['n']} | "
        f"{v['mape_pct']:.1f}% | {v['rmse']:.2f} |"
        for v in xgb_volume_rows
    )

    # ------------------------------------------------------------------
    # SHAP global importance
    # ------------------------------------------------------------------
    top_features = explainability["shap_top_features"][:5]

    top_features_str = "\n".join(
        f"- **{f['feature']}** "
        f"(mean |SHAP| = {f['mean_abs_shap']:.3f})"
        for f in top_features
    )

    # ------------------------------------------------------------------
    # Model Card
    # ------------------------------------------------------------------
    card = f"""# Model Card — FreshMart Demand Forecasting

**Date generated:** {date.today().isoformat()}

**Model type:** XGBoost Regressor

**Final model status:** {final_choice}

**Task:** One-day-ahead item-level demand forecasting (regression)

**Target variable:** `daily_demand_kg`

**Unit of analysis:** One item-day (`item_code × date`)

**Final model artifact:** `xgboost_final.joblib`

---

## Intended Use

This model produces a **decision-support forecast** for FreshMart's
procurement planning, aligned with the Monday/Wednesday/Saturday
replenishment cycle.

It is **not an automated ordering system**. The Procurement Manager retains
final responsibility for order quantities and may override model outputs using
operational information not captured by the model, such as promotions,
weather, local events or supply disruptions.

**Out-of-scope uses:** automated or unattended ordering decisions; deployment
to stores or item ranges not represented in the training data; and use as a
food-safety or spoilage-prediction model.

---

## Training Data

- Source: Module 3 validated item-day pipeline
  (`freshmart_item_day.parquet`)
- 46,599 item-day observations
- 246 items
- 6 vegetable categories
- July 2020 – June 2023
- Chronological hold-out with the final 90 days retained as an untouched
  test period
- 3-fold expanding-window rolling-origin cross-validation for model tuning
- Historical demand and price features use `.shift(1)` or later
- Rows without sufficient historical information were excluded rather than
  imputed

---

## Performance

### Hold-out Test Performance

| Metric | Seasonal-naive baseline | XGBoost before mitigation | Final XGBoost after mitigation |
|---|---:|---:|---:|
| RMSE (kg) | {results['baseline']['rmse']:.3f} | {before['rmse']:.3f} | {after['rmse']:.3f} |
| MAE (kg) | {results['baseline']['mae']:.3f} | {before['mae']:.3f} | {after['mae']:.3f} |
| MAPE | {results['baseline']['mape_pct']:.1f}% | {before['mape_pct']:.1f}% | {after['mape_pct']:.1f}% |
| R² | — | {before['r2']:.3f} | {after['r2']:.3f} |
| Forecast bias (kg) | {results['baseline']['forecast_bias']:.3f} | {before['bias']:.3f} | {after['bias']:.3f} |

### Acceptance Criterion

The predefined Module 1 acceptance criterion was:

**MAPE ≤20% on the ten highest-volume items.**

The results were:

- Before mitigation: **{top10_before:.2f}%**
- After mitigation: **{top10_after:.2f}%**
- Criterion: **Not met**

The criterion was retained unchanged after observing the results.

---

## Diagnostic Performance by Category

The following results are the **pre-mitigation XGBoost diagnostic
breakdown** generated during the evaluation stage. They identify where
performance differed across FreshMart's vegetable categories and informed
the subsequent fairness analysis.

| Category | n | MAPE | RMSE |
|---|---:|---:|---:|
{cat_rows}

---

## Diagnostic Performance by Volume Tier

The following results are the **pre-mitigation XGBoost volume-tier
breakdown**.

| Volume tier | n | MAPE | RMSE |
|---|---:|---:|---:|
{vol_rows}

---

## Fairness Analysis and Mitigation

No demographic protected attributes are present in the FreshMart dataset.
Fairness was therefore assessed through **forecast-error parity across
vegetable categories and demand-volume tiers**.

The original XGBoost evaluation identified a:

**{mitigation['gap_before_pp']:.1f}-percentage-point volume-tier MAPE gap**

between the best- and worst-performing volume tiers. This exceeded the
25-percentage-point investigation threshold.

The Module 1 Ethical Risk Register had already identified the risk of
**category bias resulting in under-forecasting of lower-volume vegetables**.

### Mitigation

Inverse-volume sample weighting was applied during XGBoost training. Lower
volume items received greater training weight so that their errors had greater
influence on the squared-error objective.

### Before versus After Mitigation

- Volume-tier MAPE gap:
  **{mitigation['gap_before_pp']:.1f}pp → {mitigation['gap_after_pp']:.1f}pp**
- Overall RMSE:
  **{before['rmse']:.3f} → {after['rmse']:.3f} kg**
- Overall MAE:
  **{before['mae']:.3f} → {after['mae']:.3f} kg**
- Overall MAPE:
  **{before['mape_pct']:.2f}% → {after['mape_pct']:.2f}%**
- Top-10 MAPE:
  **{top10_before:.2f}% → {top10_after:.2f}%**

The volume-tier disparity was reduced but not eliminated. The mitigated model
was therefore retained as the final model.

---

## Explainability

Global SHAP analysis identified the following as the five strongest
contributors to model predictions:

{top_features_str}

Local SHAP explanations, a LIME cross-check, counterfactual analysis and
sensitivity analysis were also performed.

The results indicate that recent demand history, particularly `roll_mean_7`
and `lag_1`, is substantially more influential than most price and category
variables.

Full explainability outputs are stored under:

`src/models/artifacts/explainability/`

---

## Limitations and Risks

- The top-10-volume-item MAPE criterion of ≤20% was not achieved.
- Low-volume items remain the least reliable segment despite mitigation.
- The model was trained on one flagship store and has not been validated for
  generalization to other FreshMart stores.
- Promotions, weather, local events and supply disruptions are not directly
  represented in the current feature set.
- Model drift may occur as demand patterns change.
- The model should support, rather than replace, procurement-manager judgment.

---

## Monitoring Plan

The deployed model should be monitored using:

- RMSE
- MAE
- MAPE
- Forecast bias
- Category-level error
- Volume-tier error
- Top-10-volume-item MAPE
- Input feature distribution drift, particularly `roll_mean_7` and `lag_1`

Material deterioration in predictive performance or widening of the
volume-tier error gap should trigger investigation and model review before
the next pilot phase.

---

## Version and Reproducibility

**Final model artifact:** `xgboost_final.joblib`

**Experiment tracking:** MLflow

**MLflow experiment:** `freshmart_demand_forecasting`

**Tracking database:** `mlflow.db`

**Serialization format:** Joblib

**Training and evaluation artifacts:**

`src/models/artifacts/`

"""

    return card


def run_all():

    card = build_model_card()

    out_path = ARTIFACTS_DIR / "MODEL_CARD.md"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(card)

    print(f"Model Card written to {out_path}")

    print(f"\n{'-' * 60}")
    print("Preview:")
    print(f"{'-' * 60}")
    print(card[:2000])


if __name__ == "__main__":
    run_all()