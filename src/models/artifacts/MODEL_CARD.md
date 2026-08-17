# Model Card — FreshMart Demand Forecasting

**Date generated:** 2026-08-17

**Model type:** XGBoost Regressor

**Final model status:** mitigated

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
| RMSE (kg) | 6.787 | 5.021 | 4.964 |
| MAE (kg) | 4.245 | 3.139 | 3.108 |
| MAPE | 84.1% | 71.5% | 69.7% |
| R² | — | 0.756 | 0.761 |
| Forecast bias (kg) | 0.262 | 0.139 | 0.090 |

### Acceptance Criterion

The predefined Module 1 acceptance criterion was:

**MAPE ≤20% on the ten highest-volume items.**

The results were:

- Before mitigation: **80.58%**
- After mitigation: **77.64%**
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
| Aquatic Tuberous Vegetables | 425 | 77.1% | 2.17 |
| Cabbage | 171 | 61.2% | 4.39 |
| Capsicum | 804 | 66.9% | 5.04 |
| Edible Mushroom | 642 | 50.2% | 4.83 |
| Flower/Leaf Vegetables | 1225 | 85.2% | 5.99 |
| Solanum | 242 | 70.4% | 3.97 |

---

## Diagnostic Performance by Volume Tier

The following results are the **pre-mitigation XGBoost volume-tier
breakdown**.

| Volume tier | n | MAPE | RMSE |
|---|---:|---:|---:|
| high_volume | 2001 | 66.0% | 6.28 |
| low_volume | 483 | 102.6% | 1.26 |
| mid_volume | 991 | 66.2% | 2.63 |

---

## Fairness Analysis and Mitigation

No demographic protected attributes are present in the FreshMart dataset.
Fairness was therefore assessed through **forecast-error parity across
vegetable categories and demand-volume tiers**.

The original XGBoost evaluation identified a:

**36.7-percentage-point volume-tier MAPE gap**

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
  **36.7pp → 28.6pp**
- Overall RMSE:
  **5.021 → 4.964 kg**
- Overall MAE:
  **3.139 → 3.108 kg**
- Overall MAPE:
  **71.45% → 69.73%**
- Top-10 MAPE:
  **80.58% → 77.64%**

The volume-tier disparity was reduced but not eliminated. The mitigated model
was therefore retained as the final model.

---

## Explainability

Global SHAP analysis identified the following as the five strongest
contributors to model predictions:

- **roll_mean_7** (mean |SHAP| = 4.113)
- **lag_1** (mean |SHAP| = 2.484)
- **day_of_week** (mean |SHAP| = 1.180)
- **roll_mean_14** (mean |SHAP| = 0.737)
- **item_expanding_mean_demand** (mean |SHAP| = 0.453)

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

