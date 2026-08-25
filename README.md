# FreshMart Demand Forecasting System

A decision-support analytics project for vegetable procurement planning at FreshMart's flagship store, from problem definition through a deployed, monitored forecasting model.

**Live Dashboard (includes Monitoring & Governance):** https://freshmart-vz2kmxfpq6cwmrkfn5ahix.streamlit.app/
**Live API:** [https://freshmart-9xou.onrender.com/docs](https://freshmart-9xou.onrender.com/docs)

## Project Purpose

FreshMart needs to anticipate vegetable demand for upcoming replenishment cycles (Mon/Wed/Sat) while managing uncertainty in daily sales. Over-ordering leads to spoilage and waste; under-ordering leads to stockouts and lost sales. This project builds a forecasting model and an interactive dashboard that give the Procurement Manager, ahead of each cycle:

- An expected demand estimate per vegetable
- A **95% Demand Range** - an empirically-calibrated uncertainty band built from the model's historical forecast errors. This is a visualization layer added in Module 5, **not** a formal statistical prediction interval, and the dashboard is explicit about that distinction throughout.
- Recent demand trends and the model's own reasoning (via SHAP) for each forecast

**The dashboard does not recommend an order quantity.** It is decision support only - the Procurement Manager makes the final call. This is a deliberate scope boundary, not an omission: Module 5's assignment explicitly separates "communicate model results clearly" from "make a recommendation," and the project follows that boundary throughout.

## Project Scope

The pilot covers high-volume vegetables at FreshMart's flagship store only. Automated ordering, inventory optimization, and expansion to additional stores are explicitly out of scope for this phase (see `docs/vision_document.docx`, Module 1).

## Key Results (honest summary, see the dashboard's Model Details and Monitoring tabs for full detail)

- **XGBoost model**, evaluated on a chronological hold-out (no random/shuffled splits, demand is time-dependent).
- The original acceptance target (≤20% MAPE on the top-10 highest-volume items) was **not met**: 77.6% MAPE. This is stated plainly, not hidden.
- MAPE is mathematically unstable when actual demand is near zero, which is common in this dataset (roughly 1 in 5 records fall below 2kg). Under WAPE, a volume-weighted metric far less sensitive to this — the same group scores **26.5%**, a materially different and more representative picture of typical performance.
- A volume-tier fairness gap was found (low-volume vegetables forecast meaningfully worse than high-volume ones), investigated, and partially, not fully, addressed through bias mitigation. This is documented transparently, including where mitigation fell short, rather than presented as solved.
- The model is deployed, monitored (with documented, defensible thresholds, not arbitrary ones), and has a demonstrated (not yet production-triggered) automated retraining workflow, evaluated against the established baseline before any promotion decision.

## Live Deliverables

| Deliverable | Status | Link |
|---|---|---|
| Interactive Stakeholder Dashboard (Module 5) | **Live** | https://freshmart-vz2kmxfpq6cwmrkfn5ahix.streamlit.app/ |
| Monitoring & Governance (Task 7) | **Live** - integrated as a tab in the same app (see *Architecture note* below) | Same URL, "Monitoring & Governance" tab |
| Prediction API (Module 4, `/predict`) | **Live** | https://freshmart-9xou.onrender.com/docs |
| CI/CD (GitHub Actions) | **Live** - runs on every push | `.github/workflows/ci.yml` |
| Docker Compose (multi-container local demo) | **Available** - pipeline, API, and dashboard containers, verified building and running locally | `docker-compose.yml` |

*Architecture note:* the monitoring dashboard is deliberately integrated into the same Streamlit app as the stakeholder dashboard rather than deployed separately, for centralized access and easier cross-referencing between forecast data and monitoring findings. A production rollout would add role-based access (e.g. via `streamlit-authenticator`) so the Procurement Manager's view stays focused on forecasting while analysts/governance see the Monitoring tab — not implemented in this student capstone, documented as a design intention in the final report.

## Technical Stack

- **Data & pipeline:** Python, Pandas, NumPy, Great Expectations (data quality), Apache Airflow (orchestration, `dags/`)
- **Modeling:** scikit-learn, XGBoost, SHAP (explainability), LIME (cross-check)
- **Experiment tracking & retraining:** MLflow (experiment tracking + Model Registry)
- **Serving:** FastAPI (`/predict` endpoint)
- **Dashboard:** Streamlit, Plotly
- **Governance & testing:** pytest, custom bias/PII checks (`src/governance/`)
- **Environment:** Docker (pipeline container); Docker Compose in progress

## Repository Structure

```
data/
  raw/            Original source CSVs (annex1-4)
  processed/      Cleaned, validated item-day table + governance audit log

src/
  data_prep/      Ingest, clean, transform, profile the raw data (Module 3)
  validation/     Great Expectations checks (Module 3)
  governance/     PII/anonymization checks, bias/representation checks, audit logging
  models/         Feature engineering, train/test split, model training + MLflow
                  logging, evaluation, explainability (SHAP/LIME/counterfactuals),
                  bias mitigation, model card, FastAPI serving app (Module 4)
  evaluation/     Module 5 evidence layer -- reproduces Module 4 results, adds
                  robustness metrics (WAPE/SMAPE/Median APE), tier-stratified
                  demand ranges, live What-If and per-item SHAP functions.
                  Produces the single evidence_bundle.pkl the dashboard reads.
  monitoring/     Task 7 -- backtested performance/drift/fairness monitoring,
                  incident response and decommissioning documentation.
                  Produces monitoring_bundle.pkl.
  retraining/     Automated retraining demonstration -- reuses the existing
                  preprocessing/training/evaluation code, evaluates the
                  retrained candidate against the established baseline, and
                  registers it in MLflow under a "candidate" alias only
                  (no auto-promotion; promotion requires governance review).
  utils/          Shared helpers

dashboards/
  app.py                 The live Streamlit app (Forecast Overview, Forecast
                          Explorer, Model Details, Monitoring & Governance)
  requirements.txt        Deployment-scoped dependencies (Streamlit Cloud
                          reads this file, not the root requirements.txt)

dags/             Airflow pipeline definition (Module 3)
docker/           Pipeline container (processing/validation/test environment)
docs/             Data dictionary, ethics committee notes, fairness audit log
notebooks/        Exploratory analysis
tests/            Pipeline tests (pytest)
mlflow.db         MLflow tracking store (experiments + Model Registry)
requirements.txt  Full pipeline dependencies (Modules 2-4)
.streamlit/       Dashboard theme configuration
```

## Running Locally

**Regenerate the Module 5 evidence bundle** (needed before running the dashboard for the first time, or after any change to `src/evaluation/evidence.py`):
```bash
python src/evaluation/evidence.py
```

**Regenerate the Task 7 monitoring bundle** (optional, the dashboard degrades gracefully without it):
```bash
python src/monitoring/monitoring.py
```

**Run the dashboard:**
```bash
streamlit run dashboards/app.py
```

**Run the retraining demonstration:**
```bash
python src/retraining/retrain.py
```

**Run the prediction API locally:**
```bash
uvicorn src.models.api:app --reload --port 8000
# then visit http://localhost:8000/docs for interactive Swagger UI
```

**Run the full Module 3-4 pipeline test suite:**
```bash
pytest tests/
```

## Governance

Ethical oversight sits with the **Data & AI Governance Committee** (analytics lead, procurement manager, store operations lead), which reviews fairness audits and override logs quarterly (see `docs/ethics/`). The model is decision support throughout its lifecycle; this applies to monitoring incident response and retraining candidate promotion as much as it does to individual forecasts: no automated action replaces human governance review.

## Project Status

Complete: forecasting model, data pipeline, live dashboard, monitoring, automated retraining (registered in the MLflow Model Registry under a `@candidate` alias, pending governance review before any production promotion), live API, CI/CD, and Docker Compose.
