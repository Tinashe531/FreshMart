"""
FastAPI application — Module 4 deliverable (/predict endpoint).

Serves the final (mitigated) XGBoost model for one-day-ahead item-level
demand forecasting. Accepts raw, business-meaningful inputs (recent
demand history, price, calendar context) rather than requiring the
caller to know internal feature-encoding details.

Run locally:
    uvicorn src.models.api:app --reload --port 8000

Then visit:
    http://localhost:8000/docs

for interactive Swagger UI.
"""

import sys
from pathlib import Path
from datetime import date
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.features import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "src" / "models" / "artifacts"


app = FastAPI(
    title="FreshMart Demand Forecasting API",
    description=(
        "One-day-ahead item-level demand forecast "
        "(decision support only — not an automated ordering system). "
        "See MODEL_CARD.md for performance, fairness, and limitation details."
    ),
    version="1.0.0",
)


_model = None
_category_categories = None


def _load_model():
    """
    Load the final mitigated model and recover the category encoding
    used during training.
    """
    global _model, _category_categories

    model_path = ARTIFACTS_DIR / "xgboost_final.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Final model not found: {model_path}"
        )

    _model = joblib.load(model_path)

    # Recover the category values seen during training so that
    # category_code is encoded consistently at prediction time.
    train_df = pd.read_parquet(
        ARTIFACTS_DIR / "train_df.parquet"
    )

    _category_categories = sorted(
        train_df["category_code"].unique().tolist()
    )


@app.on_event("startup")
def startup_event():
    _load_model()


class PredictRequest(BaseModel):
    """Raw, business-meaningful inputs for a single item-day forecast."""

    forecast_date: date = Field(
        ...,
        description="Date being forecast (YYYY-MM-DD)"
    )

    category_code: int = Field(
        ...,
        description="FreshMart category code for this item"
    )

    demand_lag_1: float = Field(
        ...,
        ge=0,
        description="Actual demand (kg) 1 day before forecast_date"
    )

    demand_lag_7: float = Field(
        ...,
        ge=0,
        description="Actual demand (kg) 7 days before forecast_date"
    )

    demand_lag_14: float = Field(
        ...,
        ge=0,
        description="Actual demand (kg) 14 days before forecast_date"
    )

    demand_roll_mean_7: float = Field(
        ...,
        ge=0,
        description="Mean daily demand (kg) over the prior 7 days"
    )

    demand_roll_mean_14: float = Field(
        ...,
        ge=0,
        description="Mean daily demand (kg) over the prior 14 days"
    )

    demand_roll_std_7: float = Field(
        0.0,
        ge=0,
        description="Std dev of daily demand (kg) over the prior 7 days"
    )

    price_lag_1: Optional[float] = Field(
        None,
        description="Average selling price 1 day before forecast_date"
    )

    price_roll_mean_7: Optional[float] = Field(
        None,
        description="Mean average selling price over prior 7 days"
    )

    price_change_pct_7: Optional[float] = Field(
        0.0,
        description="Percentage price change versus 7 days prior"
    )

    item_historical_avg_demand: float = Field(
        ...,
        ge=0,
        description="Item's historical average daily demand (kg)"
    )


class PredictResponse(BaseModel):
    predicted_demand_kg: float
    forecast_date: date
    model_version: str
    note: str


@app.get("/")
def root():
    return {
        "service": "FreshMart Demand Forecasting API",
        "endpoints": ["/predict", "/health", "/docs"],
        "model_card": (
            "See MODEL_CARD.md in the repository for full "
            "performance, fairness and limitation details."
        ),
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "model_not_loaded"
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    d = req.forecast_date

    # --------------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------------
    day_of_week = d.weekday()

    row = {
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week in (5, 6)),
        "is_replenishment_day": int(day_of_week in (0, 2, 5)),
        "month": d.month,
        "day_of_year": d.timetuple().tm_yday,
        "week_of_year": d.isocalendar()[1],

        # Historical demand features
        "lag_1": req.demand_lag_1,
        "lag_7": req.demand_lag_7,
        "lag_14": req.demand_lag_14,
        "roll_mean_7": req.demand_roll_mean_7,
        "roll_mean_14": req.demand_roll_mean_14,
        "roll_std_7": req.demand_roll_std_7,

        # Historical price features
        "price_lag_1": (
            req.price_lag_1
            if req.price_lag_1 is not None
            else 0.0
        ),

        "price_roll_mean_7": (
            req.price_roll_mean_7
            if req.price_roll_mean_7 is not None
            else 0.0
        ),

        "price_change_pct_7": (
            req.price_change_pct_7
            if req.price_change_pct_7 is not None
            else 0.0
        ),

        # Historical item-level demand
        "item_expanding_mean_demand":
            req.item_historical_avg_demand,

        # Category
        "category_code": req.category_code,
    }

    # --------------------------------------------------------------
    # Build model input using the exact training feature order
    # --------------------------------------------------------------
    X = pd.DataFrame([row])[FEATURE_COLUMNS]

    # Use the same category encoding observed during training.
    if req.category_code not in _category_categories:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown category_code {req.category_code}. "
                f"Known categories: {_category_categories}"
            ),
        )

    X["category_code"] = _category_categories.index(
        req.category_code
    )

    # --------------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------------
    pred = float(_model.predict(X)[0])

    # Demand cannot be negative.
    pred = max(pred, 0.0)

    return PredictResponse(
        predicted_demand_kg=round(pred, 3),
        forecast_date=req.forecast_date,
        model_version=(
            "xgboost_final "
            "(mitigated, volume-reweighted)"
        ),
        note=(
            "Decision-support forecast only. Review against "
            "operational knowledge before finalizing procurement "
            "quantities. See MODEL_CARD.md for known limitations, "
            "including the unmet top-10-volume MAPE target and "
            "lower reliability for low-volume items."
        ),
    )