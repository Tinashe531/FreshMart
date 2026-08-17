"""
Time-aware train/test split and rolling-origin CV (Module 4, Task 3).

FreshMart demand is time-dependent, so we never use a random/stratified
split (that would let future observations leak into training). Instead:

  - A chronological hold-out: the most recent HOLDOUT_DAYS of data is
    reserved, untouched, as the final test set.
  - Rolling-origin (expanding-window) cross-validation within the
    training period only, for model tuning. This mirrors
    sklearn.model_selection.TimeSeriesSplit but is written explicitly so
    the fold boundaries are visible and auditable.

References: Tashman (2000) on out-of-sample rolling-origin evaluation;
Hyndman & Athanasopoulos on time-series CV and seasonal-naive baselines;
scikit-learn TimeSeriesSplit documentation for implementation pattern.
"""

import pandas as pd

HOLDOUT_DAYS = 90  # ~3 months final untouched test set


def chronological_split(df: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS) -> dict:
    """Split by date: train = everything before the cutoff, test = the
    final `holdout_days` of the dataset. No shuffling, no per-item split."""
    df = df.sort_values("date").reset_index(drop=True)
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=holdout_days)

    train = df[df["date"] <= cutoff].reset_index(drop=True)
    test = df[df["date"] > cutoff].reset_index(drop=True)

    report = {
        "stage": "chronological_split",
        "cutoff_date": str(cutoff.date()),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_date_range": [str(train["date"].min().date()), str(train["date"].max().date())],
        "test_date_range": [str(test["date"].min().date()), str(test["date"].max().date())],
        "split_type": "chronological (no shuffling, no random/stratified split)",
    }
    return {"train": train, "test": test, "report": report}


def rolling_origin_folds(train_df: pd.DataFrame, n_folds: int = 3, min_train_days: int = 180,
                          val_days: int = 30):
    """Yield (train_idx, val_idx) folds with an expanding training window.

    Fold k's validation window always occurs strictly after fold k's
    training window, and all folds' data comes from `train_df` only
    (the chronological hold-out test set is never touched here).
    """
    train_df = train_df.sort_values("date").reset_index(drop=True)
    dates = train_df["date"]
    min_date = dates.min()
    max_date = dates.max()

    first_val_start = min_date + pd.Timedelta(days=min_train_days)
    total_val_span = (max_date - first_val_start).days
    if total_val_span < val_days:
        raise ValueError("Not enough data after min_train_days for even one validation fold.")

    fold_starts = pd.date_range(first_val_start, max_date - pd.Timedelta(days=val_days), periods=n_folds)

    folds = []
    for val_start in fold_starts:
        val_end = val_start + pd.Timedelta(days=val_days)
        train_idx = train_df.index[dates < val_start]
        val_idx = train_df.index[(dates >= val_start) & (dates < val_end)]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append((train_idx.to_numpy(), val_idx.to_numpy()))

    return folds