"""
Representation bias detection.

Question this answers: did our preparation process disproportionately
remove or alter representation of particular vegetable categories or
lower-volume items?

Not a demographic-fairness tool; this dataset has no demographic
fields. This is a representation-bias check, scoped to what the data
actually contains, comparing category share before vs. after processing.

Threshold: a category is flagged if its share of records changes by
more than 2 percentage points between raw and processed data. This is
an operational screening threshold for "investigate this," not a
statistical or ethical claim that a 2-point shift constitutes bias.
"""

import pandas as pd

THRESHOLD_PP = 2.0


def compute_category_representation(df: pd.DataFrame, category_col: str = "category_name") -> pd.Series:
    """Return each category's share of total records, as a percentage."""
    counts = df[category_col].value_counts()
    share_pct = (counts / counts.sum()) * 100
    return share_pct


def flag_representation_changes(raw_share: pd.Series, processed_share: pd.Series,
                                 threshold_pp: float = THRESHOLD_PP) -> list:
    """Compare raw vs. processed category shares and flag categories whose
    share changed by more than threshold_pp percentage points. Also reports
    absolute record-count-equivalent context via the raw/processed shares
    themselves so percentage changes aren't read without scale."""
    all_categories = sorted(set(raw_share.index) | set(processed_share.index))
    results = []
    for cat in all_categories:
        raw_pct = float(raw_share.get(cat, 0.0))
        proc_pct = float(processed_share.get(cat, 0.0))
        change_pp = proc_pct - raw_pct
        results.append({
            "category": cat,
            "raw_share_pct": round(raw_pct, 3),
            "processed_share_pct": round(proc_pct, 3),
            "change_pp": round(change_pp, 3),
            "flagged": abs(change_pp) > threshold_pp,
        })
    return results


def run_bias_check(raw_transactions: pd.DataFrame, cleaned_transactions: pd.DataFrame) -> dict:
    """Bias check comparing category representation in the raw transaction
    data against the CLEANED transaction data — same grain (one row per
    transaction) on both sides, so the comparison isolates the effect of
    the cleaning/item-master-join step itself.

    Deliberately does NOT compare against the item-day output: that
    dataset has a different grain (one row per item-date), so any shift
    there reflects transaction-frequency differences between categories,
    not representation loss from cleaning. Comparing mismatched grains
    would produce a misleading bias signal."""
    raw_named = raw_transactions.rename(columns={"Category Name": "category_name"}) \
        if "Category Name" in raw_transactions.columns else raw_transactions
    cleaned_named = cleaned_transactions.rename(columns={"Category Name": "category_name"}) \
        if "Category Name" in cleaned_transactions.columns else cleaned_transactions

    raw_share = compute_category_representation(raw_named, "category_name")
    cleaned_share = compute_category_representation(cleaned_named, "category_name")

    flags = flag_representation_changes(raw_share, cleaned_share)
    n_flagged = sum(1 for f in flags if f["flagged"])

    return {
        "comparison": "raw_transactions_vs_cleaned_transactions (same grain)",
        "threshold_pp": THRESHOLD_PP,
        "categories_checked": len(flags),
        "categories_flagged": n_flagged,
        "details": flags,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data_prep.ingest import ingest_all
    from src.data_prep.clean import clean_transactions

    raw = ingest_all()
    cleaned = clean_transactions(raw["transactions"], raw["item_master"])

    # Raw transactions joined to item master for category labels (join
    # itself is not what's being tested here — cleaning's effect is)
    raw_with_category = raw["transactions"].merge(
        raw["item_master"][["Item Code", "Category Name"]],
        on="Item Code", how="left"
    )

    result = run_bias_check(raw_with_category, cleaned["data"])
    print(f"Comparison: {result['comparison']}")
    print(f"Categories checked: {result['categories_checked']}")
    print(f"Categories flagged (>{result['threshold_pp']}pp change): {result['categories_flagged']}")
    for d in result["details"]:
        marker = "FLAG" if d["flagged"] else "ok"
        print(f"  [{marker}] {d['category']}: {d['raw_share_pct']}% -> "
              f"{d['processed_share_pct']}% ({d['change_pp']:+.2f}pp)")
