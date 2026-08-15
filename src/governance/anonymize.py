"""
Data anonymisation / PII removal.

The current FreshMart Kaggle dataset contains no customer or payment
identifiers, so there is nothing to anonymise in the current data. This
script does not invent fake PII to demonstrate the mechanism — instead
it implements a real PII scan + removal mechanism designed to protect a
future production data source, and documents that the current dataset
was checked and found clean.

Design: identify columns whose name matches known PII patterns, remove
them if present, and log what was removed (or that nothing was found).
"""

import pandas as pd

from pathlib import Path

# Known PII column name patterns for a retail POS context. Matched
# case-insensitively against column names (substring match).
PII_PATTERNS = [
    "customer_id", "customer id",
    "loyalty_id", "loyalty id",
    "receipt_id", "receipt id",
    "payment_card", "card_number", "card number",
    "phone", "email", "customer address", "billing address", "shipping address",
    "national_id", "passport",
    "customer_name", "customer name", "cardholder_name", "cardholder name",
]
# NOTE: intentionally does NOT include a bare "name" pattern. This dataset
# has legitimate business fields named "Item Name" / "Category Name" that
# are not PII. Only customer/cardholder-specific name fields are matched.


def scan_for_pii(columns: list) -> list:
    """Return the subset of column names that match a known PII pattern."""
    found = []
    for col in columns:
        col_lower = col.lower().replace("_", " ").strip()
        for pattern in PII_PATTERNS:
            pattern_norm = pattern.replace("_", " ")
            if pattern_norm in col_lower:
                found.append(col)
                break
    return found


def anonymize(df: pd.DataFrame) -> tuple:
    """Remove any columns matching a known PII pattern. Returns the
    (possibly unchanged) dataframe and the list of columns removed."""
    pii_columns = scan_for_pii(list(df.columns))
    if pii_columns:
        df = df.drop(columns=pii_columns)
    return df, pii_columns


if __name__ == "__main__":
    import pandas as pd

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "freshmart_item_day.parquet"

    processed = pd.read_parquet(PROCESSED_PATH)
    _, removed = anonymize(processed)
    if removed:
        print(f"PII columns found and removed: {removed}")
    else:
        print("No PII identified in the current dataset "
              "(checked columns: " + ", ".join(processed.columns) + ").")
        print("This script is designed to remove the following PII patterns "
              "if they appear in a future production data source: "
              + ", ".join(PII_PATTERNS))
