"""
Pytest tests for the FreshMart pipeline processing functions.

These test whether our CODE behaves correctly on small, controlled
examples. This is distinct from Great Expectations, which tests whether
the resulting DATA meets quality expectations. See src/validation/.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_prep.clean import (
    check_and_drop_exact_duplicates,
    check_quantity_sign_consistency,
    join_item_master,
    standardize_transaction_types,
)
from src.data_prep.transform import build_item_day_demand
from src.governance.bias_check import compute_category_representation, flag_representation_changes
from src.governance.anonymize import scan_for_pii, anonymize


# ---------------------------------------------------------------------
# clean.py
# ---------------------------------------------------------------------

def test_exact_duplicates_are_removed_when_present():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    result, report = check_and_drop_exact_duplicates(df, "test")
    assert len(result) == 2
    assert report["duplicates_found"] == 1


def test_no_duplicates_means_no_rows_removed():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result, report = check_and_drop_exact_duplicates(df, "test")
    assert len(result) == 3
    assert report["duplicates_found"] == 0


def test_quantity_sign_consistency_passes_when_correctly_signed():
    df = pd.DataFrame({
        "Sale or Return": ["sale", "sale", "return"],
        "Quantity Sold (kilo)": [10.0, 5.0, -2.0],
    })
    result = check_quantity_sign_consistency(df)
    assert result["consistent"] is True
    assert result["sale_rows_negative"] == 0
    assert result["return_rows_not_negative"] == 0


def test_quantity_sign_consistency_fails_when_return_not_negative():
    df = pd.DataFrame({
        "Sale or Return": ["sale", "return"],
        "Quantity Sold (kilo)": [10.0, 2.0],  # return should be negative but isn't
    })
    result = check_quantity_sign_consistency(df)
    assert result["consistent"] is False
    assert result["return_rows_not_negative"] == 1


def test_item_master_join_keeps_unmatched_records():
    """Core rule: unmatched transactions must be KEPT, not dropped
    (no inner join), and flagged as UNKNOWN category."""
    transactions = pd.DataFrame({
        "Item Code": ["A", "B", "C"],
        "Quantity Sold (kilo)": [1.0, 2.0, 3.0],
    })
    item_master = pd.DataFrame({
        "Item Code": ["A", "B"],  # "C" is not in the master
        "Item Name": ["Apple", "Bean"],
        "Category Code": ["101", "102"],
        "Category Name": ["Fruit", "Legume"],
    })
    merged, report = join_item_master(transactions, item_master)

    assert len(merged) == 3  # record count preserved
    assert report["record_count_preserved"] is True
    assert report["unmatched_records"] == 1
    assert report["unmatched_distinct_items"] == 1

    unmatched_row = merged.loc[merged["Item Code"] == "C"].iloc[0]
    assert unmatched_row["Category Code"] == "UNKNOWN"
    assert unmatched_row["Category Name"] == "Unknown"


def test_standardize_transaction_types_parses_date():
    df = pd.DataFrame({
        "Date": ["2020-07-01"],
        "Item Code": [102900005117056],
        "Quantity Sold (kilo)": ["0.396"],
        "Unit Selling Price (RMB/kg)": ["7.6"],
        "Sale or Return": [" Sale "],
    })
    result = standardize_transaction_types(df)
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])
    assert result["Sale or Return"].iloc[0] == "sale"
    assert isinstance(result["Item Code"].iloc[0], str)


# ---------------------------------------------------------------------
# transform.py
# ---------------------------------------------------------------------

def test_net_demand_is_signed_sum_not_subtraction():
    """The documented example: sale=+20, return=-2 -> net demand = 18,
    NOT 20 - (-2) = 22."""
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
        "Item Code": ["A", "A"],
        "Quantity Sold (kilo)": [20.0, -2.0],
        "Unit Selling Price (RMB/kg)": [5.0, 5.0],
        "Category Code": ["101", "101"],
        "Category Name": ["Fruit", "Fruit"],
    })
    result = build_item_day_demand(df)
    out = result["data"]
    assert len(out) == 1
    assert out.iloc[0]["daily_demand_kg"] == pytest.approx(18.0)


def test_output_grain_is_one_row_per_item_date():
    """If an item sold 5 times on one day, output must have exactly one
    row for that item-date, not five."""
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2023-01-01"] * 5),
        "Item Code": ["A"] * 5,
        "Quantity Sold (kilo)": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Unit Selling Price (RMB/kg)": [5.0] * 5,
        "Category Code": ["101"] * 5,
        "Category Name": ["Fruit"] * 5,
    })
    result = build_item_day_demand(df)
    out = result["data"]
    assert len(out) == 1
    assert out.iloc[0]["daily_demand_kg"] == pytest.approx(15.0)
    assert out.iloc[0]["transaction_count"] == 5


def test_negative_net_demand_is_flagged_not_dropped():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2023-01-01"]),
        "Item Code": ["A"],
        "Quantity Sold (kilo)": [-5.0],  # net negative for the day
        "Unit Selling Price (RMB/kg)": [5.0],
        "Category Code": ["101"],
        "Category Name": ["Fruit"],
    })
    result = build_item_day_demand(df)
    out = result["data"]
    assert len(out) == 1  # not dropped
    assert out.iloc[0]["negative_net_demand_flag"] == True  # noqa: E712
    assert result["report"]["negative_net_demand_item_days"] == 1


def test_multiple_items_and_dates_produce_correct_row_count():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"]),
        "Item Code": ["A", "B", "A"],
        "Quantity Sold (kilo)": [1.0, 2.0, 3.0],
        "Unit Selling Price (RMB/kg)": [5.0, 5.0, 5.0],
        "Category Code": ["101", "102", "101"],
        "Category Name": ["Fruit", "Veg", "Fruit"],
    })
    result = build_item_day_demand(df)
    assert len(result["data"]) == 3  # (A,1/1), (B,1/1), (A,1/2)


# ---------------------------------------------------------------------
# governance/bias_check.py
# ---------------------------------------------------------------------

def test_representation_flags_change_above_threshold():
    raw = pd.DataFrame({"category_name": ["A"] * 25 + ["B"] * 75})
    processed = pd.DataFrame({"category_name": ["A"] * 20 + ["B"] * 80})  # A: 25%->20%, -5pp
    raw_share = compute_category_representation(raw)
    proc_share = compute_category_representation(processed)
    flags = flag_representation_changes(raw_share, proc_share, threshold_pp=2.0)
    flagged_categories = [f["category"] for f in flags if f["flagged"]]
    assert "A" in flagged_categories


def test_representation_does_not_flag_small_change():
    raw = pd.DataFrame({"category_name": ["A"] * 25 + ["B"] * 75})
    processed = pd.DataFrame({"category_name": ["A"] * 24 + ["B"] * 76})  # A: 25%->24%, -1pp
    raw_share = compute_category_representation(raw)
    proc_share = compute_category_representation(processed)
    flags = flag_representation_changes(raw_share, proc_share, threshold_pp=2.0)
    flagged_categories = [f["category"] for f in flags if f["flagged"]]
    assert "A" not in flagged_categories


# ---------------------------------------------------------------------
# governance/anonymize.py
# ---------------------------------------------------------------------

def test_scan_for_pii_finds_no_pii_in_current_schema():
    columns = ["date", "item_code", "category_code", "category_name",
               "daily_demand_kg", "avg_selling_price"]
    found = scan_for_pii(columns)
    assert found == []


def test_scan_for_pii_detects_known_pii_column_names():
    columns = ["date", "item_code", "customer_id", "loyalty_id", "payment_card_number"]
    found = scan_for_pii(columns)
    assert "customer_id" in found
    assert "loyalty_id" in found
    assert "payment_card_number" in found


def test_anonymize_removes_detected_pii_columns():
    df = pd.DataFrame({
        "date": ["2023-01-01"],
        "item_code": ["A"],
        "customer_id": ["CUST123"],
    })
    result, removed = anonymize(df)
    assert "customer_id" not in result.columns
    assert "customer_id" in removed
    assert "date" in result.columns and "item_code" in result.columns
