"""
FreshMart Demand Forecasting Dashboard
========================================

Streamlit app for Module 5. Reads everything static from the single
evidence bundle produced by src/evaluation/evidence.py. The only live
model calls happen in the What-If tool and the per-item SHAP explanation
in the Forecast Explorer -- both imported directly from that same script.

Theme colors live in .streamlit/config.toml (Streamlit's native theming,
the equivalent of R Shiny's bslib::bs_theme()) so native widgets --
buttons, tabs, sliders, the sidebar -- pick up the FreshMart palette
automatically. Custom CSS below is only for elements the theme config
can't reach (status badges, section dividers, KPI card accents). Plotly
charts do NOT inherit the Streamlit theme, so every figure explicitly
sets its own font/axis/title colors via style_fig().

Run (from the repo root):
    streamlit run dashboards/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evidence import (
    run_what_if,
    load_baseline_row,
    explain_forecast_live,
    WHAT_IF_INTERPRETABLE_FEATURES,
    FEATURE_PLAIN_LANGUAGE,
)

BUNDLE_PATH = REPO_ROOT / "src" / "evaluation" / "evidence_bundle.pkl"

# ============================================================================
# PALETTE (mirrors .streamlit/config.toml -- kept here too since Plotly
# and custom CSS need the exact hex values directly, not via theme lookup)
# ============================================================================

COLOR_PRIMARY = "#1B4332"       # deep basil green
COLOR_SECONDARY = "#2D6A4F"     # mid green, for secondary chart series
COLOR_BG = "#FFFFFF"
COLOR_CARD_BG = "#F7F7F5"       # warm light gray, distinct layer from white
COLOR_TEXT = "#1A1A1A"
COLOR_MUTED = "#5C6660"
COLOR_CAUTION = "#D4A017"       # amber -- background/border ONLY, never text
COLOR_CAUTION_BG = "#FBF1D9"
COLOR_WARNING = "#D9642B"       # burnt orange -- "no forecast", negative signals
COLOR_WARNING_BG = "#FBE6DA"
COLOR_OK_BG = "#E2EFE7"

st.set_page_config(
    page_title="FreshMart Demand Forecasting",
    page_icon="\U0001F96C",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# CUSTOM CSS -- only for things the theme config can't style: status badges,
# section headers, KPI card accents, caution/limitation boxes. Every color
# below is set explicitly (no reliance on inherited/default text color),
# since that inheritance gap was the root cause of last round's invisible-
# text bugs.
# ============================================================================

st.markdown(f"""
<style>
    .block-container {{ padding-top: 2rem; max-width: 1400px; }}

    /* Tabs -- make them unmistakable, not swallowed into body text */
    button[data-baseweb="tab"] {{
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: {COLOR_MUTED} !important;
        padding: 12px 22px !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {COLOR_PRIMARY} !important;
    }}
    div[data-baseweb="tab-highlight"] {{ background-color: {COLOR_PRIMARY} !important; height: 4px !important; }}
    div[data-baseweb="tab-border"] {{ background-color: #E3E5E1 !important; height: 2px !important; }}

    /* Section headers -- ONE consistent style used everywhere via
       section_header(), so tabs can no longer drift out of sync in size */
    .section-header {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {COLOR_TEXT} !important;
        border-bottom: 2px solid {COLOR_PRIMARY};
        padding-bottom: 6px;
        margin: 28px 0 14px 0;
    }}
    .tab-question {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {COLOR_TEXT} !important;
        margin: 4px 0 20px 0;
    }}
    .widget-label {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {COLOR_TEXT} !important;
        margin-bottom: 4px;
    }}

    /* KPI cards */
    div[data-testid="stMetric"] {{
        background-color: {COLOR_CARD_BG};
        border: 1px solid #E3E5E1;
        border-left: 5px solid {COLOR_PRIMARY};
        border-radius: 8px;
        padding: 14px 18px;
    }}
    div[data-testid="stMetricLabel"] p {{ color: {COLOR_MUTED} !important; font-size: 0.88rem !important; }}
    div[data-testid="stMetricValue"] {{ color: {COLOR_PRIMARY} !important; }}
    div[data-testid="stMetricDelta"] {{ color: {COLOR_SECONDARY} !important; }}

    /* Status badges */
    .status-badge {{
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
    }}
    .status-current {{ background-color: {COLOR_OK_BG}; color: {COLOR_PRIMARY} !important; }}
    .status-older {{ background-color: {COLOR_CAUTION_BG}; color: #8A6600 !important; }}
    .status-none {{ background-color: {COLOR_WARNING_BG}; color: {COLOR_WARNING} !important; }}

    /* Caution / info / limitation boxes -- text color always explicit */
    .caution-box {{
        background-color: {COLOR_CAUTION_BG};
        border-left: 4px solid {COLOR_CAUTION};
        border-radius: 6px;
        padding: 14px 18px;
        font-size: 0.95rem;
        color: {COLOR_TEXT} !important;
        margin: 10px 0;
    }}
    .info-box {{
        background-color: {COLOR_CARD_BG};
        border-left: 4px solid {COLOR_SECONDARY};
        border-radius: 6px;
        padding: 14px 18px;
        font-size: 0.95rem;
        color: {COLOR_TEXT} !important;
        margin: 10px 0;
    }}
    .limitation-box {{
        background-color: {COLOR_CARD_BG};
        border: 1px solid #E3E5E1;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 10px;
        color: {COLOR_TEXT} !important;
        font-size: 0.95rem;
    }}
    .about-box {{
        background-color: {COLOR_CARD_BG};
        border-radius: 8px;
        padding: 16px 20px;
        color: {COLOR_TEXT} !important;
        font-size: 0.92rem;
    }}
    /* Stable, documented targeting via st.container(key=...) -- NOT an
       internal/minified Streamlit class name, which broke last round. */
    div.st-key-vegetable_selector_box {{
        border: 3px solid {COLOR_CAUTION};
        border-radius: 10px;
        padding: 16px 20px 4px 20px;
        margin-bottom: 18px;
        background-color: #FFFDF7;
    }}
</style>
""", unsafe_allow_html=True)


def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def tab_question(text: str):
    st.markdown(f'<div class="tab-question">{text}</div>', unsafe_allow_html=True)


def widget_label(text: str):
    st.markdown(f'<div class="widget-label">{text}</div>', unsafe_allow_html=True)


def status_badge(status: str) -> str:
    if status == "Current forecast":
        cls = "status-current"
    elif status.startswith("Older"):
        cls = "status-older"
    else:
        cls = "status-none"
    return f'<span class="status-badge {cls}">{status}</span>'


def style_fig(fig, height=360, show_legend=True):
    """Applied to every Plotly figure. Plotly does NOT inherit Streamlit's
    theme -- every color here is explicit, which is what last round's
    invisible-axis-label bug was missing."""
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=COLOR_TEXT, size=13),
        legend=dict(orientation="h", y=-0.22, font=dict(color=COLOR_TEXT)) if show_legend else dict(),
        showlegend=show_legend,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_xaxes(
        title_font=dict(color=COLOR_TEXT, size=13),
        tickfont=dict(color=COLOR_TEXT, size=12),
        gridcolor="#EDEFEB", linecolor="#C9CDC7", zerolinecolor="#C9CDC7",
    )
    fig.update_yaxes(
        title_font=dict(color=COLOR_TEXT, size=13),
        tickfont=dict(color=COLOR_TEXT, size=12),
        gridcolor="#EDEFEB", linecolor="#C9CDC7", zerolinecolor="#C9CDC7",
    )
    return fig


# ============================================================================
# DATA LOADING (cached -- bundle is read once per session, not per rerun)
# ============================================================================

@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        st.error(
            f"Evidence bundle not found at `{BUNDLE_PATH}`. Run "
            f"`python src/evaluation/evidence.py` from the repo root first."
        )
        st.stop()
    return joblib.load(BUNDLE_PATH)


bundle = load_bundle()
forward = bundle["forward_forecasts"]
narrative = bundle["narrative_content"]
labels = narrative["labels"]
highlights = bundle["dashboard_highlights"]
trend_series = bundle["trend_series"]

RANGE_TERM = labels["range_term"]  # "95% Demand Range" -- never CI/PI anywhere below

# ============================================================================
# HEADER
# ============================================================================

st.title("\U0001F96C FreshMart Demand Forecasting")
st.caption(
    "Decision support for procurement planning. Forecasts are estimates, not "
    "guarantees -- the Procurement Manager makes the final call."
)

n_current = int((forward["forecast_status"] == "Current forecast").sum())
n_older = int(forward["forecast_status"].str.startswith("Older", na=False).sum())
n_none = int((forward["forecast_status"] == "No forecast available").sum())

tab1, tab2, tab3 = st.tabs(["\U0001F4CA Forecast Overview", "\U0001F50D Forecast Explorer", "\u2699\uFE0F Model Details"])

# ============================================================================
# TAB 1 -- FORECAST OVERVIEW
# ============================================================================

with tab1:
    tab_question("What should I pay attention to?")

    kc1, kc2, kc3 = st.columns(3)
    with kc1:
        h = highlights["highest_forecast_demand"]
        st.metric("Highest Forecast Demand", f"{h['value_kg']:.1f} kg")
        st.caption(h["display_label"])
    with kc2:
        h = highlights["largest_expected_increase"]
        st.metric("Largest Expected Increase", f"+{h['value_pct']:.0f}%")
        st.caption(h["display_label"])
    with kc3:
        h = highlights["largest_expected_decrease"]
        st.metric("Largest Expected Decrease", f"{h['value_pct']:.0f}%")
        st.caption(h["display_label"])

    current_fwd = forward[forward["forecast_status"] == "Current forecast"].copy()

    section_header("Vegetables with the Highest Forecast Demand")
    left, right = st.columns([3, 2])

    with left:
        show_all = st.checkbox("Show all vegetables", value=False, key="tab1_show_all_bar")
        chart_data = current_fwd if show_all else current_fwd.nlargest(10, "forecast_demand_kg")
        chart_data = chart_data.sort_values("forecast_demand_kg", ascending=True)

        fig_bar = px.bar(
            chart_data, x="forecast_demand_kg", y="display_label", orientation="h",
            labels={"forecast_demand_kg": "Forecast Demand (kg)", "display_label": ""},
            color_discrete_sequence=[COLOR_PRIMARY],
        )
        style_fig(fig_bar, height=max(340, 30 * len(chart_data)), show_legend=False)
        st.plotly_chart(fig_bar, width='stretch', key='tab1_bar_chart')

    with right:
        widget_label("Demand Trend: Historical + Forecast (kg)")
        top10_current = current_fwd.nlargest(10, "forecast_demand_kg")
        label_to_item = dict(zip(top10_current["display_label"], top10_current["item_code"]))
        chosen_label = st.selectbox("Vegetable", list(label_to_item.keys()), index=0,
                                     key="tab1_trend_select", label_visibility="collapsed")
        chosen_item = label_to_item[chosen_label]

        item_hist = trend_series[(trend_series["item_code"] == chosen_item) & (trend_series["series_type"] == "historical")].tail(120)
        item_fwd = trend_series[(trend_series["item_code"] == chosen_item) & (trend_series["series_type"] == "forecast")]
        item_row = forward[forward["item_code"] == chosen_item].iloc[0]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=item_hist["date"], y=item_hist["demand_kg"], mode="lines",
            name="Historical demand", line=dict(color=COLOR_MUTED, width=1.5),
        ))
        if not item_fwd.empty and len(item_hist):
            fig_trend.add_trace(go.Scatter(
                x=[item_hist["date"].iloc[-1], item_fwd["date"].iloc[0]],
                y=[item_hist["demand_kg"].iloc[-1], item_fwd["demand_kg"].iloc[0]],
                mode="lines+markers", name="Forecast",
                line=dict(color=COLOR_PRIMARY, width=2, dash="dot"),
                marker=dict(size=9, color=COLOR_PRIMARY),
            ))
            fig_trend.add_trace(go.Scatter(
                x=[item_fwd["date"].iloc[0], item_fwd["date"].iloc[0]],
                y=[item_row["range_lower_kg"], item_row["range_upper_kg"]],
                mode="lines", line=dict(color=COLOR_CAUTION, width=6),
                name=RANGE_TERM, opacity=0.6,
            ))
        style_fig(fig_trend, height=340)
        st.plotly_chart(fig_trend, width='stretch', key='tab1_trend_chart')

    section_header("Forecast Demand by Vegetable")
    table_scope = st.radio("Scope", ["High-volume vegetables", "All vegetables"],
                            horizontal=True, key="tab1_table_scope")
    if table_scope == "High-volume vegetables":
        table_df = current_fwd[current_fwd["volume_tier"] == "high_volume"]
    else:
        table_df = current_fwd

    display_table = table_df.copy()
    display_table["Vegetable"] = display_table["display_label"]
    display_table["Forecast (kg)"] = display_table["forecast_demand_kg"].round(1)
    display_table["Recent Avg (kg)"] = display_table["recent_demand_avg_kg"].round(1)
    display_table["Change"] = display_table["change_vs_recent_demand_avg_pct"].round(0).astype(int).astype(str) + "%"
    display_table[RANGE_TERM] = (
        display_table["range_lower_kg"].round(1).astype(str) + " \u2013 " + display_table["range_upper_kg"].round(1).astype(str) + " kg"
    )
    display_table = display_table[["Vegetable", "Forecast (kg)", "Recent Avg (kg)", "Change", RANGE_TERM]]

    if table_scope == "High-volume vegetables":
        display_table = display_table.sort_values("Forecast (kg)", ascending=False)
        st.caption(f"Showing {len(display_table)} high-volume vegetables, sorted by forecast demand (highest first).")
    else:
        display_table = display_table.sort_values("Vegetable", ascending=True)
        st.caption(f"Showing all {len(display_table)} vegetables with a current forecast, sorted alphabetically.")

    st.dataframe(
        display_table, hide_index=True, width='stretch', height=380,
        column_config={
            "Vegetable": st.column_config.TextColumn(width="medium"),
            "Forecast (kg)": st.column_config.NumberColumn(width="small"),
            "Recent Avg (kg)": st.column_config.NumberColumn(width="small"),
            "Change": st.column_config.TextColumn(width="small"),
            RANGE_TERM: st.column_config.TextColumn(width="medium"),
        },
    )

# ============================================================================
# TAB 2 -- FORECAST EXPLORER
# ============================================================================

with tab2:
    tab_question("What is the forecast for this vegetable, and why?")

    with st.container(key="vegetable_selector_box"):
        widget_label("Select a vegetable \u2014 everything below depends on this choice")
        all_items_sorted = forward.sort_values(
            ["forecast_status", "display_label"],
            key=lambda s: s.map({"Current forecast": 0, "Older activity \u2014 treat with caution": 1, "No forecast available": 2}) if s.name == "forecast_status" else s,
        )
        label_to_item2 = dict(zip(all_items_sorted["display_label"], all_items_sorted["item_code"]))
        chosen_label2 = st.selectbox("Select a vegetable", list(label_to_item2.keys()),
                                      key="tab2_select", label_visibility="collapsed")
        sel_item = label_to_item2[chosen_label2]
        sel_row = forward[forward["item_code"] == sel_item].iloc[0]

        st.markdown(status_badge(sel_row["forecast_status"]), unsafe_allow_html=True)
        st.caption(f"Category: {sel_row['category_name']} \u00b7 Last known activity: {sel_row['last_known_date']:%d %b %Y} "
                   f"({int(sel_row['days_since_last_activity'])} days before this dataset's most recent date)")

    if sel_row["forecast_status"] == "No forecast available":
        st.markdown(
            '<div class="caution-box">This item does not have enough recorded history to generate a '
            'forward forecast. No forecast, range, or scenario analysis is available for it.</div>',
            unsafe_allow_html=True,
        )
    else:
        if sel_row["forecast_status"].startswith("Older"):
            st.markdown(
                '<div class="caution-box"><b>Older activity.</b> This item\u2019s last recorded sale was '
                'more than 90 days before the most recent data in this set. Treat this forecast with '
                'extra caution -- demand patterns may have changed since.</div>',
                unsafe_allow_html=True,
            )

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Expected Demand", f"{sel_row['forecast_demand_kg']:.1f} kg")
        with m2:
            st.metric(RANGE_TERM, f"{sel_row['range_lower_kg']:.1f} \u2013 {sel_row['range_upper_kg']:.1f} kg")
        st.caption(narrative["uncertainty_explanation"])

        section_header("Demand Trend: Historical + Forecast")
        item_hist2 = trend_series[(trend_series["item_code"] == sel_item) & (trend_series["series_type"] == "historical")].tail(120)
        item_fwd2 = trend_series[(trend_series["item_code"] == sel_item) & (trend_series["series_type"] == "forecast")]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=item_hist2["date"], y=item_hist2["demand_kg"], mode="lines",
                                   name="Historical demand", line=dict(color=COLOR_MUTED, width=1.5)))
        if not item_fwd2.empty and len(item_hist2):
            fig2.add_trace(go.Scatter(
                x=[item_hist2["date"].iloc[-1], item_fwd2["date"].iloc[0]],
                y=[item_hist2["demand_kg"].iloc[-1], item_fwd2["demand_kg"].iloc[0]],
                mode="lines+markers", name="Forecast", line=dict(color=COLOR_PRIMARY, width=2, dash="dot"),
                marker=dict(size=9, color=COLOR_PRIMARY),
            ))
            fig2.add_trace(go.Scatter(
                x=[item_fwd2["date"].iloc[0], item_fwd2["date"].iloc[0]],
                y=[sel_row["range_lower_kg"], sel_row["range_upper_kg"]],
                mode="lines", line=dict(color=COLOR_CAUTION, width=6), name=RANGE_TERM, opacity=0.6,
            ))
        style_fig(fig2, height=340)
        st.plotly_chart(fig2, width='stretch', key='tab2_trend_chart')

        col_a, col_b = st.columns(2)
        with col_a:
            section_header("How Is This Forecast Calculated?")
            st.markdown(f'<div class="info-box">{narrative["plain_language_model_explanation"]}</div>', unsafe_allow_html=True)
        with col_b:
            section_header("Why Is This Forecast Higher or Lower?")
            baseline_features = load_baseline_row(forward, sel_item)
            live_explanation = explain_forecast_live(baseline_features)
            st.markdown(f'<div class="info-box">{live_explanation["plain_language_explanation"]}</div>', unsafe_allow_html=True)

        section_header("What Happens If Demand Conditions Change?")
        st.caption("Scenario analysis -- this changes hypothetical inputs and reruns the model live. It is not a new observed forecast.")

        wc1, wc2, wc3, wc4 = st.columns(4)
        with wc1:
            widget_label("Previous recorded day (kg)")
            new_lag1 = st.slider("Previous recorded day (kg)", 0.0, 60.0, float(baseline_features["lag_1"]),
                                  key="wi_lag1", label_visibility="collapsed")
        with wc2:
            widget_label("Recent demand, 7-day avg (kg)")
            new_roll7 = st.slider("Recent demand, 7-day avg (kg)", 0.0, 60.0, float(baseline_features["roll_mean_7"]),
                                   key="wi_roll7", label_visibility="collapsed")
        with wc3:
            widget_label("Recent selling price")
            new_price = st.slider("Recent selling price", 0.0, 40.0, float(baseline_features["price_lag_1"]),
                                   key="wi_price", label_visibility="collapsed")
        with wc4:
            widget_label("Day of week")
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            new_dow = st.selectbox("Day of week", day_names, index=int(baseline_features["day_of_week"]),
                                    key="wi_dow", label_visibility="collapsed")

        if st.button("Run scenario", type="primary"):
            overrides = {
                "lag_1": new_lag1, "roll_mean_7": new_roll7,
                "price_lag_1": new_price, "day_of_week": day_names.index(new_dow),
            }
            result = run_what_if(baseline_features, overrides)
            r1, r2, r3 = st.columns(3)
            r1.metric("Baseline Forecast", f"{result['baseline_forecast_kg']:.1f} kg")
            r2.metric("Scenario Forecast", f"{result['scenario_forecast_kg']:.1f} kg",
                      delta=f"{result['change_kg']:+.1f} kg")
            r3.metric("Change", f"{result['change_pct']:+.1f}%" if result["change_pct"] is not None else "n/a")
            st.caption("This is a hypothetical scenario based on the inputs above, not an observed forecast.")

# ============================================================================
# TAB 3 -- MODEL DETAILS
# ============================================================================

with tab3:
    tab_question("How well does the model work, how does it predict, and what are its limits?")

    section_header("Model Performance")

    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        widget_label("XGBoost (final) vs. seasonal-naive baseline")
        bc = bundle["baseline_comparison"].copy()
        bc_display = bc.rename(columns={"model": "Model", "mae": "MAE", "rmse": "RMSE", "mape_pct": "MAPE %", "r2": "R\u00b2"})
        bc_display["MAE"] = bc_display["MAE"].round(2)
        bc_display["RMSE"] = bc_display["RMSE"].round(2)
        bc_display["MAPE %"] = bc_display["MAPE %"].round(1)
        bc_display["R\u00b2"] = bc_display["R\u00b2"].round(3)
        st.dataframe(bc_display, hide_index=True, width='stretch')
        st.caption("XGBoost reduces error substantially versus a naive same-day-last-week baseline.")

    with perf_col2:
        widget_label("Overall vs. Top-10 (multiple metrics)")
        ov_display = bundle["overall_top10_metrics"][["scope", "mae", "rmse", "mape_pct", "wape_pct", "median_ape_pct"]].rename(
            columns={"scope": "Scope", "mae": "MAE", "rmse": "RMSE", "mape_pct": "MAPE %",
                     "wape_pct": "WAPE %", "median_ape_pct": "Med APE %"}
        ).round(1)
        ov_display["Scope"] = ov_display["Scope"].replace({"overall": "Overall", "top10_by_train_volume": "Top-10"})
        st.dataframe(ov_display, hide_index=True, width='stretch',
                     column_config={"Scope": st.column_config.TextColumn(width="small")})

    top10_row = bundle["overall_top10_metrics"].iloc[1]
    met_met = top10_row["meets_20pct_mape_target"]
    st.markdown(
        f'<div class="caution-box">'
        f'<b>Why does MAPE show {top10_row["mape_pct"]:.1f}% on the top-10 highest-volume vegetables?</b><br>'
        f'MAPE measures error as a percentage of the actual value, so it becomes unstable whenever actual '
        f'demand is very small \u2014 a handful of near-zero-demand days can inflate this figure sharply even '
        f'when the forecast is off by less than a kilogram. In this dataset, roughly 1 in 5 records fall '
        f'below 2kg, and 2 of the original 10 highest-volume items had no recorded sales at all during the '
        f'test period. Because of this, we report WAPE and Median APE alongside MAPE throughout this tab \u2014 '
        f'both are far less sensitive to near-zero demand and give a more representative picture of typical '
        f'performance: <b>WAPE is {top10_row["wape_pct"]:.1f}%</b>, <b>Median APE is {top10_row["median_ape_pct"]:.1f}%</b>.'
        f'<br><br>'
        f'<b>Acceptance target:</b> \u2264 20% MAPE on the top-10 highest-volume vegetables. '
        f'<b>{"Met" if met_met else "Not met"}</b> on MAPE \u2014 but WAPE and Median APE both land much closer to target, '
        f'consistent with the explanation above.'
        f'</div>',
        unsafe_allow_html=True,
    )

    section_header("What Drives the Forecast?")
    dcol1, dcol2 = st.columns([3, 2])
    with dcol1:
        gshap = bundle["global_shap"].copy()
        fig_shap = px.bar(
            gshap.sort_values("mean_abs_shap", ascending=True).tail(10),
            x="mean_abs_shap", y="feature", orientation="h",
            labels={"mean_abs_shap": "Mean |SHAP value|", "feature": ""},
            color_discrete_sequence=[COLOR_PRIMARY],
        )
        style_fig(fig_shap, height=380, show_legend=False)
        st.plotly_chart(fig_shap, width='stretch', key='tab3_shap_chart')
        st.caption(narrative["plain_language_shap_explanation"])
    with dcol2:
        widget_label("LIME cross-check")
        st.caption(
            "Module 4 ran a one-time LIME cross-check on a representative prediction to validate "
            "the SHAP-identified drivers using an independent method. Top factors agreed with SHAP:"
        )
        lime_feats = bundle["lime_cross_check_top_features"][:5]
        for feat, weight in lime_feats:
            st.markdown(f'<div class="info-box" style="padding:8px 14px; margin:6px 0;">'
                        f'<code>{feat}</code> &nbsp; weight: {weight:+.2f}</div>', unsafe_allow_html=True)

    section_header("Forecast Performance Across Vegetables")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        widget_label("By volume tier")
        vt = bundle["volume_tier_metrics"]
        vt_after = vt[vt["model_stage"] == "after_mitigation"]
        fig_vt = px.bar(vt_after, x="volume_tier", y="wape_pct", color="volume_tier",
                         labels={"wape_pct": "WAPE (%)", "volume_tier": ""},
                         color_discrete_sequence=[COLOR_PRIMARY, COLOR_SECONDARY, COLOR_CAUTION])
        style_fig(fig_vt, height=320, show_legend=False)
        st.plotly_chart(fig_vt, width='stretch', key='tab3_volume_tier_chart')
    with fcol2:
        widget_label("By category")
        cm = bundle["category_metrics"].sort_values("wape_pct")
        fig_cat = px.bar(cm, x="wape_pct", y="category_name", orientation="h",
                          labels={"wape_pct": "WAPE (%)", "category_name": ""},
                          color_discrete_sequence=[COLOR_PRIMARY])
        style_fig(fig_cat, height=320, show_legend=False)
        st.plotly_chart(fig_cat, width='stretch', key='tab3_category_chart')
    st.caption(narrative["fairness_summary"])
    st.caption(narrative["bias_mitigation_explanation"])

    section_header("Model Limitations")
    for lim in narrative["model_limitations"]:
        st.markdown(f'<div class="limitation-box">{lim}</div>', unsafe_allow_html=True)

    section_header("Human Oversight & Transparency")
    st.markdown(f'<div class="info-box">{narrative["transparency_statement"]}</div>', unsafe_allow_html=True)
