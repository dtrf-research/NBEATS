"""
N-BEATS Model Comparison Dashboard
====================================
Streamlit app for evaluating & comparing trained N-BEATS models.

Run from the *streamlit_app* folder:
    cd streamlit_app
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---- ensure parent directory is on path for local imports ----
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# Also add project root so Darts can find the checkpoint dirs via model_name
_PROJECT_ROOT = _APP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
os.chdir(_PROJECT_ROOT)  # model checkpoint paths are relative to project root

import numpy as np
import pandas as pd
import streamlit as st

from data_loader import (
    AVAILABLE_ZONES,
    MONTH_NAMES,
    TIME_COLUMN,
    filter_data,
    get_date_bounds,
    load_raw_data,
)
from inference import compute_metrics, run_inference
from model_loader import (
    list_models,
    load_model,
    load_registry,
    load_scaler,
    model_display_label,
)
from visualization import build_comparison_chart, build_error_chart

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="N-BEATS Model Comparison",
    page_icon="📈",
    layout="wide",
)

st.title("📈 N-BEATS Model Comparison Dashboard")

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading demand data …")
def _load_data():
    return load_raw_data()


@st.cache_data(show_spinner="Loading model registry …")
def _load_registry():
    return load_registry()


@st.cache_resource(show_spinner="Loading model checkpoint …")
def _load_model_cached(model_name: str, _entry_json: str):
    """Load model + scaler (cached by model_name)."""
    import json as _json
    entry = _json.loads(_entry_json)
    model = load_model(model_name, entry)
    scaler = load_scaler(entry)
    return model, scaler

# ---------------------------------------------------------------------------
# Data & registry
# ---------------------------------------------------------------------------
df_raw = _load_data()
registry = _load_registry()
model_names = list_models(registry)

min_date, max_date = get_date_bounds(df_raw)

# ---------------------------------------------------------------------------
# Sidebar – controls
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Evaluation Settings")

zone = st.sidebar.selectbox("Region / Zone", AVAILABLE_ZONES, index=0)

col_d1, col_d2 = st.sidebar.columns(2)
start_date = col_d1.date_input("Start date", value=pd.Timestamp("2025-01-01").date(), min_value=min_date, max_value=max_date)
end_date = col_d2.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

months = st.sidebar.multiselect(
    "Months (leave empty for all)",
    options=list(range(1, 13)),
    format_func=lambda m: MONTH_NAMES[m],
    default=[],
)

st.sidebar.markdown("---")
st.sidebar.header("🧠 Model Selection")

# Build display labels
display_labels = {name: model_display_label(name, registry[name]) for name in model_names}
label_to_name = {v: k for k, v in display_labels.items()}

selected_labels = st.sidebar.multiselect(
    "Choose models to compare",
    options=list(label_to_name.keys()),
    default=[],
)
selected_models = [label_to_name[lbl] for lbl in selected_labels]

run_button = st.sidebar.button("🚀 Run Evaluation", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

if not selected_models:
    st.info("Select one or more models from the sidebar, choose a region & date range, then press **Run Evaluation**.")
    st.stop()

if not run_button:
    st.warning("Press **Run Evaluation** to generate predictions.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# ---- Filter data ----
with st.spinner("Filtering data …"):
    df_filtered = filter_data(
        df_raw,
        zone,
        str(start_date),
        str(end_date),
        months if months else None,
    )

if df_filtered.empty:
    st.error("No data found for the selected filters.")
    st.stop()

st.caption(f"📊 Evaluation data: **{len(df_filtered):,}** rows from **{start_date}** to **{end_date}**")

# ---- Load models & run inference ----
predictions: dict[str, pd.DataFrame] = {}
metrics_table: list[dict] = []
progress = st.progress(0, text="Loading models & running inference …")

import json as _json

for i, model_name in enumerate(selected_models):
    entry = registry[model_name]
    entry_json = _json.dumps(entry)

    progress.progress(
        (i) / len(selected_models),
        text=f"Running **{model_name}** ({i + 1}/{len(selected_models)}) …",
    )

    model, scaler = _load_model_cached(model_name, entry_json)

    result_df = run_inference(model, scaler, df_filtered, zone)

    if result_df.empty:
        st.warning(f"No predictions for **{model_name}** — series may be too short for the model's input length.")
        continue

    predictions[model_name] = result_df

    # Metrics
    m = compute_metrics(result_df["Actual"].values, result_df["Predicted"].values)
    m["Model"] = model_name
    metrics_table.append(m)

progress.progress(1.0, text="Done ✅")

if not predictions:
    st.error("None of the selected models produced predictions. Try a larger date range.")
    st.stop()

# ---- Build an actual-only df (from the first prediction's merged actual) ----
first_pred = next(iter(predictions.values()))
actual_df = first_pred[["Timestamp", "Actual"]].copy()

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
st.subheader("📉 Actual vs Predictions")
fig = build_comparison_chart(actual_df, predictions, zone)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📐 Prediction Error")
fig_err = build_error_chart(predictions)
st.plotly_chart(fig_err, use_container_width=True)

# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------
st.subheader("📋 Model Metrics")
metrics_df = pd.DataFrame(metrics_table)
cols_order = ["Model", "MAPE (%)", "MAE", "RMSE", "R²"]
metrics_df = metrics_df[[c for c in cols_order if c in metrics_df.columns]]
metrics_df = metrics_df.sort_values("MAPE (%)", ascending=True).reset_index(drop=True)

# Highlight best values
st.dataframe(
    metrics_df.style.format({
        "MAPE (%)": "{:.2f}",
        "MAE": "{:.2f}",
        "RMSE": "{:.2f}",
        "R²": "{:.4f}",
    }).highlight_min(
        subset=["MAPE (%)", "MAE", "RMSE"],
        color="#d4edda",
    ).highlight_max(
        subset=["R²"],
        color="#d4edda",
    ),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Raw data expander
# ---------------------------------------------------------------------------
with st.expander("📄 View raw predictions"):
    for model_name, pred_df in predictions.items():
        st.markdown(f"**{model_name}**")
        st.dataframe(pred_df.head(200), use_container_width=True)
