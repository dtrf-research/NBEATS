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
    discover_models_in_directory,
    get_available_zones,
    list_models,
    load_all_registries,
    load_model,
    load_registry,
    load_scaler,
    model_display_label,
)
from visualization import (
    build_blending_chart,
    build_blending_error_chart,
    build_comparison_chart,
    build_error_chart,
)
from seasonal_blending import (
    apply_hard_merge,
    apply_soft_transition,
    compute_contribution_summary,
    compute_segment_metrics,
    validate_schedule,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="N-BEATS Dashboard",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Dark-mode CSS overrides (tables, backgrounds, text)
# ---------------------------------------------------------------------------
_DARK_CSS = """
<style>
/* Ensure dataframe text stays visible on dark bg */
.stDataFrame td, .stDataFrame th {
    color: #FAFAFA !important;
}
/* Subtle border between rows */
.stDataFrame tr {
    border-bottom: 1px solid #2A2D35;
}
/* Expander header */
details summary {
    color: #FAFAFA !important;
}
/* Metric cards */
[data-testid="stMetric"] {
    background-color: #1A1C23;
    border-radius: 8px;
    padding: 12px;
}
</style>
"""
st.markdown(_DARK_CSS, unsafe_allow_html=True)

# Dark-friendly highlight colour for best-metric cells
_HIGHLIGHT_BEST = "#1B4332"  # muted green on dark bg

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading demand data …")
def _load_data():
    return load_raw_data()


@st.cache_data(show_spinner="Loading model registry …")
def _load_registry():
    return load_all_registries()


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
# Sidebar – optional directory-based model loading
# ---------------------------------------------------------------------------
with st.sidebar.expander("📂 Load models from directory", expanded=False):
    dir_path_str = st.text_input(
        "Model directory path",
        value=str(_PROJECT_ROOT / "models"),
        key="model_dir_path",
    )
    if st.button("Scan directory", key="scan_dir_btn"):
        scan_dir = Path(dir_path_str)
        if not scan_dir.is_dir():
            st.error("Directory does not exist.")
        else:
            discovered = discover_models_in_directory(scan_dir)
            if not discovered:
                st.warning("No valid model artifacts found in this directory.")
            else:
                # Merge discovered models into registry (discovered takes precedence)
                new_count = sum(1 for k in discovered if k not in registry)
                registry.update(discovered)
                model_names = list_models(registry)
                st.success(f"Found **{len(discovered)}** models ({new_count} new).")
                st.rerun()

# ---------------------------------------------------------------------------
# Sidebar – mode switch
# ---------------------------------------------------------------------------
app_mode = st.sidebar.radio(
    "Dashboard Mode",
    ["📈 Model Comparison", "🌾 Seasonal Blending"],
    index=0,
    horizontal=True,
)

if app_mode == "📈 Model Comparison":
    st.title("📈 N-BEATS Model Comparison Dashboard")
else:
    st.title("🌾 Seasonal Model Blending")

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

# Zone filter for models
_model_zones = get_available_zones(registry)
model_zone_filter = st.sidebar.selectbox(
    "Filter models by zone",
    options=["All Zones"] + _model_zones,
    index=0,
    key="model_zone_filter",
)

# Build display labels, apply zone filter
if model_zone_filter == "All Zones":
    _filtered_names = model_names
else:
    _filtered_names = [n for n in model_names if registry[n].get("zone") == model_zone_filter]

display_labels = {name: model_display_label(name, registry[name]) for name in _filtered_names}
label_to_name = {v: k for k, v in display_labels.items()}

import json as _json

# ===================================================================
# MODE A – Model Comparison  (original flow, untouched)
# ===================================================================
if app_mode == "📈 Model Comparison":
    # --- Select All / Clear helpers ---
    _cmp_options = list(label_to_name.keys())
    sa_col1, sa_col2 = st.sidebar.columns(2)
    if sa_col1.button("✅ Select All", key="sel_all_cmp", use_container_width=True):
        st.session_state["cmp_models"] = _cmp_options
        st.rerun()
    if sa_col2.button("❌ Clear", key="clr_all_cmp", use_container_width=True):
        st.session_state["cmp_models"] = []
        st.rerun()

    # Remove stale selections that are no longer in the filtered options
    if "cmp_models" in st.session_state:
        st.session_state["cmp_models"] = [
            lbl for lbl in st.session_state["cmp_models"] if lbl in label_to_name
        ]

    selected_labels = st.sidebar.multiselect(
        "Choose models to compare",
        options=_cmp_options,
        key="cmp_models",
    )
    selected_models = [label_to_name[lbl] for lbl in selected_labels]

    run_button = st.sidebar.button("🚀 Run Evaluation", type="primary", use_container_width=True)

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

    # ---- Visualizations ----
    st.subheader("📉 Actual vs Predictions")
    fig = build_comparison_chart(actual_df, predictions, zone)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📐 Prediction Error")
    fig_err = build_error_chart(predictions)
    st.plotly_chart(fig_err, use_container_width=True)

    # ---- Metrics table ----
    st.subheader("📋 Model Metrics")
    metrics_df = pd.DataFrame(metrics_table)
    cols_order = ["Model", "MAPE (%)", "MAE", "RMSE", "R²"]
    metrics_df = metrics_df[[c for c in cols_order if c in metrics_df.columns]]
    metrics_df = metrics_df.sort_values("MAPE (%)", ascending=True).reset_index(drop=True)

    st.dataframe(
        metrics_df.style.format({
            "MAPE (%)": "{:.2f}",
            "MAE": "{:.2f}",
            "RMSE": "{:.2f}",
            "R²": "{:.4f}",
        }).highlight_min(
            subset=["MAPE (%)", "MAE", "RMSE"],
            color=_HIGHLIGHT_BEST,
        ).highlight_max(
            subset=["R²"],
            color=_HIGHLIGHT_BEST,
        ),
        use_container_width=True,
    )

    with st.expander("📄 View raw predictions"):
        for model_name, pred_df in predictions.items():
            st.markdown(f"**{model_name}**")
            st.dataframe(pred_df.head(200), use_container_width=True)

# ===================================================================
# MODE B – Seasonal Blending
# ===================================================================
else:
    # --- Model multi-select with Select All / Clear ---
    _bld_options = list(label_to_name.keys())
    sa_b1, sa_b2 = st.sidebar.columns(2)
    if sa_b1.button("✅ Select All", key="sel_all_bld", use_container_width=True):
        st.session_state["blend_models"] = _bld_options
        st.rerun()
    if sa_b2.button("❌ Clear", key="clr_all_bld", use_container_width=True):
        st.session_state["blend_models"] = []
        st.rerun()

    # Remove stale selections that are no longer in the filtered options
    if "blend_models" in st.session_state:
        st.session_state["blend_models"] = [
            lbl for lbl in st.session_state["blend_models"] if lbl in label_to_name
        ]

    selected_labels_b = st.sidebar.multiselect(
        "Available Models",
        options=_bld_options,
        key="blend_models",
    )
    selected_models_b = [label_to_name[lbl] for lbl in selected_labels_b]

    # --- Fallback model ---
    fallback_label = st.sidebar.selectbox(
        "Fallback Model (fills gaps)",
        options=selected_labels_b if selected_labels_b else [""],
        index=0,
        key="blend_fallback",
    )
    fallback_model = label_to_name.get(fallback_label, "")

    # --- Soft transition ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔀 Transition Settings")
    enable_soft = st.sidebar.checkbox("Enable soft transition", value=False, key="blend_soft")
    transition_days = 0
    if enable_soft:
        transition_days = st.sidebar.number_input(
            "Transition window (days)", min_value=1, max_value=30, value=3, key="blend_td"
        )

    # -----------------------------------------------------------------
    # Schedule management (region-aware, session-state persisted)
    # -----------------------------------------------------------------
    if "blend_schedules" not in st.session_state:
        st.session_state["blend_schedules"] = {z: [] for z in AVAILABLE_ZONES}

    schedule_key = zone  # current region
    schedule: list = st.session_state["blend_schedules"][schedule_key]

    st.subheader("📅 Model Schedule Config")
    st.caption(f"Region: **{zone}** — add schedule entries mapping date ranges to models.")

    # --- Add new entry form ---
    with st.expander("➕ Add schedule entry", expanded=len(schedule) == 0):
        acol1, acol2, acol3 = st.columns([2, 2, 3])
        new_start = acol1.date_input("Segment start", value=start_date, min_value=min_date, max_value=max_date, key="new_seg_start")
        new_end = acol2.date_input("Segment end", value=end_date, min_value=min_date, max_value=max_date, key="new_seg_end")
        model_opts = selected_labels_b if selected_labels_b else ["(select models first)"]
        new_model_label = acol3.selectbox("Assign model", options=model_opts, key="new_seg_model")
        if st.button("Add Entry", key="add_entry_btn"):
            if new_model_label in label_to_name and new_start < new_end:
                schedule.append({
                    "start": str(new_start),
                    "end": str(new_end),
                    "model": label_to_name[new_model_label],
                })
                st.session_state["blend_schedules"][schedule_key] = schedule
                st.rerun()
            else:
                st.error("Invalid entry — ensure models are selected and start < end.")

    # --- Display current schedule with per-row remove buttons ---
    if schedule:
        st.markdown("**Current schedule:**")
        for idx, entry in enumerate(schedule):
            short_model = entry["model"].split("_")[1] if "_" in entry["model"] else entry["model"]
            cols = st.columns([2, 2, 4, 1])
            cols[0].text(entry["start"])
            cols[1].text(entry["end"])
            cols[2].text(entry["model"])
            if cols[3].button("🗑️", key=f"rm_sched_{idx}", help=f"Remove entry {idx + 1}"):
                schedule.pop(idx)
                st.session_state["blend_schedules"][schedule_key] = schedule
                st.rerun()

        # Clear all button
        if st.button("🗑️ Clear All Entries", key="clear_all_sched"):
            st.session_state["blend_schedules"][schedule_key] = []
            st.rerun()
    else:
        st.info("No schedule entries yet. Add one above.")

    # --- Run blending ---
    run_blend = st.sidebar.button("🚀 Run Blending", type="primary", use_container_width=True)

    if not selected_models_b:
        st.info("Select models from the sidebar to begin.")
        st.stop()

    if not schedule:
        st.warning("Add at least one schedule entry before running.")
        st.stop()

    if not run_blend:
        st.warning("Press **Run Blending** to generate merged predictions.")
        st.stop()

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    if not fallback_model:
        st.error("Select a fallback model.")
        st.stop()

    # --- Validate schedule ---
    sched_warnings = validate_schedule(schedule, selected_models_b,
                                       start_date, end_date)
    for w in sched_warnings:
        st.warning(f"⚠️ {w['message']}")

    # --- Filter data ---
    with st.spinner("Filtering data …"):
        df_filtered_b = filter_data(
            df_raw, zone, str(start_date), str(end_date),
            months if months else None,
        )

    if df_filtered_b.empty:
        st.error("No data found for the selected filters.")
        st.stop()

    st.caption(f"📊 Evaluation data: **{len(df_filtered_b):,}** rows from **{start_date}** to **{end_date}**")

    # --- Run inference for each selected model ---
    blend_predictions: dict[str, pd.DataFrame] = {}
    progress_b = st.progress(0, text="Loading models & running inference …")

    models_to_run = list(set(selected_models_b) | {fallback_model})
    for i, mname in enumerate(models_to_run):
        entry = registry.get(mname)
        if entry is None:
            st.warning(f"Model '{mname}' not found in registry.")
            continue
        entry_json = _json.dumps(entry)
        progress_b.progress(i / len(models_to_run), text=f"Running **{mname}** ({i + 1}/{len(models_to_run)}) …")
        mdl, scl = _load_model_cached(mname, entry_json)
        res = run_inference(mdl, scl, df_filtered_b, zone)
        if res.empty:
            st.warning(f"No predictions for **{mname}** — series may be too short.")
            continue
        blend_predictions[mname] = res

    progress_b.progress(1.0, text="Done ✅")

    if fallback_model not in blend_predictions:
        st.error("Fallback model produced no predictions. Try a larger date range.")
        st.stop()

    # --- Apply merge ---
    if enable_soft and transition_days > 0:
        merged_df = apply_soft_transition(
            blend_predictions, schedule, fallback_model, transition_days
        )
    else:
        merged_df = apply_hard_merge(blend_predictions, schedule, fallback_model)

    if merged_df.empty:
        st.error("Merge produced no results.")
        st.stop()

    # --- Actual DF ---
    actual_df_b = merged_df[["Timestamp", "Actual"]].copy()

    # --- Visualization ---
    st.subheader("📉 Actual vs Blended Prediction")
    show_indiv = st.checkbox("Show individual model predictions", value=True, key="show_indiv")
    fig_blend = build_blending_chart(
        actual_df_b, merged_df, blend_predictions, zone, schedule,
        show_individuals=show_indiv,
        transition_days=transition_days if enable_soft else 0,
    )
    st.plotly_chart(fig_blend, use_container_width=True)

    st.subheader("📐 Blended Prediction Error")
    fig_berr = build_blending_error_chart(merged_df)
    st.plotly_chart(fig_berr, use_container_width=True)

    # --- Metrics ---
    st.subheader("📋 Overall Merged Metrics")
    overall_m = compute_metrics(merged_df["Actual"].values, merged_df["Predicted"].values)
    st.dataframe(
        pd.DataFrame([overall_m]).style.format({
            "MAPE (%)": "{:.2f}", "MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}",
        }),
        use_container_width=True,
    )

    # --- Per-segment metrics ---
    st.subheader("📊 Per-Segment Metrics")
    seg_metrics = compute_segment_metrics(merged_df, schedule, fallback_model)
    if not seg_metrics.empty:
        st.dataframe(
            seg_metrics.style.format({
                "MAPE (%)": "{:.2f}", "MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}",
            }),
            use_container_width=True,
        )

    # --- Contribution summary ---
    st.subheader("🧩 Model Contribution Summary")
    contrib = compute_contribution_summary(merged_df)
    if not contrib.empty:
        st.dataframe(contrib, use_container_width=True, hide_index=True)

    # --- Raw merged data ---
    with st.expander("📄 View merged predictions"):
        st.dataframe(merged_df.head(500), use_container_width=True)
