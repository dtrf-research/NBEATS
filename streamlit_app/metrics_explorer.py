"""Model metrics explorer for visualizing and selecting best models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZONE_DIRS = ["TPCODL", "TPWODL", "TPNODL", "TPSOSDL", "Total"]
RESULTS_SUBDIRS = {zone: PROJECT_ROOT / zone / "results" for zone in ZONE_DIRS}
SELECTED_MODELS_FILE = PROJECT_ROOT / "streamlit_app" / ".selected_models.json"


@st.cache_data(show_spinner="Loading model metrics…")
def load_all_metrics() -> pd.DataFrame:
    """Load metrics from all zone directories."""
    dfs = []
    
    for zone, results_dir in RESULTS_SUBDIRS.items():
        metrics_file = results_dir / "model_registry_flat.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            df["zone_group"] = zone
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def get_metrics_columns(df: pd.DataFrame) -> List[str]:
    """Extract available metrics columns from the dataframe."""
    metrics = []
    possible_metrics = [
        "metrics_test_MAPE",
        "metrics_test_MAE", 
        "metrics_test_RMSE",
        "metrics_test_R2",
        "metrics_test_TotalAbsError",
        "metrics_test_Bias",
        "train_time_sec",
    ]
    
    for col in possible_metrics:
        if col in df.columns and not df[col].isna().all():
            metrics.append(col)
    
    return metrics


def format_metric_name(metric: str) -> str:
    """Format metric name for display."""
    return metric.replace("metrics_test_", "").replace("_", " ").title()


def shorten_model_name(model_name: str, max_length: int = 20) -> str:
    """
    Create a shortened version of model name for chart legends.
    
    E.g., nbeats_Total_1l_256w_ctx7d_20260319T151250Z -> Total_1l_256w_ctx7d
    """
    # Extract the main parts: zone, layers, width, context
    parts = model_name.split("_")
    if len(parts) < 4:
        return model_name[:max_length] if len(model_name) > max_length else model_name
    
    # Try to get: [zone, layers, width, context]
    try:
        zone = parts[1]  # e.g., "Total"
        layers = next((p for p in parts if "l" in p and p[0].isdigit()), "")  # e.g., "1l"
        width = next((p for p in parts if "w" in p and p[0].isdigit()), "")  # e.g., "256w"
        context = next((p for p in parts if "ctx" in p), "")  # e.g., "ctx7d"
        
        short = f"{zone}_{layers}_{width}_{context}"
        return short if len(short) <= max_length else f"{zone}_{layers}_{width}"
    except:
        return model_name[:max_length] if len(model_name) > max_length else model_name


def save_selected_models(selected_models: Dict[str, List[str]]) -> None:
    """Save selected models to a JSON file for use across pages."""
    SELECTED_MODELS_FILE.parent.mkdir(exist_ok=True)
    with open(SELECTED_MODELS_FILE, "w") as f:
        json.dump(selected_models, f, indent=2)


def get_model_details(df: pd.DataFrame, model_name: str) -> dict:
    """Extract full details for a specific model."""
    model_row = df[df["model_name"] == model_name]
    if model_row.empty:
        return {}
    
    model_row = model_row.iloc[0]
    return {
        "model_name": model_row.get("model_name", "N/A"),
        "zone": model_row.get("zone_group", "N/A"),
        "layers": model_row.get("config_num_layers", "N/A"),
        "layer_width": model_row.get("config_layer_widths", "N/A"),
        "context": model_row.get("config_input_chunk_length", "N/A"),
        "stacks": model_row.get("config_num_stacks", "N/A"),
        "batch_size": model_row.get("config_batch_size", "N/A"),
        "training_years": model_row.get("training_data_training_years", "N/A"),
        "training_months": model_row.get("training_data_training_months_present", "N/A"),
        "train_start": model_row.get("training_data_actual_start_timestamp", "N/A"),
        "train_end": model_row.get("training_data_actual_end_timestamp", "N/A"),
        "train_time": model_row.get("train_time_sec", "N/A"),
    }


def display_model_info(model_details: dict) -> None:
    """Display model information in a formatted way."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Layers", model_details.get("layers", "N/A"))
        st.metric("Layer Width", model_details.get("layer_width", "N/A"))
    
    with col2:
        st.metric("Context", model_details.get("context", "N/A"))
        st.metric("Stacks", model_details.get("stacks", "N/A"))
    
    with col3:
        st.metric("Batch Size", model_details.get("batch_size", "N/A"))
        st.metric("Train Time (s)", f"{model_details.get('train_time', 'N/A'):.1f}" if isinstance(model_details.get('train_time'), (int, float)) else "N/A")
    
    with col4:
        st.metric("Training Years", model_details.get("training_years", "N/A"))
        st.metric("Training Months", str(model_details.get("training_months", "N/A"))[:20])
    
    # Training date range
    st.caption(
        f"**Training Period**: {model_details.get('train_start', 'N/A')} → {model_details.get('train_end', 'N/A')}"
    )




def create_metrics_table(
    df: pd.DataFrame,
    metrics_cols: List[str],
    zone_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Create a formatted metrics comparison table with architecture and training info."""
    if zone_filter and zone_filter != "All Zones":
        df = df[df["zone_group"] == zone_filter]
    
    # Select columns for display - architecture, training data, and metrics
    display_cols = ["model_name", "zone_group"]
    
    # Architecture parameters
    arch_params = [
        "config_num_layers", 
        "config_layer_widths",
        "config_input_chunk_length",
        "config_num_stacks",
        "config_batch_size",
    ]
    
    # Training data info
    training_info = [
        "training_data_training_years",
        "training_data_training_months_present",
        "training_data_actual_start_timestamp",
        "training_data_actual_end_timestamp",
    ]
    
    # Add available columns
    for col in arch_params:
        if col in df.columns:
            display_cols.append(col)
    
    for col in training_info:
        if col in df.columns:
            display_cols.append(col)
    
    display_cols.extend([col for col in metrics_cols if col in df.columns])
    
    display_df = df[display_cols].copy()
    
    # Rename for readability
    rename_map = {
        "model_name": "Model",
        "zone_group": "Zone",
        "config_num_layers": "Layers",
        "config_layer_widths": "Layer Width",
        "config_input_chunk_length": "Context",
        "config_num_stacks": "Stacks",
        "config_batch_size": "Batch Size",
        "training_data_training_years": "Training Years",
        "training_data_training_months_present": "Training Months",
        "training_data_actual_start_timestamp": "Train Start",
        "training_data_actual_end_timestamp": "Train End",
    }
    for col in metrics_cols:
        rename_map[col] = format_metric_name(col)
    
    display_df = display_df.rename(columns=rename_map)
    
    return display_df.dropna(subset=[format_metric_name(m) for m in metrics_cols if m in display_cols], how="all")


def create_metric_comparison_chart(
    df: pd.DataFrame,
    metric: str,
    zone_filter: Optional[str] = None,
) -> go.Figure:
    """Create a bar chart comparing metric values across models."""
    if zone_filter and zone_filter != "All Zones":
        plot_df = df[df["zone_group"] == zone_filter].copy()
    else:
        plot_df = df.copy()
    
    plot_df = plot_df.dropna(subset=[metric])
    plot_df = plot_df.sort_values(metric, ascending="MAPE" in metric or "MAE" in metric or "Error" in metric)
    
    # Limit to top models to avoid overcrowding
    if len(plot_df) > 20:
        best_indices = (
            plot_df[metric].nsmallest(20).index 
            if "MAPE" in metric or "MAE" in metric or "Error" in metric or "Bias" in metric
            else plot_df[metric].nlargest(20).index
        )
        plot_df = plot_df.loc[best_indices]
    
    fig = go.Figure()
    
    # Color by zone
    colors = []
    color_map = {
        "TPCODL": "#636EFA",
        "TPWODL": "#00CC96",
        "TPNODL": "#AB63FA",
        "TPSOSDL": "#FFA15A",
        "Total": "#19D3F3",
    }
    for zone in plot_df["zone_group"]:
        colors.append(color_map.get(zone, "#FF6692"))
    
    # Add shortened names
    plot_df["short_name"] = plot_df["model_name"].apply(shorten_model_name)
    
    fig.add_trace(go.Bar(
        x=plot_df["short_name"],
        y=plot_df[metric],
        marker=dict(color=colors),
        text=plot_df[metric].round(3),
        textposition="outside",
        hovertext=plot_df["model_name"],  # Show full name on hover
        hoverinfo="text+y",
    ))
    
    fig.update_layout(
        title=f"{format_metric_name(metric)} Comparison",
        xaxis_title="Model (hover for full name)",
        yaxis_title=format_metric_name(metric),
        template="plotly_dark",
        height=500,
        xaxis_tickangle=-45,
        showlegend=False,
    )
    
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    metric_x: str,
    metric_y: str,
    zone_filter: Optional[str] = None,
) -> go.Figure:
    """Create a scatter plot comparing two metrics."""
    if zone_filter and zone_filter != "All Zones":
        plot_df = df[df["zone_group"] == zone_filter].copy()
    else:
        plot_df = df.copy()
    
    plot_df = plot_df.dropna(subset=[metric_x, metric_y])
    
    fig = px.scatter(
        plot_df,
        x=metric_x,
        y=metric_y,
        color="zone_group",
        hover_data=["model_name"],
        title=f"{format_metric_name(metric_x)} vs {format_metric_name(metric_y)}",
        template="plotly_dark",
    )
    
    fig.update_layout(height=500)
    
    return fig


def create_metrics_heatmap(
    df: pd.DataFrame,
    metrics_cols: List[str],
    zone_filter: Optional[str] = None,
    top_n: int = 15,
) -> go.Figure:
    """Create a heatmap of normalized metrics for top models."""
    if zone_filter and zone_filter != "All Zones":
        plot_df = df[df["zone_group"] == zone_filter].copy()
    else:
        plot_df = df.copy()
    
    # Select columns
    available_metrics = [m for m in metrics_cols if m in plot_df.columns]
    if not available_metrics:
        return go.Figure().add_annotation(text="No metrics data available")
    
    heatmap_df = plot_df[["model_name"] + available_metrics].dropna(subset=available_metrics, how="all").head(top_n)
    
    if heatmap_df.empty:
        return go.Figure().add_annotation(text="No data available for heatmap")
    
    # Add shortened names
    heatmap_df["short_name"] = heatmap_df["model_name"].apply(shorten_model_name)
    
    # Normalize metrics (0-1 scale)
    normalized = heatmap_df[available_metrics].copy()
    for col in available_metrics:
        min_val = normalized[col].min()
        max_val = normalized[col].max()
        if max_val > min_val:
            # For "better is lower" metrics, invert the scale
            if "MAPE" in col or "MAE" in col or "Error" in col or "Bias" in col:
                normalized[col] = 1 - (normalized[col] - min_val) / (max_val - min_val)
            else:
                normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
        else:
            normalized[col] = 0.5
    
    fig = go.Figure(data=go.Heatmap(
        z=normalized[available_metrics].values,
        x=[format_metric_name(m) for m in available_metrics],
        y=heatmap_df["short_name"].values,
        colorscale="RdYlGn",
        text=heatmap_df[available_metrics].values.round(3),
        texttemplate="%{text}",
        hovertext=heatmap_df["model_name"].values,
        hoverinfo="y+text",
        textfont={"size": 9},
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Models - Normalized Metrics",
        template="plotly_dark",
        height=500,
        xaxis_title="Metrics",
        yaxis_title="Model",
    )
    
    return fig


def get_best_models_by_metric(
    df: pd.DataFrame,
    metric: str,
    top_n: int = 5,
    zone_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Get top N models for a specific metric."""
    if zone_filter and zone_filter != "All Zones":
        plot_df = df[df["zone_group"] == zone_filter].copy()
    else:
        plot_df = df.copy()
    
    plot_df = plot_df.dropna(subset=[metric])
    
    # Lower is better for these metrics
    if "MAPE" in metric or "MAE" in metric or "Error" in metric or "Bias" in metric:
        best_df = plot_df.nsmallest(top_n, metric)
    else:  # Higher is better
        best_df = plot_df.nlargest(top_n, metric)
    
    return best_df[["model_name", "zone_group", metric]].copy()


def explore_metrics():
    """Main metrics explorer Streamlit app."""
    st.title("📊 Model Metrics Explorer")
    
    # Load data
    df = load_all_metrics()
    
    if df.empty:
        st.error("No metrics data found. Please ensure results directories contain model_registry_flat.csv files.")
        return
    
    metrics_cols = get_metrics_columns(df)
    
    if not metrics_cols:
        st.error("No metric columns found in data.")
        return
    
    # =====================================================================
    # SIDEBAR CONTROLS
    # =====================================================================
    st.sidebar.header("🎯 Filters & Selection")
    
    zone_filter = st.sidebar.selectbox(
        "Select Zone / Region",
        options=["All Zones"] + ZONE_DIRS,
        index=0,
    )
    
    metric_options = [format_metric_name(m) for m in metrics_cols]
    
    # Initialize session state for selected models
    if "marked_models" not in st.session_state:
        st.session_state.marked_models = load_selected_models()
    
    # =====================================================================
    # TAB STRUCTURE
    # =====================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Metrics Overview",
        "🔍 Metric Comparison",
        "📊 Advanced Analysis",
        "✅ Model Selection"
    ])
    
    # =====================================================================
    # TAB 1: METRICS OVERVIEW
    # =====================================================================
    with tab1:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader("📊 Model Metrics Summary")
        
        with col2:
            if st.button("✅ Mark All", key="mark_all_visible", use_container_width=True):
                if zone_filter not in st.session_state.marked_models:
                    st.session_state.marked_models[zone_filter] = []
                
                if zone_filter != "All Zones":
                    visible_models = df[df["zone_group"] == zone_filter]["model_name"].unique().tolist()
                else:
                    visible_models = df["model_name"].unique().tolist()
                
                for model in visible_models:
                    if model not in st.session_state.marked_models[zone_filter]:
                        st.session_state.marked_models[zone_filter].append(model)
                st.success(f"✅ Marked {len(visible_models)} model(s)")
                st.rerun()
        
        with col3:
            if st.button("🔄 Refresh", key="refresh_metrics", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        # Model selection checkboxes
        st.write("**🎯 Select models to compare:**")
        if zone_filter != "All Zones":
            visible_models = df[df["zone_group"] == zone_filter]["model_name"].unique().tolist()
        else:
            visible_models = df["model_name"].unique().tolist()
        
        if visible_models:
            cols_per_row = 5
            for i in range(0, len(visible_models), cols_per_row):
                row_models = visible_models[i:i+cols_per_row]
                cols = st.columns(len(row_models))
                
                for col_idx, model in enumerate(row_models):
                    with cols[col_idx]:
                        is_marked = zone_filter in st.session_state.marked_models and model in st.session_state.marked_models[zone_filter]
                        
                        if st.checkbox(
                            shorten_model_name(model),
                            value=is_marked,
                            key=f"chk_{zone_filter}_{model}",
                        ):
                            if zone_filter not in st.session_state.marked_models:
                                st.session_state.marked_models[zone_filter] = []
                            if model not in st.session_state.marked_models[zone_filter]:
                                st.session_state.marked_models[zone_filter].append(model)
                        else:
                            if zone_filter in st.session_state.marked_models and model in st.session_state.marked_models[zone_filter]:
                                st.session_state.marked_models[zone_filter].remove(model)
                                if not st.session_state.marked_models[zone_filter]:
                                    del st.session_state.marked_models[zone_filter]
        
        # Save marked models
        if st.button("💾 Save Marked Models", key="save_marks_overview", type="primary", use_container_width=True):
            save_selected_models(st.session_state.marked_models)
            st.success("✅ Marked models saved!")
        
        st.divider()
        st.write("**📋 Detailed Metrics:**")
        
        # Display metrics table
        metrics_table = create_metrics_table(df, metrics_cols, zone_filter)
        
        st.dataframe(
            metrics_table,
            use_container_width=True,
            height=400,
        )
        
        # Model details viewer
        st.divider()
        st.subheader("🔍 Model Details")
        
        if zone_filter != "All Zones":
            filtered_df = df[df["zone_group"] == zone_filter]
        else:
            filtered_df = df
        
        available_models = filtered_df["model_name"].unique().tolist()
        
        selected_model = st.selectbox(
            "Select a model to view details",
            options=available_models,
            format_func=lambda x: shorten_model_name(x),
            key="detail_model_select",
        )
        
        if selected_model:
            model_details = get_model_details(df, selected_model)
            if model_details:
                st.caption(f"**Full model name**: `{selected_model}`")
                display_model_info(model_details)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Models", len(df))
        with col2:
            st.metric("Zones", df["zone_group"].nunique())
        with col3:
            st.metric("Metrics Tracked", len(metrics_cols))
    
    # =====================================================================
    # TAB 2: METRIC COMPARISON
    # =====================================================================
    with tab2:
        st.subheader("Single Metric Comparison")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_metric = st.selectbox(
                "Select metric to visualize",
                options=metric_options,
                key="metric_select",
            )
        with col2:
            show_best = st.number_input("Show Top N", value=20, min_value=5, max_value=len(df))
        
        # Convert display name back to column name
        metric_col = next(
            (m for m in metrics_cols if format_metric_name(m) == selected_metric),
            metrics_cols[0]
        )
        
        fig = create_metric_comparison_chart(df, metric_col, zone_filter)
        st.plotly_chart(fig, use_container_width=True)
        
        # Best models for this metric
        with st.expander("📌 Best Models for this Metric"):
            best_models = get_best_models_by_metric(df, metric_col, top_n=5, zone_filter=zone_filter)
            st.dataframe(best_models, use_container_width=True)
    
    # =====================================================================
    # TAB 3: ADVANCED ANALYSIS
    # =====================================================================
    with tab3:
        st.subheader("Advanced Metric Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Scatter Plot: Two Metrics**")
            metric_x_name = st.selectbox("X-axis metric", options=metric_options, key="x_metric")
            metric_x = next((m for m in metrics_cols if format_metric_name(m) == metric_x_name), metrics_cols[0])
        
        with col2:
            st.write("**Scatter Plot: Two Metrics**")
            metric_y_name = st.selectbox("Y-axis metric", options=metric_options, key="y_metric", index=1 if len(metric_options) > 1 else 0)
            metric_y = next((m for m in metrics_cols if format_metric_name(m) == metric_y_name), metrics_cols[0])
        
        fig_scatter = create_scatter_plot(df, metric_x, metric_y, zone_filter)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Heatmap
        st.write("**Performance Heatmap**")
        top_n_heatmap = st.slider("Include Top N models", min_value=5, max_value=min(30, len(df)), value=15)
        fig_heatmap = create_metrics_heatmap(df, metrics_cols, zone_filter, top_n=top_n_heatmap)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # =====================================================================
    # TAB 4: MODEL SELECTION & MARKING
    # =====================================================================
    with tab4:
        st.subheader("✅ Mark Models for Use")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("Select and mark the best models to use across other pages.")
        
        with col2:
            if st.button("💾 Save Marked Models", type="primary"):
                save_selected_models(st.session_state.marked_models)
                st.success("✅ Models saved!")
        
        # Get filtered model list
        if zone_filter != "All Zones":
            filtered_df = df[df["zone_group"] == zone_filter]
        else:
            filtered_df = df
        
        available_models = filtered_df["model_name"].unique().tolist()
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Available Models**")
            models_to_add = st.multiselect(
                "Select models to mark",
                options=available_models,
                key="models_to_mark",
            )
            
            if st.button("➕ Add Selected Models", use_container_width=True):
                if models_to_add:
                    for model in models_to_add:
                        if zone_filter not in st.session_state.marked_models:
                            st.session_state.marked_models[zone_filter] = []
                        if model not in st.session_state.marked_models[zone_filter]:
                            st.session_state.marked_models[zone_filter].append(model)
                    st.success(f"✅ Added {len(models_to_add)} model(s)")
                    st.rerun()
        
        with col2:
            st.write("**Marked Models**")
            
            marked_by_zone = st.session_state.marked_models
            
            if marked_by_zone:
                for zone_name, models in marked_by_zone.items():
                    with st.expander(f"📍 {zone_name} ({len(models)} models)", expanded=True):
                        for i, model in enumerate(models):
                            inner_col1, inner_col2 = st.columns([4, 1])
                            with inner_col1:
                                if st.button(f"ℹ️ {shorten_model_name(model)}", key=f"expand_{zone_name}_{i}"):
                                    st.session_state[f"expand_{zone_name}_{i}"] = not st.session_state.get(f"expand_{zone_name}_{i}", False)
                            with inner_col2:
                                if st.button("❌", key=f"remove_{zone_name}_{i}", use_container_width=True):
                                    st.session_state.marked_models[zone_name].remove(model)
                                    if not st.session_state.marked_models[zone_name]:
                                        del st.session_state.marked_models[zone_name]
                                    st.rerun()
                            
                            # Show details if expanded
                            if st.session_state.get(f"expand_{zone_name}_{i}", False):
                                with st.container():
                                    st.caption(f"**Full name**: {model}")
                                    model_details = get_model_details(df, model)
                                    if model_details:
                                        display_model_info(model_details)
                                    st.divider()
            else:
                st.info("No models marked yet. Select models from the left to get started!")
        
        # Summary and quick access
        st.divider()
        st.write("**📊 Summary**")
        total_marked = sum(len(models) for models in marked_by_zone.values())
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Marked", total_marked)
        with col_s2:
            st.metric("Zones", len(marked_by_zone))
        with col_s3:
            if st.button("📋 Copy to Clipboard", use_container_width=True, key="copy_models"):
                models_str = "\n".join([f"{z}: " + ", ".join([shorten_model_name(m) for m in models]) for z, models in marked_by_zone.items()])
                st.info(f"Marked models:\n{models_str}")
        
        st.divider()
        st.write("**💡 Tip**: Use marked models on the Model Comparison or Seasonal Blending pages")
