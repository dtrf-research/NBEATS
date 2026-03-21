"""Plotly-based visualization helpers."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go


# Distinct colour palette for model traces
_COLORS = [
    "#EF553B", "#636EFA", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def build_comparison_chart(
    actual_df: pd.DataFrame,
    predictions: Dict[str, pd.DataFrame],
    zone: str,
) -> go.Figure:
    """
    Build a single Plotly figure with actual values and one line per model.

    Parameters
    ----------
    actual_df : DataFrame with [Timestamp, Actual]
    predictions : {model_name: DataFrame with [Timestamp, Predicted]}
    zone : zone label for chart title
    """
    fig = go.Figure()

    # Actual line
    fig.add_trace(go.Scatter(
        x=actual_df["Timestamp"],
        y=actual_df["Actual"],
        mode="lines",
        name="Actual",
        line=dict(color="black", width=2),
    ))

    # One trace per model
    for idx, (model_name, pred_df) in enumerate(predictions.items()):
        short_name = _short_label(model_name)
        color = _COLORS[idx % len(_COLORS)]
        fig.add_trace(go.Scatter(
            x=pred_df["Timestamp"],
            y=pred_df["Predicted"],
            mode="lines",
            name=short_name,
            line=dict(color=color, width=1.5, dash="dot"),
        ))

    fig.update_layout(
        title=f"Actual vs Predictions — {zone}",
        xaxis_title="Timestamp",
        yaxis_title="Demand",
        hovermode="x unified",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_error_chart(
    predictions: Dict[str, pd.DataFrame],
) -> go.Figure:
    """Build error (Predicted - Actual) chart for each model."""
    fig = go.Figure()
    for idx, (model_name, pred_df) in enumerate(predictions.items()):
        if "Actual" not in pred_df.columns:
            continue
        error = pred_df["Predicted"] - pred_df["Actual"]
        short_name = _short_label(model_name)
        color = _COLORS[idx % len(_COLORS)]
        fig.add_trace(go.Scatter(
            x=pred_df["Timestamp"],
            y=error,
            mode="lines",
            name=short_name,
            line=dict(color=color, width=1),
        ))

    fig.update_layout(
        title="Prediction Error Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Error (Predicted − Actual)",
        hovermode="x unified",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _short_label(name: str) -> str:
    """Shorten a model name for legend readability."""
    # Drop the timestamp suffix and 'nbeats_' prefix
    parts = name.split("_")
    # Keep meaningful architecture tokens
    return "_".join(parts[1:-1]) if len(parts) > 2 else name
