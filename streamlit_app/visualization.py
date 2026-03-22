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


# ---------------------------------------------------------------------------
# Seasonal Blending charts
# ---------------------------------------------------------------------------

_SEGMENT_COLORS = [
    "rgba(99,110,250,0.10)",   # blue
    "rgba(0,204,150,0.10)",    # green
    "rgba(171,99,250,0.10)",   # purple
    "rgba(255,161,90,0.10)",   # orange
    "rgba(25,211,243,0.10)",   # cyan
    "rgba(255,102,146,0.10)",  # pink
]

_TRANSITION_COLOR = "rgba(255,215,0,0.18)"  # gold for blend zones


def build_blending_chart(
    actual_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    individual_preds: Dict[str, pd.DataFrame],
    zone: str,
    schedule: List[dict],
    show_individuals: bool = True,
    transition_days: int = 0,
) -> go.Figure:
    """Actual + merged prediction with segment background shading."""
    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=actual_df["Timestamp"],
        y=actual_df["Actual"],
        mode="lines",
        name="Actual",
        line=dict(color="black", width=2),
    ))

    # Merged prediction
    fig.add_trace(go.Scatter(
        x=merged_df["Timestamp"],
        y=merged_df["Predicted"],
        mode="lines",
        name="Merged Prediction",
        line=dict(color="#EF553B", width=2.5),
    ))

    # Individual model predictions (optional, dotted)
    if show_individuals:
        for idx, (model_name, pred_df) in enumerate(individual_preds.items()):
            short_name = _short_label(model_name)
            color = _COLORS[(idx + 2) % len(_COLORS)]
            fig.add_trace(go.Scatter(
                x=pred_df["Timestamp"],
                y=pred_df["Predicted"],
                mode="lines",
                name=short_name,
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.55,
            ))

    # Segment background shading
    for idx, entry in enumerate(schedule):
        seg_color = _SEGMENT_COLORS[idx % len(_SEGMENT_COLORS)]
        fig.add_vrect(
            x0=entry["start"], x1=entry["end"],
            fillcolor=seg_color,
            layer="below",
            line_width=0,
            annotation_text=_short_label(entry["model"]),
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color="gray",
        )

    # Transition zone highlighting
    if transition_days > 0:
        from datetime import timedelta
        sorted_sched = sorted(schedule, key=lambda e: pd.Timestamp(e["start"]))
        td = timedelta(days=transition_days)
        for i in range(len(sorted_sched) - 1):
            boundary = pd.Timestamp(sorted_sched[i + 1]["start"])
            fig.add_vrect(
                x0=boundary - td / 2, x1=boundary + td / 2,
                fillcolor=_TRANSITION_COLOR,
                layer="below",
                line=dict(width=1, dash="dash", color="goldenrod"),
            )

    fig.update_layout(
        title=f"Seasonal Blending — {zone}",
        xaxis_title="Timestamp",
        yaxis_title="Demand",
        hovermode="x unified",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_blending_error_chart(
    merged_df: pd.DataFrame,
) -> go.Figure:
    """Prediction error chart for the merged blended series."""
    fig = go.Figure()
    if "Actual" in merged_df.columns and "Predicted" in merged_df.columns:
        error = merged_df["Predicted"] - merged_df["Actual"]
        fig.add_trace(go.Scatter(
            x=merged_df["Timestamp"],
            y=error,
            mode="lines",
            name="Merged Error",
            line=dict(color="#EF553B", width=1),
        ))
    fig.update_layout(
        title="Blended Prediction Error Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Error (Predicted − Actual)",
        hovermode="x unified",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
