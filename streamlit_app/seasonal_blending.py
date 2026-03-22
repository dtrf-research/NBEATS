"""Seasonal model blending: schedule validation, hard/soft merge, segment metrics."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from inference import compute_metrics


# ---------------------------------------------------------------------------
# Schedule validation
# ---------------------------------------------------------------------------

def validate_schedule(
    schedule: List[Dict[str, Any]],
    available_models: List[str],
    date_min: datetime,
    date_max: datetime,
) -> List[Dict[str, str]]:
    """Return list of warning dicts ``{"type": ..., "message": ...}``."""
    warnings: List[Dict[str, str]] = []
    if not schedule:
        return warnings

    for idx, entry in enumerate(schedule):
        s, e = pd.Timestamp(entry["start"]), pd.Timestamp(entry["end"])
        if s >= e:
            warnings.append({"type": "invalid_range", "message": f"Entry {idx + 1}: start >= end."})
        if entry["model"] not in available_models:
            warnings.append({"type": "missing_model", "message": f"Entry {idx + 1}: model '{entry['model']}' not loaded."})

    # Detect overlaps (later entry wins, but warn)
    sorted_entries = sorted(enumerate(schedule), key=lambda t: pd.Timestamp(t[1]["start"]))
    for i in range(len(sorted_entries) - 1):
        _, cur = sorted_entries[i]
        _, nxt = sorted_entries[i + 1]
        if pd.Timestamp(cur["end"]) > pd.Timestamp(nxt["start"]):
            warnings.append({
                "type": "overlap",
                "message": f"Overlap between entries ending {cur['end']} and starting {nxt['start']}. Later entry overrides.",
            })

    # Detect gaps
    for i in range(len(sorted_entries) - 1):
        _, cur = sorted_entries[i]
        _, nxt = sorted_entries[i + 1]
        if pd.Timestamp(cur["end"]) < pd.Timestamp(nxt["start"]):
            warnings.append({
                "type": "gap",
                "message": f"Gap from {cur['end']} to {nxt['start']}. Fallback model will be used.",
            })

    return warnings


# ---------------------------------------------------------------------------
# Hard merge
# ---------------------------------------------------------------------------

def apply_hard_merge(
    predictions_dict: Dict[str, pd.DataFrame],
    schedule: List[Dict[str, Any]],
    fallback_model: str,
) -> pd.DataFrame:
    """
    Build a single merged prediction series using timestamp-level assignment.

    Parameters
    ----------
    predictions_dict : {model_name: DataFrame[Timestamp, Actual, Predicted]}
    schedule : list of {start, end, model} dicts — later entries win on overlap
    fallback_model : model to use for timestamps not covered by any entry

    Returns
    -------
    DataFrame[Timestamp, Actual, Predicted, Source] aligned with existing format.
    """
    # Collect all unique prediction timestamps with actuals from fallback
    if fallback_model not in predictions_dict:
        raise ValueError(f"Fallback model '{fallback_model}' has no predictions.")

    base_df = predictions_dict[fallback_model][["Timestamp", "Actual"]].copy()
    base_df = base_df.sort_values("Timestamp").reset_index(drop=True)

    merged_pred = np.full(len(base_df), np.nan)
    source = np.full(len(base_df), fallback_model, dtype=object)
    timestamps = base_df["Timestamp"].values

    # Apply schedule entries in order — later entries override earlier ones
    for entry in schedule:
        model_name = entry["model"]
        if model_name not in predictions_dict:
            continue
        s = pd.Timestamp(entry["start"])
        e = pd.Timestamp(entry["end"])
        mask = (timestamps >= s.to_numpy()) & (timestamps < e.to_numpy())
        if not mask.any():
            continue

        model_pred = predictions_dict[model_name].set_index("Timestamp")["Predicted"]
        for i in np.where(mask)[0]:
            ts = pd.Timestamp(timestamps[i])
            if ts in model_pred.index:
                merged_pred[i] = model_pred[ts]
                source[i] = model_name

    # Fill remaining NaN with fallback model
    fallback_pred = predictions_dict[fallback_model].set_index("Timestamp")["Predicted"]
    nan_mask = np.isnan(merged_pred)
    for i in np.where(nan_mask)[0]:
        ts = pd.Timestamp(timestamps[i])
        if ts in fallback_pred.index:
            merged_pred[i] = fallback_pred[ts]
            source[i] = fallback_model

    result = base_df.copy()
    result["Predicted"] = merged_pred
    result["Source"] = source
    return result.dropna(subset=["Predicted"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Soft transition merge
# ---------------------------------------------------------------------------

def apply_soft_transition(
    predictions_dict: Dict[str, pd.DataFrame],
    schedule: List[Dict[str, Any]],
    fallback_model: str,
    transition_days: int = 0,
) -> pd.DataFrame:
    """
    Like hard merge but with linear blending at segment boundaries.

    At each boundary between consecutive schedule entries the transition
    window is ``transition_days`` days.  Within that window the outgoing
    model weight decays linearly 1 → 0 and the incoming model ramps 0 → 1.
    """
    if transition_days <= 0:
        return apply_hard_merge(predictions_dict, schedule, fallback_model)

    # Start with hard merge to get base structure
    hard = apply_hard_merge(predictions_dict, schedule, fallback_model)
    timestamps = hard["Timestamp"].values
    predicted = hard["Predicted"].values.copy()
    source_arr = hard["Source"].values.copy()

    td = timedelta(days=transition_days)
    sorted_sched = sorted(schedule, key=lambda e: pd.Timestamp(e["start"]))

    for i in range(len(sorted_sched) - 1):
        cur_entry = sorted_sched[i]
        nxt_entry = sorted_sched[i + 1]
        boundary = pd.Timestamp(nxt_entry["start"])

        cur_model = cur_entry["model"]
        nxt_model = nxt_entry["model"]
        if cur_model == nxt_model:
            continue
        if cur_model not in predictions_dict or nxt_model not in predictions_dict:
            continue

        win_start = boundary - td / 2
        win_end = boundary + td / 2

        # Clamp to segment bounds
        seg_cur_start = pd.Timestamp(cur_entry["start"])
        seg_nxt_end = pd.Timestamp(nxt_entry["end"])
        win_start = max(win_start, seg_cur_start)
        win_end = min(win_end, seg_nxt_end)
        if win_start >= win_end:
            continue

        cur_pred = predictions_dict[cur_model].set_index("Timestamp")["Predicted"]
        nxt_pred = predictions_dict[nxt_model].set_index("Timestamp")["Predicted"]

        win_mask = (timestamps >= win_start.to_numpy()) & (timestamps < win_end.to_numpy())
        total_dur = (win_end - win_start).total_seconds()
        if total_dur <= 0:
            continue

        for j in np.where(win_mask)[0]:
            ts = pd.Timestamp(timestamps[j])
            elapsed = (ts - win_start).total_seconds()
            w_nxt = elapsed / total_dur  # 0 → 1
            w_cur = 1.0 - w_nxt
            val_cur = cur_pred.get(ts, np.nan)
            val_nxt = nxt_pred.get(ts, np.nan)
            if np.isnan(val_cur) or np.isnan(val_nxt):
                continue
            predicted[j] = w_cur * val_cur + w_nxt * val_nxt
            source_arr[j] = f"blend({cur_model},{nxt_model})"

    result = hard.copy()
    result["Predicted"] = predicted
    result["Source"] = source_arr
    return result


# ---------------------------------------------------------------------------
# Per-segment metrics
# ---------------------------------------------------------------------------

def compute_segment_metrics(
    merged_df: pd.DataFrame,
    schedule: List[Dict[str, Any]],
    fallback_model: str,
) -> pd.DataFrame:
    """Compute MAE/RMSE/MAPE/R² for each schedule segment + fallback gaps."""
    rows = []
    for idx, entry in enumerate(schedule):
        s = pd.Timestamp(entry["start"])
        e = pd.Timestamp(entry["end"])
        seg = merged_df[(merged_df["Timestamp"] >= s) & (merged_df["Timestamp"] < e)]
        if seg.empty:
            continue
        m = compute_metrics(seg["Actual"].values, seg["Predicted"].values)
        m["Segment"] = f"{idx + 1}: {entry['model']}"
        m["Start"] = str(s.date())
        m["End"] = str(e.date())
        m["Points"] = len(seg)
        rows.append(m)

    # Overall
    if not merged_df.empty:
        m = compute_metrics(merged_df["Actual"].values, merged_df["Predicted"].values)
        m["Segment"] = "Overall (merged)"
        m["Start"] = str(merged_df["Timestamp"].min().date())
        m["End"] = str(merged_df["Timestamp"].max().date())
        m["Points"] = len(merged_df)
        rows.append(m)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = ["Segment", "Start", "End", "Points", "MAPE (%)", "MAE", "RMSE", "R²"]
    return df[[c for c in cols if c in df.columns]]


# ---------------------------------------------------------------------------
# Model contribution summary
# ---------------------------------------------------------------------------

def compute_contribution_summary(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Count how many timestamps each source model contributed."""
    if merged_df.empty or "Source" not in merged_df.columns:
        return pd.DataFrame()
    counts = merged_df["Source"].value_counts().reset_index()
    counts.columns = ["Model / Source", "Timestamps"]
    total = counts["Timestamps"].sum()
    counts["Coverage (%)"] = (counts["Timestamps"] / total * 100).round(2)
    return counts.sort_values("Timestamps", ascending=False).reset_index(drop=True)
