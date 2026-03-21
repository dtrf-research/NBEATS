"""Inference: run predictions for loaded models on filtered data."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import MissingValuesFiller

from data_loader import TIME_COLUMN

OUTPUT_CHUNK_LENGTH = 96  # 1 day = 96 × 15-min steps


def _make_series(df: pd.DataFrame, zone: str) -> TimeSeries:
    """Convert 2-column filtered DataFrame to a Darts TimeSeries."""
    filler = MissingValuesFiller()
    series = TimeSeries.from_dataframe(
        df,
        time_col=TIME_COLUMN,
        value_cols=[zone],
        freq="15min",
    )
    return filler.transform(series)


def run_inference(
    model,
    scaler,
    df_filtered: pd.DataFrame,
    zone: str,
) -> pd.DataFrame:
    """
    Generate predictions for a single model on the filtered dataset.

    Returns a DataFrame with [Timestamp, Actual, Predicted].
    """
    series = _make_series(df_filtered, zone)
    scaled_series = scaler.transform(series)

    preds_scaled = model.historical_forecasts(
        scaled_series,
        forecast_horizon=OUTPUT_CHUNK_LENGTH,
        stride=OUTPUT_CHUNK_LENGTH,
        last_points_only=False,
        retrain=False,
        verbose=False,
    )

    if not preds_scaled:
        return pd.DataFrame(columns=["Timestamp", "Actual", "Predicted"])

    preds_scaled = concatenate(preds_scaled)
    preds_unscaled = scaler.inverse_transform(preds_scaled)

    pred_df = preds_unscaled.to_dataframe().reset_index()
    pred_df = pred_df.rename(
        columns={pred_df.columns[0]: "Timestamp", pred_df.columns[1]: "Predicted"}
    )
    pred_df = pred_df[["Timestamp", "Predicted"]]
    pred_df["Timestamp"] = pd.to_datetime(pred_df["Timestamp"])

    # Align actuals to the prediction timestamps
    actual_df = df_filtered[[TIME_COLUMN, zone]].rename(
        columns={TIME_COLUMN: "Timestamp", zone: "Actual"}
    )
    merged = pd.merge(pred_df, actual_df, on="Timestamp", how="inner")
    return merged


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Regression metrics dict."""
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / (np.abs(y_true) + 1e-8))) * 100
    r2 = 1 - np.sum(errors**2) / (np.sum((y_true - y_true.mean()) ** 2) + 1e-8)
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R²": r2}
