from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.metrics import r2_score

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from darts.models import NBEATSModel

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "cleaned_demand_data.csv"
REGISTRY_PATH = BASE_DIR / "models" / "model_registry.json"
MODELS_DIR = BASE_DIR / "models"
DARTS_LOGS_DIR = BASE_DIR / "darts_logs"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ZONE = "Total Demand"
TEST_START_DATE = "2025-01-01"
TEST_END_DATE = "2026-01-14"
OUTPUT_CHUNK_LENGTH = 96
STRIDE = 96
MAX_PLOT_POINTS = 5000
AUTO_DISCOVER_MONTH_GROUPS = True


def load_registry(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    if "Total Demand (as recorded)" in df.columns and "Total Demand" not in df.columns:
        df = df.rename(columns={"Total Demand (as recorded)": "Total Demand"})
    return df.sort_values("Timestamp").reset_index(drop=True)


def normalize_month_group(value):
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        value = [int(x) for x in value.split("|") if str(x).strip()]
    return tuple(sorted(int(x) for x in value))


def month_group_label(months) -> str:
    months = normalize_month_group(months)
    if not months:
        return "all_months"
    return "months_" + "_".join(str(m) for m in months)


def extract_requested_months(meta: dict):
    value = meta.get("requested_months")
    if isinstance(value, list) and value:
        return value
    for section in ("training_data", "config"):
        nested = meta.get(section, {})
        if isinstance(nested, dict):
            value = nested.get("requested_months")
            if isinstance(value, list) and value:
                return value
    return None


def filter_by_zone_and_months(
    df: pd.DataFrame,
    zone: str,
    start_date: str,
    end_date: str,
    months=None,
) -> pd.DataFrame:
    df_filtered = df[(df["Timestamp"] >= start_date) & (df["Timestamp"] <= end_date)].copy()
    if months:
        df_filtered = df_filtered[df_filtered["Timestamp"].dt.month.isin(list(months))]
    return df_filtered[["Timestamp", zone]].copy()


def prepare_test_series(df: pd.DataFrame, target_col: str, scaler: Scaler) -> TimeSeries:
    series = TimeSeries.from_dataframe(df, time_col="Timestamp", value_cols=[target_col], freq="15min")
    filler = MissingValuesFiller()
    series = filler.transform(series)
    series = scaler.transform(series)
    return series


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str = "Model") -> dict:
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs(errors / (np.abs(y_true) + 1e-8))) * 100)
    mae_pct = float((mae / (np.abs(y_true).mean() + 1e-8)) * 100)
    r2 = float(r2_score(y_true, y_pred))
    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "MAE%": mae_pct,
        "R2": r2,
        "TotalAbsError": float(np.abs(errors).sum()),
        "Bias": float(errors.mean()),
        "n_points": int(len(y_true)),
    }


def safe_load_checkpoint(model_name: str):
    torch.serialization.add_safe_globals([torch.optim.Adam])
    try:
        return NBEATSModel.load_from_checkpoint(model_name=model_name, best=True, work_dir=str(DARTS_LOGS_DIR))
    except TypeError:
        return NBEATSModel.load_from_checkpoint(model_name=model_name, best=True, work_dir=str(DARTS_LOGS_DIR))


def load_model_and_scaler(model_name: str):
    model = safe_load_checkpoint(model_name)
    scaler_path = MODELS_DIR / model_name / "scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    return model, scaler


def prediction_df_for_model(model_name: str, zone: str, start_date: str, end_date: str, months=None) -> pd.DataFrame:
    model, scaler = load_model_and_scaler(model_name)
    df_eval = filter_by_zone_and_months(df_full, zone, start_date, end_date, months)
    if df_eval.empty:
        return pd.DataFrame(columns=["Timestamp", "Actual", "Predicted"])

    series_scaled = prepare_test_series(df_eval, zone, scaler)
    preds_scaled = model.historical_forecasts(
        series_scaled,
        forecast_horizon=OUTPUT_CHUNK_LENGTH,
        stride=STRIDE,
        last_points_only=False,
        retrain=False,
        verbose=False,
    )
    if not preds_scaled:
        return pd.DataFrame(columns=["Timestamp", "Actual", "Predicted"])

    preds_scaled = concatenate(preds_scaled)
    preds_unscaled = scaler.inverse_transform(preds_scaled)
    preds_df = preds_unscaled.to_dataframe().reset_index()
    preds_df = preds_df.rename(columns={preds_df.columns[0]: "Timestamp", preds_df.columns[1]: "Predicted"})
    preds_df["Timestamp"] = pd.to_datetime(preds_df["Timestamp"])

    actual_df = df_eval.rename(columns={zone: "Actual"}).copy()
    actual_df["Timestamp"] = pd.to_datetime(actual_df["Timestamp"])

    merged = pd.merge(actual_df, preds_df[["Timestamp", "Predicted"]], on="Timestamp", how="inner").dropna()
    return merged.sort_values("Timestamp").reset_index(drop=True)


def build_selection_table(registry: dict, zone: str) -> pd.DataFrame:
    rows = []
    for model_name, meta in registry.items():
        if meta.get("zone") != zone:
            continue
        metrics = meta.get("metrics", {})
        requested_months = extract_requested_months(meta)
        rows.append(
            {
                "model_name": model_name,
                "month_group": normalize_month_group(requested_months),
                "month_group_label": month_group_label(requested_months),
                "test_MAPE": metrics.get("test_MAPE"),
                "test_MAE": metrics.get("test_MAE"),
                "test_R2": metrics.get("test_R2"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No registry entries found for zone: {zone}")

    for col in ["test_MAPE", "test_MAE", "test_R2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pick_best_models_per_group(selection_df: pd.DataFrame) -> pd.DataFrame:
    seasonal_df = selection_df[selection_df["month_group"].notna()].copy()
    seasonal_df = seasonal_df[seasonal_df["test_MAPE"].notna()].copy()
    if seasonal_df.empty:
        raise ValueError("No seasonal models with stored test metrics were found.")

    if AUTO_DISCOVER_MONTH_GROUPS:
        groups = sorted(seasonal_df["month_group"].dropna().unique().tolist())
    else:
        groups = [
            (1, 11, 12),
            (2, 3),
            (4, 5, 6),
            (7, 8, 9, 10),
        ]

    best_rows = []
    for group in groups:
        group_df = seasonal_df[seasonal_df["month_group"] == group].sort_values("test_MAPE", ascending=True)
        if group_df.empty:
            continue
        best_rows.append(group_df.iloc[0])

    best_df = pd.DataFrame(best_rows).reset_index(drop=True)
    if best_df.empty:
        raise ValueError("No best seasonal models could be selected.")
    return best_df


def pick_best_global_model(selection_df: pd.DataFrame):
    global_df = selection_df[selection_df["month_group"].isna()].copy()
    global_df = global_df[global_df["test_MAPE"].notna()].sort_values("test_MAPE", ascending=True)
    if global_df.empty:
        return None
    return global_df.iloc[0]


def evaluate_model_row(row: pd.Series, zone: str, start_date: str, end_date: str):
    months = row["month_group"] if row["month_group_label"] != "all_months" else None
    pred_df = prediction_df_for_model(row["model_name"], zone, start_date, end_date, months=months)
    if pred_df.empty:
        metrics = {
            "Model": row["model_name"],
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "MAE%": np.nan,
            "R2": np.nan,
            "TotalAbsError": np.nan,
            "Bias": np.nan,
            "n_points": 0,
        }
    else:
        metrics = compute_metrics(
            pred_df["Actual"].to_numpy(),
            pred_df["Predicted"].to_numpy(),
            name=row["model_name"],
        )
    return pred_df, metrics


def stitch_seasonal_predictions(best_models_df: pd.DataFrame, zone: str, start_date: str, end_date: str):
    stitched_parts = []
    group_metrics = []

    for _, row in best_models_df.iterrows():
        pred_df, metrics = evaluate_model_row(row, zone, start_date, end_date)
        pred_df["month_group_label"] = row["month_group_label"]
        pred_df["model_name"] = row["model_name"]
        stitched_parts.append(pred_df)

        metrics["month_group_label"] = row["month_group_label"]
        group_metrics.append(metrics)

    stitched_df = pd.concat(stitched_parts, ignore_index=True)
    stitched_df = stitched_df.sort_values("Timestamp").reset_index(drop=True)
    group_metrics_df = pd.DataFrame(group_metrics).sort_values("month_group_label").reset_index(drop=True)

    overall_metrics = compute_metrics(
        stitched_df["Actual"].to_numpy(),
        stitched_df["Predicted"].to_numpy(),
        name="SeasonalBestModels",
    )

    return stitched_df, overall_metrics, group_metrics_df


def plot_predictions(seasonal_df: pd.DataFrame, baseline_df: pd.DataFrame | None = None):
    plot_df = seasonal_df.copy()
    if len(plot_df) > MAX_PLOT_POINTS:
        step = max(1, len(plot_df) // MAX_PLOT_POINTS)
        plot_df = plot_df.iloc[::step].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["Timestamp"], y=plot_df["Actual"], mode="lines", name="Actual"))
    fig.add_trace(
        go.Scatter(
            x=plot_df["Timestamp"],
            y=plot_df["Predicted"],
            mode="lines",
            name="Seasonal Best Models",
        )
    )

    if baseline_df is not None and not baseline_df.empty:
        baseline_plot_df = baseline_df.copy()
        if len(baseline_plot_df) > MAX_PLOT_POINTS:
            step = max(1, len(baseline_plot_df) // MAX_PLOT_POINTS)
            baseline_plot_df = baseline_plot_df.iloc[::step].copy()
        fig.add_trace(
            go.Scatter(
                x=baseline_plot_df["Timestamp"],
                y=baseline_plot_df["Predicted"],
                mode="lines",
                name="Best Global Model",
            )
        )

    fig.update_layout(
        title=f"{ZONE}: Seasonal Composite vs Actuals",
        xaxis_title="Timestamp",
        yaxis_title=ZONE,
        template="plotly_white",
        legend_title="Series",
        height=650,
    )
    fig.show()


if __name__ == "__main__":
    print(f"Notebook/script directory: {BASE_DIR}")
    print(f"Zone: {ZONE}")
    print(f"Evaluation window: {TEST_START_DATE} to {TEST_END_DATE}")

    registry = load_registry(REGISTRY_PATH)
    df_full = load_data(DATA_PATH)
    selection_df = build_selection_table(registry, ZONE)
    best_seasonal_models_df = pick_best_models_per_group(selection_df)
    best_global_model = pick_best_global_model(selection_df)

    print("\nBest seasonal models by stored month-group test_MAPE:")
    print(best_seasonal_models_df[["month_group_label", "model_name", "test_MAPE", "test_MAE", "test_R2"]].to_string(index=False))

    if best_global_model is not None:
        print("\nBest global baseline model:")
        print(pd.DataFrame([best_global_model[["month_group_label", "model_name", "test_MAPE", "test_MAE", "test_R2"]]]).to_string(index=False))
    else:
        print("\nNo global baseline model with stored test metrics was found.")

    seasonal_stitched_df, seasonal_overall_metrics, seasonal_group_metrics_df = stitch_seasonal_predictions(
        best_seasonal_models_df,
        zone=ZONE,
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
    )

    comparison_rows = [seasonal_overall_metrics]
    baseline_prediction_df = None

    if best_global_model is not None:
        baseline_prediction_df, baseline_metrics = evaluate_model_row(
            best_global_model,
            zone=ZONE,
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
        )
        baseline_metrics["Model"] = "BestGlobalModel"
        comparison_rows.append(baseline_metrics)

    comparison_df = pd.DataFrame(comparison_rows).sort_values("MAPE").reset_index(drop=True)

    print("\nAggregate comparison:")
    print(comparison_df[["Model", "MAE", "RMSE", "MAPE", "MAE%", "R2", "TotalAbsError", "Bias", "n_points"]].to_string(index=False))

    print("\nPer-seasonal-group metrics:")
    print(
        seasonal_group_metrics_df[
            ["month_group_label", "Model", "MAE", "RMSE", "MAPE", "R2", "TotalAbsError", "n_points"]
        ].to_string(index=False)
    )

    plot_predictions(seasonal_stitched_df, baseline_prediction_df)

    seasonal_stitched_df.to_csv(RESULTS_DIR / "seasonal_best_models_eval_predictions.csv", index=False)
    seasonal_group_metrics_df.to_csv(RESULTS_DIR / "seasonal_best_models_eval_group_metrics.csv", index=False)
    comparison_df.to_csv(RESULTS_DIR / "seasonal_best_models_eval_comparison.csv", index=False)

    print("\nSaved:")
    print(RESULTS_DIR / "seasonal_best_models_eval_predictions.csv")
    print(RESULTS_DIR / "seasonal_best_models_eval_group_metrics.csv")
    print(RESULTS_DIR / "seasonal_best_models_eval_comparison.csv")
