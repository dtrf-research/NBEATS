from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

PathLike = Union[str, Path]


def _pick_list_value(payload: Dict[str, Any], key: str) -> List[Any]:
    """Get a list field from top-level first, then training_data, then config."""
    candidates = [
        payload.get(key),
        payload.get("training_data", {}).get(key) if isinstance(payload.get("training_data"), dict) else None,
        payload.get("config", {}).get(key) if isinstance(payload.get("config"), dict) else None,
    ]
    for value in candidates:
        if isinstance(value, list):
            return value
    return []


def _normalize_scalar(value: Any) -> Any:
    """Convert nested/list values to CSV-friendly scalars."""
    if isinstance(value, list):
        # Keep lists readable and groupable in spreadsheets/SQL tools.
        return "|".join(str(x) for x in value)
    return value


def _flatten_dict(
    data: Dict[str, Any],
    prefix: str = "",
    out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Recursively flatten nested dictionaries using underscore-separated keys."""
    if out is None:
        out = {}

    for key, value in data.items():
        new_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            _flatten_dict(value, prefix=new_key, out=out)
        else:
            out[new_key] = _normalize_scalar(value)

    return out


def model_registry_to_rows(registry_path: PathLike) -> List[Dict[str, Any]]:
    """
    Convert model_registry.json (dict keyed by model_name) to flat tabular rows.

    Parameters
    ----------
    registry_path:
        Path to model_registry.json.

    Returns
    -------
    list[dict[str, Any]]
        One row per model with flattened keys such as:
        - config_input_chunk_length
        - metrics_test_MAPE
        - training_data_requested_months
    """
    registry_path = Path(registry_path)
    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    if not isinstance(registry, dict):
        raise ValueError("Expected model registry JSON to be a dict keyed by model name.")

    rows: List[Dict[str, Any]] = []
    for model_name, payload in registry.items():
        if not isinstance(payload, dict):
            payload = {"raw_value": payload}

        flat = _flatten_dict(payload)
        flat["model_name"] = model_name
        rows.append(flat)
    return rows


def rows_to_dataframe(rows: List[Dict[str, Any]]):
    """Optional: convert flattened rows to a pandas DataFrame if pandas is available."""
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError("pandas is not installed. Install it to use DataFrame conversion.") from exc

    df = pd.DataFrame(rows)
    preferred_first = [
        "model_name",
        "zone",
        "created_at",
        "checkpoint_dir",
        "scaler_path",
        "train_time_sec",
    ]
    first_cols = [c for c in preferred_first if c in df.columns]
    other_cols = [c for c in df.columns if c not in first_cols]
    return df[first_cols + other_cols]


def model_registry_to_csv(
    registry_path: PathLike,
    csv_output_path: PathLike,
    sort_by: Optional[Iterable[str]] = None,
    ascending: bool = True,
) -> List[Dict[str, Any]]:
    """
    Flatten model registry into CSV and return the written rows.

    Parameters
    ----------
    registry_path:
        Path to model_registry.json.
    csv_output_path:
        Output CSV path.
    sort_by:
        Optional iterable of column names to sort by before writing.
    ascending:
        Sort direction.
    """
    rows = model_registry_to_rows(registry_path)

    if sort_by:
        sort_cols = list(sort_by)

        def sort_key(item: Dict[str, Any]):
            return tuple(item.get(col) for col in sort_cols)

        rows.sort(key=sort_key, reverse=not ascending)

    # Build complete field list across heterogenous records.
    fieldnames: List[str] = []
    seen = set()
    preferred_first = [
        "model_name",
        "zone",
        "created_at",
        "checkpoint_dir",
        "scaler_path",
        "train_time_sec",
    ]
    for field in preferred_first:
        if any(field in row for row in rows):
            fieldnames.append(field)
            seen.add(field)

    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    csv_output_path = Path(csv_output_path)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def model_registry_metrics_to_long_csv(
    registry_path: PathLike,
    csv_output_path: PathLike,
    include_non_test_metrics: bool = True,
) -> List[Dict[str, Any]]:
    """
    Export metrics in long format: one metric per row per model.

    Output columns:
    - model_name
    - zone
    - created_at
    - metric_name
    - metric_value
    """
    registry_path = Path(registry_path)
    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    if not isinstance(registry, dict):
        raise ValueError("Expected model registry JSON to be a dict keyed by model name.")

    metric_rows: List[Dict[str, Any]] = []
    for model_name, payload in registry.items():
        if not isinstance(payload, dict):
            continue

        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            continue

        for metric_name, metric_value in metrics.items():
            if not include_non_test_metrics and not str(metric_name).startswith("test_"):
                continue

            metric_rows.append(
                {
                    "model_name": model_name,
                    "zone": payload.get("zone"),
                    "created_at": payload.get("created_at"),
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                }
            )

    csv_output_path = Path(csv_output_path)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_name", "zone", "created_at", "metric_name", "metric_value"]
    with csv_output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    return metric_rows


def model_registry_to_normalized_csvs(
    registry_path: PathLike,
    output_dir: PathLike,
) -> Dict[str, int]:
    """
    Export normalized bridge tables for list-like metadata.

    Creates:
    - model_registry_core.csv (one row per model with non-list fields)
    - model_registry_requested_months.csv (one row per requested month)
    - model_registry_training_years.csv (one row per training year)
    - model_registry_training_months_present.csv (one row per month present in training data)
    """
    registry_path = Path(registry_path)
    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    if not isinstance(registry, dict):
        raise ValueError("Expected model registry JSON to be a dict keyed by model name.")

    core_rows: List[Dict[str, Any]] = []
    requested_month_rows: List[Dict[str, Any]] = []
    training_year_rows: List[Dict[str, Any]] = []
    training_month_present_rows: List[Dict[str, Any]] = []

    for model_name, payload in registry.items():
        if not isinstance(payload, dict):
            payload = {"raw_value": payload}

        flat = _flatten_dict(payload)
        # Remove pipe-joined list columns from core; keep them in normalized tables instead.
        for noisy_list_col in [
            "requested_months",
            "training_years",
            "training_months_present",
            "training_data_requested_months",
            "training_data_training_years",
            "training_data_training_months_present",
            "config_requested_months",
        ]:
            flat.pop(noisy_list_col, None)

        flat["model_name"] = model_name
        core_rows.append(flat)

        for month in _pick_list_value(payload, "requested_months"):
            requested_month_rows.append(
                {
                    "model_name": model_name,
                    "requested_month": month,
                }
            )

        for year in _pick_list_value(payload, "training_years"):
            training_year_rows.append(
                {
                    "model_name": model_name,
                    "training_year": year,
                }
            )

        for month in _pick_list_value(payload, "training_months_present"):
            training_month_present_rows.append(
                {
                    "model_name": model_name,
                    "training_month_present": month,
                }
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write core table using existing wide CSV helper behavior.
    core_path = output_dir / "model_registry_core.csv"
    if core_rows:
        fieldnames: List[str] = []
        seen = set()
        preferred_first = ["model_name", "zone", "created_at", "checkpoint_dir", "scaler_path"]
        for field in preferred_first:
            if any(field in row for row in core_rows):
                fieldnames.append(field)
                seen.add(field)
        for row in core_rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

        with core_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(core_rows)

    def _write_simple(rows: List[Dict[str, Any]], filename: str, fields: List[str]) -> None:
        out_path = output_dir / filename
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    _write_simple(
        requested_month_rows,
        "model_registry_requested_months.csv",
        ["model_name", "requested_month"],
    )
    _write_simple(
        training_year_rows,
        "model_registry_training_years.csv",
        ["model_name", "training_year"],
    )
    _write_simple(
        training_month_present_rows,
        "model_registry_training_months_present.csv",
        ["model_name", "training_month_present"],
    )

    return {
        "model_registry_core.csv": len(core_rows),
        "model_registry_requested_months.csv": len(requested_month_rows),
        "model_registry_training_years.csv": len(training_year_rows),
        "model_registry_training_months_present.csv": len(training_month_present_rows),
    }
