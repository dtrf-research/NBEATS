"""Model registry access and checkpoint loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import torch
from darts.models import NBEATSModel

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REGISTRY_FILE = MODELS_DIR / "model_registry.json"


def load_registry(registry_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the full model registry dict."""
    registry_path = registry_path or REGISTRY_FILE
    with registry_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_models(registry: Dict[str, Any]) -> List[str]:
    """Return sorted model names."""
    return sorted(registry.keys())


def model_display_label(name: str, entry: Dict[str, Any]) -> str:
    """Readable one-liner for dropdown display."""
    cfg = entry.get("config", {})
    zone = entry.get("zone", "?")
    layers = cfg.get("num_layers", "?")
    width = cfg.get("layer_widths", "?")
    icl = cfg.get("input_chunk_length", "?")
    months_label = entry.get("requested_months_label") or cfg.get("requested_months_label", "")
    mape = entry.get("metrics", {}).get("test_MAPE")
    mape_str = f" | MAPE={mape:.2f}%" if mape is not None else ""
    return f"{name}  [{zone} | {layers}L {width}W icl={icl} {months_label}{mape_str}]"


def load_model(model_name: str, entry: Dict[str, Any]) -> NBEATSModel:
    """Load best checkpoint for a model."""
    torch.serialization.add_safe_globals([torch.optim.Adam])
    model = NBEATSModel.load_from_checkpoint(
        model_name=model_name,
        best=True,
    )
    return model


def load_scaler(entry: Dict[str, Any]):
    """Load the persisted Darts Scaler for a model."""
    scaler_path = entry.get("scaler_path", "")
    # Registry stores relative paths from project root
    abs_path = Path(__file__).resolve().parent.parent / scaler_path
    return joblib.load(abs_path)
