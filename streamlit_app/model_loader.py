"""Model registry access and checkpoint loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import torch
from darts.models import NBEATSModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_FILE = MODELS_DIR / "model_registry.json"

# Known zone subdirectories that may contain their own models/ and darts_logs/
_ZONE_DIRS = ["TPCODL", "TPWODL", "TPNODL", "TPSOSDL", "Total"]


def load_registry(registry_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the full model registry dict."""
    registry_path = registry_path or REGISTRY_FILE
    with registry_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_all_registries() -> Dict[str, Any]:
    """Load and merge registries from root *and* all zone subdirectories.

    For each zone directory (e.g. ``TPCODL/models/model_registry.json``),
    entries are loaded, enriched with a ``work_dir`` pointing to the zone
    folder, and merged into the combined registry.  Discovered models
    (via ``scaler.joblib`` presence) that are *not* already in any
    registry are added too.
    """
    combined: Dict[str, Any] = {}

    # 1. Root-level registry
    if REGISTRY_FILE.is_file():
        root_reg = load_registry(REGISTRY_FILE)
        # Add work_dir for root models (project root itself)
        for name, entry in root_reg.items():
            entry.setdefault("work_dir", ".")
        combined.update(root_reg)

    # 2. Zone-level registries
    for zone_name in _ZONE_DIRS:
        zone_dir = PROJECT_ROOT / zone_name
        zone_registry_file = zone_dir / "models" / "model_registry.json"
        if zone_registry_file.is_file():
            try:
                zone_reg = load_registry(zone_registry_file)
            except Exception:
                continue
            zone_rel = str(zone_dir.relative_to(PROJECT_ROOT))
            for name, entry in zone_reg.items():
                entry.setdefault("work_dir", zone_rel)
                # scaler_path in zone registries is relative to the zone
                # dir — re-root it relative to the project root.
                sp = entry.get("scaler_path", "")
                if sp:
                    abs_sp = PROJECT_ROOT / sp
                    if not abs_sp.exists():
                        # Path is relative to zone dir, not project root
                        abs_sp = zone_dir / sp
                    if abs_sp.exists():
                        entry["scaler_path"] = str(abs_sp.relative_to(PROJECT_ROOT))
            combined.update(zone_reg)

    # 3. Discover any additional models not yet in a registry
    for search_dir in [MODELS_DIR] + [PROJECT_ROOT / z / "models" for z in _ZONE_DIRS]:
        if search_dir.is_dir():
            discovered = discover_models_in_directory(search_dir)
            for name, entry in discovered.items():
                if name not in combined:
                    combined[name] = entry

    return combined


def list_models(registry: Dict[str, Any]) -> List[str]:
    """Return sorted model names."""
    return sorted(registry.keys())


def model_display_label(name: str, entry: Dict[str, Any]) -> str:
    """Readable one-liner for dropdown display."""
    cfg = entry.get("config", {})
    zone = entry.get("zone", "Unknown")
    # Shorten zone name (e.g. "TPCODL Demand" -> "TPCODL")
    zone_short = zone.replace(" Demand", "") if zone else "?"

    layers = cfg.get("num_layers", "?")
    width = cfg.get("layer_widths", "?")

    # Convert input_chunk_length back to days for readability
    icl = cfg.get("input_chunk_length")
    if icl is not None and isinstance(icl, (int, float)) and icl >= 96:
        ctx_str = f"{int(icl) // 96}d ctx"
    elif icl is not None:
        ctx_str = f"icl={icl}"
    else:
        ctx_str = ""

    # Month label
    _MONTH_ABBR = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    req_months = cfg.get("requested_months") or entry.get("training_data", {}).get("requested_months")
    if req_months and isinstance(req_months, list):
        months_str = ", ".join(_MONTH_ABBR.get(m, str(m)) for m in sorted(req_months))
    else:
        raw = entry.get("requested_months_label") or cfg.get("requested_months_label", "")
        months_str = raw.replace("months_", "M").replace("_", ",") if raw else ""

    mape = entry.get("metrics", {}).get("test_MAPE")
    mape_str = f"MAPE {mape:.2f}%" if mape is not None else ""

    # Assemble parts
    parts = [f"{zone_short}"]
    parts.append(f"{layers}L \u00d7 {width}W")
    if ctx_str:
        parts.append(ctx_str)
    if months_str:
        parts.append(months_str)
    if mape_str:
        parts.append(mape_str)
    return " | ".join(parts)


def get_available_zones(registry: Dict[str, Any]) -> List[str]:
    """Return sorted unique zones present in the registry."""
    zones = {entry.get("zone", "Unknown") for entry in registry.values()}
    return sorted(zones)


def load_model(model_name: str, entry: Dict[str, Any]) -> NBEATSModel:
    """Load best checkpoint for a model."""
    torch.serialization.add_safe_globals([torch.optim.Adam])
    work_dir = entry.get("work_dir")
    if work_dir:
        project_root = Path(__file__).resolve().parent.parent
        # Darts only appends "darts_logs" automatically when work_dir is None.
        # When passed explicitly, work_dir must already include it.
        abs_work_dir = str(project_root / work_dir / "darts_logs")
        model = NBEATSModel.load_from_checkpoint(
            model_name=model_name,
            work_dir=abs_work_dir,
            best=True,
        )
    else:
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


# ---------------------------------------------------------------------------
# Directory-based model discovery
# ---------------------------------------------------------------------------

def discover_models_in_directory(directory: Path) -> Dict[str, Any]:
    """Scan *directory* (and subdirectories up to 2 levels deep) for valid
    N-BEATS model artifacts and return a registry-compatible dict
    ``{model_name: entry}``.

    A subfolder is considered a valid model when:
    - it contains a ``scaler.joblib`` file, AND
    - a matching checkpoint directory exists under a sibling or
      ancestor ``darts_logs/`` folder.

    This supports both flat layouts (``models/{model}/``) and
    zone-based layouts (``ZONE/models/{model}/`` with
    ``ZONE/darts_logs/{model}/``).
    """
    project_root = Path(__file__).resolve().parent.parent
    discovered: Dict[str, Any] = {}

    if not directory.is_dir():
        return discovered

    _scan_for_models(directory, project_root, discovered, max_depth=2)
    return discovered


def _scan_for_models(
    directory: Path,
    project_root: Path,
    discovered: Dict[str, Any],
    max_depth: int,
) -> None:
    """Recursively collect model directories containing ``scaler.joblib``."""
    if max_depth < 0 or not directory.is_dir():
        return

    # Directories to skip during recursive scan
    _SKIP = {"darts_logs", "logs", "results", "__pycache__", ".git", "streamlit_app"}

    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue

        scaler_file = child / "scaler.joblib"
        if scaler_file.exists():
            # This child is a model directory
            model_name = child.name
            work_dir = _find_work_dir(model_name, child, project_root)
            if work_dir is None:
                continue
            try:
                rel_scaler = str(scaler_file.relative_to(project_root))
                rel_work = str(work_dir.relative_to(project_root))
            except ValueError:
                continue
            discovered[model_name] = {
                "zone": _infer_zone(model_name),
                "config": _parse_config_from_name(model_name),
                "metrics": {},
                "checkpoint_dir": str(child.relative_to(project_root)),
                "scaler_path": rel_scaler,
                "work_dir": rel_work,
            }
        elif child.name not in _SKIP:
            # Recurse into potential parent / zone directories
            _scan_for_models(child, project_root, discovered, max_depth - 1)


def _find_work_dir(
    model_name: str, model_dir: Path, project_root: Path
) -> Optional[Path]:
    """Return the directory that contains ``darts_logs/{model_name}/``.

    Walks up from the model directory's parent toward the project root.
    """
    candidate = model_dir.parent  # e.g. ZONE/models
    while True:
        if (candidate / "darts_logs" / model_name).is_dir():
            return candidate
        if candidate == project_root or candidate == candidate.parent:
            break
        candidate = candidate.parent

    # Final check at project root
    if (project_root / "darts_logs" / model_name).is_dir():
        return project_root

    return None


def _infer_zone(model_name: str) -> str:
    """Best-effort zone inference from folder name."""
    low = model_name.lower()
    for zone_tag, zone_label in [
        ("tpcodl", "TPCODL Demand"),
        ("tpwodl", "TPWODL Demand"),
        ("tpnodl", "TPNODL Demand"),
        ("tpsosdl", "TPSOSDL Demand"),
        ("total", "Total Demand"),
    ]:
        if zone_tag in low:
            return zone_label
    return "Unknown"


def _parse_config_from_name(model_name: str) -> Dict[str, Any]:
    """Extract layer / width / context info embedded in the folder name."""
    import re
    cfg: Dict[str, Any] = {}
    m = re.search(r"(\d+)l", model_name)
    if m:
        cfg["num_layers"] = int(m.group(1))
    m = re.search(r"(\d+)w", model_name)
    if m:
        cfg["layer_widths"] = int(m.group(1))
    m = re.search(r"ctx(\d+)d", model_name)
    if m:
        days = int(m.group(1))
        cfg["input_chunk_length"] = days * 96  # 96 steps/day @ 15-min
    return cfg
