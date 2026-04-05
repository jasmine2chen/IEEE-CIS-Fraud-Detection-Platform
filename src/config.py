"""Central config loader — single source of truth for all pipeline parameters.

Usage:
    from src.config import load_config
    cfg = load_config()          # loads configs/model_config.yaml by default
    cfg = load_config("path/to/other.yaml")
"""
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "model_config.yaml"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load and return the YAML config as a plain dict."""
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
