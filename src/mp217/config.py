"""Configuration module for MP217 Ration Shop Distribution project."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def get_default_config_path() -> Path:
    """Find default path to params.yaml starting from current dir or project root."""
    # Check current working directory
    cwd_path = Path("params.yaml")
    if cwd_path.is_file():
        return cwd_path.resolve()

    # Check relative to this module: src/mp217/config.py -> project root
    module_root = Path(__file__).resolve().parents[2] / "params.yaml"
    if module_root.is_file():
        return module_root

    return cwd_path.resolve()


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load and parse params.yaml using yaml.safe_load.

    Args:
        config_path: Optional path to the configuration YAML file.
            If None, uses get_default_config_path().

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        ValueError: If the file is empty or cannot be parsed as valid YAML dictionary.
    """
    path = Path(config_path) if config_path is not None else get_default_config_path()

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found at: {path}. "
            "Please ensure params.yaml exists in the project root."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML configuration at {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid configuration format in {path}: expected a dictionary, got {type(config).__name__}"
        )

    return config
