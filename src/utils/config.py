"""
Configuration management system.
Reads YAML configuration files into structured dictionaries.
"""
from pathlib import Path
from typing import Any, Dict, Union
import yaml


def load_config(config_path: Union[str, Path] = "configs/default_config.yaml") -> Dict[str, Any]:
    """
    Loads and parses a YAML configuration file.

    Args:
        config_path (Union[str, Path]): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}
