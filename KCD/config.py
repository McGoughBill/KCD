"""Centralised path configuration for KCD.

Resolution order (first hit wins):
    1. Environment variable (e.g. KCD_DATA_ROOT, KCD_OV_DATA_BASE)
    2. configs/paths.yaml at the repo root (if PyYAML is installed)
    3. Raise FileNotFoundError — no silent fallback.

Keys exposed:
    data_root, ov_data_base, raw_data, predictions, model_weights
"""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path

_ENV_PREFIX = "KCD_"
_REQUIRED = ("data_root", "ov_data_base", "raw_data", "predictions", "model_weights")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _from_yaml() -> dict | None:
    cfg = _repo_root() / "configs" / "paths.yaml"
    if not cfg.exists():
        return None
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(f"{cfg} exists but PyYAML is not installed.") from e
    with cfg.open() as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_paths() -> dict:
    yaml_cfg = _from_yaml() or {}
    paths = {}
    for key in _REQUIRED:
        env_val = os.environ.get(_ENV_PREFIX + key.upper())
        paths[key] = env_val or yaml_cfg.get(key)
        if paths[key] is None:
            raise FileNotFoundError(
                f"Path '{key}' not configured. Set {_ENV_PREFIX}{key.upper()} "
                f"or add it to configs/paths.yaml (see configs/paths.example.yaml)."
            )
    if "OV_DATA_BASE" not in os.environ:
        os.environ["OV_DATA_BASE"] = paths["ov_data_base"]
    return paths
