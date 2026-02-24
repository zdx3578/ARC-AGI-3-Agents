from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _runtime_config_paths() -> list[Path]:
    root = _repo_root()
    return [
        root / "config" / "runtime_config.json",
        root / "config" / "runtime_config.local.json",
    ]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = out.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            out[key] = _deep_merge(base_value, value)
            continue
        out[key] = value
    return out


@lru_cache(maxsize=1)
def _load_runtime_config() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in _runtime_config_paths():
        if not path.is_file():
            continue
        try:
            parsed_any = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(parsed_any, dict):
            continue
        merged = _deep_merge(merged, parsed_any)
    return merged


def reload_runtime_config() -> None:
    _load_runtime_config.cache_clear()


def get_runtime_setting(
    key: str,
    default: Any = None,
    *,
    section: str | None = None,
) -> Any:
    config = _load_runtime_config()
    if section:
        section_obj = config.get(section)
        if isinstance(section_obj, dict) and key in section_obj:
            return section_obj[key]
    if key in config:
        return config[key]
    return default


def get_runtime_str(
    key: str,
    default: str,
    *,
    section: str | None = None,
) -> str:
    value = get_runtime_setting(key, default, section=section)
    if value is None:
        return str(default)
    return str(value)


def get_runtime_int(
    key: str,
    default: int,
    *,
    section: str | None = None,
) -> int:
    value = get_runtime_setting(key, default, section=section)
    try:
        return int(value)
    except Exception:
        return int(default)


def get_runtime_float(
    key: str,
    default: float,
    *,
    section: str | None = None,
) -> float:
    value = get_runtime_setting(key, default, section=section)
    try:
        return float(value)
    except Exception:
        return float(default)


def get_runtime_bool(
    key: str,
    default: bool,
    *,
    section: str | None = None,
) -> bool:
    value = get_runtime_setting(key, default, section=section)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)
