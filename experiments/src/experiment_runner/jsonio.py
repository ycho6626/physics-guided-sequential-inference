"""Strict JSON sanitation/writing helpers for experiments artifacts."""

from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
from typing import Any


def sanitize_json(value: Any) -> Any:
    """Recursively convert non-finite floats to JSON-safe null (None)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        num = float(value)
        if not math.isfinite(num):
            return None
        return num
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return sanitize_json(value.tolist())
        except Exception:
            pass
    return value


def dumps_json(value: Any, *, sort_keys: bool = True, indent: int | None = 2) -> str:
    sanitized = sanitize_json(value)
    return json.dumps(sanitized, sort_keys=sort_keys, indent=indent, allow_nan=False)


def write_json(path: Path, value: Any, *, sort_keys: bool = True, indent: int | None = 2) -> Any:
    sanitized = sanitize_json(value)
    path.write_text(
        json.dumps(sanitized, sort_keys=sort_keys, indent=indent, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return sanitized
