"""Template utilities for deterministic report rendering."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from semgen.reports.errors import RenderingError


_PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def resolve_template_paths(config: dict[str, Any], *, module_root: Path) -> dict[str, Path]:
    """Resolve configured template paths relative to module root."""
    paths: dict[str, Path] = {}
    for key in ("operator", "commander"):
        raw = Path(str(config["templates"][key]))
        paths[key] = raw if raw.is_absolute() else module_root / raw
    return paths


def load_template(template_path: Path) -> str:
    """Load template text from disk."""
    try:
        return template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RenderingError(f"failed to read template: {template_path}") from exc


def render_template(template_text: str, context: dict[str, str]) -> str:
    """Render a template by deterministic placeholder substitution."""

    def _replacement(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            raise RenderingError(f"missing template context key: {key}")
        return context[key]

    rendered = _PLACEHOLDER_RE.sub(_replacement, template_text)

    # Guard against unresolved placeholders due malformed template text.
    unresolved = _PLACEHOLDER_RE.findall(rendered)
    if unresolved:
        names = ", ".join(sorted(set(unresolved)))
        raise RenderingError(f"unresolved placeholders after render: {names}")

    return rendered
