"""Configuration helpers for TravelTime surface pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from pyprojroot import here


def project_path(path: str | Path) -> Path:
    """Resolve a repository-relative path."""
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(here() / path)


def load_params(path: str | Path = "params.yaml") -> Dict[str, Any]:
    with open(project_path(path), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_parent(path: str | Path) -> Path:
    out = project_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out
