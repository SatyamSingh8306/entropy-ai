"""Plugins package (spec §18)."""
from __future__ import annotations

from .manager import (
    exporter, list_exporters, register_exporter, discover,
)

__all__ = ["exporter", "list_exporters", "register_exporter", "discover"]
