"""Dashboard package (spec §17)."""
from __future__ import annotations

from .app import build_dashboard, serve, metric_bar, behavioral_radar

__all__ = ["build_dashboard", "serve", "metric_bar", "behavioral_radar"]
