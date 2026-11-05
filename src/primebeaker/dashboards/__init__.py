"""Reusable, read-only live dashboards and pluggable training views."""

from .config import DashboardConfig
from .store import SnapshotStore

__all__ = ["DashboardConfig", "SnapshotStore"]
