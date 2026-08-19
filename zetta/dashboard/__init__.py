"""Optional dashboard layer for live-monitoring a Zetta run.

Opt in via ``python zetta/cli/main.py --dashboard``; never imported on the normal
CLI path.
"""
from zetta.dashboard.server import DashboardServer
from zetta.dashboard.state import State

__all__ = ["DashboardServer", "State"]
