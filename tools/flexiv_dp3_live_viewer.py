"""Deprecated compatibility shim for the removed legacy live viewer.

The transport and rendering implementation now lives in
``visualizer.monitor``.  New code must construct :class:`MonitorClient`
directly; this module intentionally contains no GUI or rendering logic.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "tools.flexiv_dp3_live_viewer is deprecated; use visualizer.monitor",
    DeprecationWarning,
    stacklevel=2,
)

from visualizer.monitor import MonitorClient  # noqa: E402
from visualizer.monitor.config import ViewerConfig  # noqa: E402

LiveVisualizationPublisher = MonitorClient

__all__ = ["LiveVisualizationPublisher", "MonitorClient", "ViewerConfig"]
