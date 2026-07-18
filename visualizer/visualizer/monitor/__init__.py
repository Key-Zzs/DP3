"""Process-isolated telemetry for the DP3 live inference path.

The package intentionally does not import :mod:`rerun`.  The optional SDK is
loaded only by the telemetry child when a Rerun sink is selected.
"""

from .client import CyclePlan, MonitorClient
from .config import MonitorConfig, TelemetryShapes, load_monitor_config
from .schema import TelemetrySchema
from .shared_ring import SharedMemoryTelemetryBus

__all__ = [
    "CyclePlan",
    "MonitorClient",
    "MonitorConfig",
    "SharedMemoryTelemetryBus",
    "TelemetrySchema",
    "TelemetryShapes",
    "load_monitor_config",
]
