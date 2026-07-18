"""Validated monitor configuration and runtime shape contracts."""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ViewerConfig:
    mode: str = "spawn"
    url: str = "rerun+http://127.0.0.1:9876/proxy"
    port: int = 9876
    memory_limit: str = "2GB"
    server_memory_limit: str = "512MB"
    hide_welcome_screen: bool = True
    # Keep the native Viewer alive after inference exits. A later inference
    # run reuses the listener on ``port`` instead of spawning a duplicate.
    detach_process: bool = True
    # Applying the generated Blueprint makes log_time follow the newest sample.
    # Disable this only when an already-open Viewer must retain a custom layout.
    activate_blueprint_on_start: bool = True
    reconnect_attempts: int = 3
    reconnect_interval_sec: float = 1.0


@dataclass(frozen=True)
class MonitorRates:
    control_hz: float = 10.0
    camera_hz: float = 5.0
    sampled_pointcloud_hz: float = 5.0
    stage_pointcloud_hz: float = 1.0
    health_hz: float = 1.0


@dataclass(frozen=True)
class MonitorPayloads:
    state: bool = True
    rgb: bool = True
    depth: bool = True
    sampled_pointcloud: bool = True
    raw_pointcloud: bool = False
    cropped_pointcloud: bool = False
    policy_horizon: bool = True
    selected_action: bool = True
    filtered_action: bool = True
    commanded_action: bool = True
    timing: bool = True


@dataclass(frozen=True)
class MonitorDisplay:
    max_raw_points: int = 10_000
    max_cropped_points: int = 10_000
    point_radius: float = 0.003


@dataclass(frozen=True)
class MonitorRecording:
    application_id: str = "flexiv_dp3_monitor"
    save_rrd: bool = False
    rrd_path: str | None = None


@dataclass(frozen=True)
class MonitorConfig:
    enabled: bool = False
    backend: str = "rerun"
    fail_open: bool = True
    startup_timeout_sec: float = 5.0
    shutdown_timeout_sec: float = 3.0
    min_bulk_slack_ms: float = 5.0
    heartbeat_interval_sec: float = 1.0
    ring_capacity: int = 3
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    rates: MonitorRates = field(default_factory=MonitorRates)
    payloads: MonitorPayloads = field(default_factory=MonitorPayloads)
    display: MonitorDisplay = field(default_factory=MonitorDisplay)
    recording: MonitorRecording = field(default_factory=MonitorRecording)
    depth_dtype: str = "uint16"

    @property
    def stages_enabled(self) -> bool:
        return self.payloads.raw_pointcloud or self.payloads.cropped_pointcloud

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TelemetryShapes:
    """All fixed transport dimensions, derived from runtime contracts."""

    camera_height: int
    camera_width: int
    point_count: int
    point_dim: int
    state_dim: int
    action_dim: int
    policy_horizon: int
    depth_dtype: str = "uint16"
    max_raw_points: int = 10_000
    max_cropped_points: int = 10_000

    def __post_init__(self) -> None:
        for name in (
            "camera_height",
            "camera_width",
            "point_count",
            "point_dim",
            "state_dim",
            "action_dim",
            "policy_horizon",
            "max_raw_points",
            "max_cropped_points",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"Telemetry shape {name} must be positive, got {value}")
        if self.point_dim not in (3, 6):
            raise ValueError(f"point_dim must be 3 or 6, got {self.point_dim}")
        np.dtype(self.depth_dtype)


def load_monitor_config(
    config: Mapping[str, Any],
    *,
    inference_rate_hz: float | None = None,
) -> MonitorConfig:
    """Parse the new ``monitor`` section and warn for legacy configuration."""

    raw = config.get("monitor")
    legacy = config.get("visualization")
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("monitor must be a mapping")
    if raw is None:
        if legacy is not None:
            warnings.warn(
                "visualization is deprecated; migrate to monitor",
                DeprecationWarning,
                stacklevel=2,
            )
            raw = _legacy_to_monitor(legacy, inference_rate_hz=inference_rate_hz)
        else:
            raw = {"enabled": False}
    elif legacy is not None:
        warnings.warn(
            "Both monitor and visualization are present; monitor takes precedence",
            RuntimeWarning,
            stacklevel=2,
        )
    result = _parse_monitor(raw, inference_rate_hz=inference_rate_hz)
    if not result.enabled:
        return result
    return result


def _legacy_to_monitor(value: Mapping[str, Any], *, inference_rate_hz: float | None) -> dict[str, Any]:
    return {
        "enabled": bool(value.get("enabled", False)),
        "rates": {
            "control_hz": inference_rate_hz or float(value.get("rate_hz", 10.0)),
            "camera_hz": float(value.get("rate_hz", 2.0)),
            "sampled_pointcloud_hz": float(value.get("rate_hz", 2.0)),
            "stage_pointcloud_hz": float(value.get("rate_hz", 2.0)),
        },
        "payloads": {
            "raw_pointcloud": True,
            "cropped_pointcloud": True,
        },
        "display": {
            "max_raw_points": int(value.get("max_raw_points", 10_000)),
            "max_cropped_points": int(value.get("max_cropped_points", 10_000)),
            "point_radius": float(value.get("point_size", 3.0)),
        },
    }


def _parse_monitor(raw: Mapping[str, Any], *, inference_rate_hz: float | None) -> MonitorConfig:
    rates_raw = _mapping(raw, "rates")
    rates = MonitorRates(
        control_hz=float(rates_raw.get("control_hz", inference_rate_hz or 10.0)),
        camera_hz=float(rates_raw.get("camera_hz", 5.0)),
        sampled_pointcloud_hz=float(rates_raw.get("sampled_pointcloud_hz", 5.0)),
        stage_pointcloud_hz=float(rates_raw.get("stage_pointcloud_hz", 1.0)),
        health_hz=float(rates_raw.get("health_hz", 1.0)),
    )
    for name, value in asdict(rates).items():
        if not np.isfinite(value) or value <= 0.0 or value > 1000.0:
            raise ValueError(f"monitor.rates.{name} must be in (0, 1000], got {value}")
    payloads = _dataclass_from_mapping(MonitorPayloads, _mapping(raw, "payloads"))
    display = MonitorDisplay(
        max_raw_points=_positive_int(_mapping(raw, "display").get("max_raw_points", 10_000), "max_raw_points"),
        max_cropped_points=_positive_int(_mapping(raw, "display").get("max_cropped_points", 10_000), "max_cropped_points"),
        point_radius=float(_mapping(raw, "display").get("point_radius", 0.003)),
    )
    if not np.isfinite(display.point_radius) or display.point_radius <= 0:
        raise ValueError("monitor.display.point_radius must be positive")
    viewer_raw = _mapping(raw, "viewer")
    mode = str(viewer_raw.get("mode", "spawn")).strip().lower()
    if mode not in {"spawn", "connect"}:
        raise ValueError("monitor.viewer.mode must be spawn or connect")
    url = str(viewer_raw.get("url", "rerun+http://127.0.0.1:9876/proxy"))
    if mode == "connect" and not url.startswith("rerun+"):
        raise ValueError("monitor.viewer.url must start with rerun+ in connect mode")
    port = _positive_int(viewer_raw.get("port", 9876), "viewer.port")
    if port > 65535:
        raise ValueError("monitor.viewer.port must be <= 65535")
    viewer = ViewerConfig(
        mode=mode,
        url=url,
        port=port,
        memory_limit=str(viewer_raw.get("memory_limit", "2GB")),
        server_memory_limit=str(viewer_raw.get("server_memory_limit", "512MB")),
        hide_welcome_screen=bool(viewer_raw.get("hide_welcome_screen", True)),
        detach_process=bool(viewer_raw.get("detach_process", True)),
        activate_blueprint_on_start=bool(
            viewer_raw.get("activate_blueprint_on_start", True)
        ),
        reconnect_attempts=_positive_int(viewer_raw.get("reconnect_attempts", 3), "reconnect_attempts"),
        reconnect_interval_sec=float(viewer_raw.get("reconnect_interval_sec", 1.0)),
    )
    if not np.isfinite(viewer.reconnect_interval_sec) or viewer.reconnect_interval_sec < 0:
        raise ValueError("monitor.viewer.reconnect_interval_sec must be finite and non-negative")
    recording_raw = _mapping(raw, "recording")
    recording = MonitorRecording(
        application_id=str(recording_raw.get("application_id", "flexiv_dp3_monitor")),
        save_rrd=bool(recording_raw.get("save_rrd", False)),
        rrd_path=None if recording_raw.get("rrd_path") is None else str(recording_raw["rrd_path"]),
    )
    if recording.save_rrd and not recording.rrd_path:
        raise ValueError("monitor.recording.rrd_path is required when save_rrd=true")
    capacity = _positive_int(_mapping(raw, "ring").get("capacity", 3), "ring.capacity")
    if capacity < 2 or capacity > 64:
        raise ValueError("monitor.ring.capacity must be in [2, 64]")
    startup = float(raw.get("startup_timeout_sec", 5.0))
    shutdown = float(raw.get("shutdown_timeout_sec", 3.0))
    slack = float(raw.get("min_bulk_slack_ms", 5.0))
    heartbeat = float(raw.get("heartbeat_interval_sec", 1.0))
    if min(startup, shutdown, heartbeat) <= 0 or not np.isfinite(startup + shutdown + heartbeat):
        raise ValueError("monitor startup/shutdown timeouts must be positive")
    if slack < 0 or not np.isfinite(slack):
        raise ValueError("monitor.min_bulk_slack_ms must be non-negative")
    backend = str(raw.get("backend", "rerun")).strip().lower()
    if backend != "rerun":
        raise ValueError("monitor.backend currently only supports rerun")
    depth_dtype = str(raw.get("depth_dtype", "uint16"))
    try:
        np.dtype(depth_dtype)
    except TypeError as exc:
        raise ValueError(f"monitor.depth_dtype is invalid: {depth_dtype!r}") from exc
    return MonitorConfig(
        enabled=bool(raw.get("enabled", False)),
        backend=backend,
        fail_open=bool(raw.get("fail_open", True)),
        startup_timeout_sec=startup,
        shutdown_timeout_sec=shutdown,
        min_bulk_slack_ms=slack,
        heartbeat_interval_sec=heartbeat,
        ring_capacity=capacity,
        viewer=viewer,
        rates=rates,
        payloads=payloads,
        display=display,
        recording=recording,
        depth_dtype=depth_dtype,
    )


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"monitor.{key} must be a mapping")
    return value


def _dataclass_from_mapping(cls: Any, raw: Mapping[str, Any]) -> Any:
    names = {field.name for field in cls.__dataclass_fields__.values()}
    return cls(**{key: value for key, value in raw.items() if key in names})


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"monitor.{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"monitor.{name} must be a positive integer") from exc
    if isinstance(value, float) and value != result or result <= 0:
        raise ValueError(f"monitor.{name} must be a positive integer")
    return result


__all__ = [
    "MonitorConfig",
    "MonitorDisplay",
    "MonitorPayloads",
    "MonitorRates",
    "MonitorRecording",
    "TelemetryShapes",
    "ViewerConfig",
    "load_monitor_config",
]
