"""Rerun logger used exclusively inside the telemetry child."""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .blueprint import build_dp3_blueprint
from .config import MonitorConfig
from .schema import (
    FLAG_COMMANDED_ACTION,
    FLAG_CROPPED,
    FLAG_DEPTH,
    FLAG_FILTERED_ACTION,
    FLAG_HORIZON,
    FLAG_RAW,
    FLAG_RGB,
    FLAG_SAMPLED,
    FLAG_SELECTED_ACTION,
    FLAG_STATE,
    FLAG_TIMING,
    ChannelSnapshot,
)

LOGGER = logging.getLogger(__name__)


class RerunSink:
    def __init__(self, *, config: MonitorConfig, static_metadata: Mapping[str, Any] | None = None) -> None:
        self.config = config
        self.static_metadata = dict(static_metadata or {})
        self.rr: Any | None = None
        self.recording: Any | None = None
        self.viewer_client: Any | None = None
        self.connected = False
        self._last_prediction_id = -1

    def start(self) -> None:
        self.rr = importlib.import_module("rerun")
        _ensure_viewer_on_path()
        application_id = self.config.recording.application_id
        recording_id = f"dp3-{time.strftime('%Y%m%d-%H%M%S')}-{time.monotonic_ns()}"
        # Rerun 0.26.x returns None from init but installs a global recording;
        # newer SDKs may return the explicit stream.  Both are supported.
        result = self.rr.init(
            application_id,
            recording_id=recording_id,
            spawn=False,
            default_enabled=True,
        )
        self.recording = result or getattr(self.rr, "get_global_data_recording", lambda: None)()
        if self.recording is None:
            self.recording = self.rr
        if self.config.recording.save_rrd and self.config.recording.rrd_path:
            # Rerun's file sink must be installed before the first log call;
            # calling save() during close would create an empty/incomplete RRD.
            saver = getattr(self.recording, "save", None)
            if not callable(saver):
                raise RuntimeError("Rerun RecordingStream does not support save()")
            saver(str(Path(self.config.recording.rrd_path)))
        last_error: Exception | None = None
        attempts = max(1, int(self.config.viewer.reconnect_attempts))
        for attempt in range(attempts):
            try:
                self._connect_viewer()
                self.connected = True
                self._send_blueprint()
                self._log_text("/events", "startup")
                self._log_text("/events", "viewer_connected")
                self._log_text("/metadata/static", _jsonish(self.static_metadata))
                self._log_scalar("/monitor/viewer_connected", 1.0)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self.connected = False
                if attempt + 1 < attempts:
                    time.sleep(float(self.config.viewer.reconnect_interval_sec))
        raise RuntimeError(f"Rerun viewer connection failed after {attempts} attempt(s): {last_error}") from last_error

    def _connect_viewer(self) -> None:
        if self.config.viewer.mode == "spawn":
            experimental = getattr(self.rr, "experimental", None)
            viewer_client_type = getattr(experimental, "ViewerClient", None)
            persistent_url = _as_grpc_proxy_url(
                f"rerun+http://127.0.0.1:{self.config.viewer.port}"
            )
            if self.config.viewer.detach_process and _local_port_open(
                self.config.viewer.port
            ):
                # A detached Viewer from an earlier inference run is already
                # serving this port. Reuse it so the operator keeps the same
                # window and history instead of failing on EADDRINUSE.
                if viewer_client_type is not None:
                    self.viewer_client = viewer_client_type.connect(persistent_url)
                    persistent_url = self.viewer_client.url
                self.rr.connect_grpc(persistent_url, recording=self.recording)
                return
            if viewer_client_type is not None:
                # Unlike rr.spawn(), ViewerClient retains the spawned process
                # id and can terminate both the Python shim and native Viewer.
                self.viewer_client = viewer_client_type.spawn(
                    port=self.config.viewer.port,
                    memory_limit=self.config.viewer.memory_limit,
                    server_memory_limit=self.config.viewer.server_memory_limit,
                    hide_welcome_screen=self.config.viewer.hide_welcome_screen,
                    detach_process=self.config.viewer.detach_process,
                )
                self.rr.connect_grpc(self.viewer_client.url, recording=self.recording)
            else:
                self.rr.spawn(
                    port=self.config.viewer.port,
                    memory_limit=self.config.viewer.memory_limit,
                    server_memory_limit=self.config.viewer.server_memory_limit,
                    hide_welcome_screen=self.config.viewer.hide_welcome_screen,
                    detach_process=self.config.viewer.detach_process,
                    recording=self.recording,
                )
        else:
            self.rr.connect_grpc(_as_grpc_proxy_url(self.config.viewer.url), recording=self.recording)

    def try_reconnect(self) -> bool:
        if self.rr is None:
            return False
        attempts = max(1, int(self.config.viewer.reconnect_attempts))
        for attempt in range(attempts):
            try:
                self._connect_viewer()
                self.connected = True
                self._send_blueprint()
                self._log_text("/events", "viewer_connected")
                self._log_scalar("/monitor/viewer_connected", 1.0)
                return True
            except Exception:
                self.connected = False
                if attempt + 1 < attempts:
                    time.sleep(float(self.config.viewer.reconnect_interval_sec))
        return False

    def _send_blueprint(self) -> None:
        blueprint = build_dp3_blueprint(
            rr_module=self.rr,
            state_fields=tuple(self.static_metadata.get("state_field_names", ())),
            action_fields=tuple(self.static_metadata.get("action_field_names", ())),
            show_stage_pointclouds=self.config.stages_enabled,
        )
        if blueprint is None:
            return
        sender = getattr(self.rr, "send_blueprint", None)
        if callable(sender):
            try:
                sender(
                    blueprint,
                    make_active=self.config.viewer.activate_blueprint_on_start,
                    make_default=True,
                    recording=self.recording,
                )
                return
            except TypeError:
                sender(blueprint)

    def consume(self, snapshot: ChannelSnapshot) -> None:
        if not self.connected:
            return
        data = snapshot.arrays
        metadata = snapshot.metadata
        self._set_time(metadata, snapshot.sequence)
        if snapshot.name == "camera":
            flags = int(metadata["valid_flags"])
            if flags & FLAG_RGB:
                self._log_image("/observation/camera/head/rgb", data["rgb"])
            if flags & FLAG_DEPTH:
                self._log_depth("/observation/camera/head/depth", data["depth"], float(metadata.get("depth_scale", 0.0)))
            self._log_channel_health(snapshot)
            return
        if snapshot.name == "sampled_pointcloud":
            count = int(metadata.get("point_count", data["points"].shape[0]))
            self._log_points("/observation/point_cloud/sampled", data["points"][:count])
            self._log_channel_health(snapshot)
            return
        if snapshot.name == "stage_pointcloud":
            flags = int(metadata["valid_flags"])
            if flags & FLAG_RAW:
                self._log_points("/observation/point_cloud/raw", data["raw_points"][: int(metadata.get("point_count", 0))])
            if flags & FLAG_CROPPED:
                self._log_points("/observation/point_cloud/cropped", data["cropped_points"][: int(metadata.get("point_count_2", 0))])
            self._log_channel_health(snapshot)
            return
        if snapshot.name == "control":
            self._consume_control(snapshot)

    def _consume_control(self, snapshot: ChannelSnapshot) -> None:
        data = snapshot.arrays
        flags = int(snapshot.metadata["valid_flags"])
        if flags & FLAG_STATE:
            for index, value in enumerate(data["measured_state"].reshape(-1)):
                self._log_scalar(f"/observation/state/{self._field_name('state_field_names', index)}", float(value))
        if flags & FLAG_HORIZON:
            prediction_id = int(snapshot.metadata.get("prediction_id", -1))
            horizon_len = min(int(snapshot.metadata.get("horizon_length", data["policy_horizon"].shape[0])), data["policy_horizon"].shape[0])
            for step in range(max(0, horizon_len)):
                for index, value in enumerate(data["policy_horizon"][step].reshape(-1)):
                    name = self._field_name("action_field_names", index)
                    self._log_scalar(f"/policy/prediction/horizon/{name}", float(value), horizon_step=step, prediction_id=prediction_id)
            self._log_scalar("/policy/prediction/id", float(prediction_id))
            self._log_scalar("/policy/prediction/selected_index", float(snapshot.metadata.get("selected_index", -1)))
        if flags & FLAG_SELECTED_ACTION:
            self._log_action("/control/action_selected_raw", data["selected_raw_action"])
        if flags & FLAG_FILTERED_ACTION:
            self._log_action("/control/action_filtered", data["filtered_action"])
        if flags & FLAG_COMMANDED_ACTION and bool(snapshot.metadata.get("commanded_valid", 1)):
            self._log_action("/robot/action_commanded", data["commanded_action"])
        for field in (
            "pointcloud_build_ms",
            "policy_predict_ms",
            "policy_latency_ms",
            "action_age_ms",
            "camera_frame_age_ms",
            "send_duration_ms",
            "cycle_time_ms",
            "loop_overrun_ms",
        ):
            if field in data and flags & FLAG_TIMING:
                self._log_scalar(f"/timing/{field}", float(np.asarray(data[field]).reshape(())))
        self._log_channel_health(snapshot)
        event_text = snapshot.metadata.get("event_text", b"")
        if event_text:
            if isinstance(event_text, bytes):
                event_text = event_text.rstrip(b"\x00").decode("utf-8", errors="replace")
            self._log_text("/events", str(event_text))
        self._log_scalar("/monitor/telemetry_process_alive", 1.0)

    def _log_channel_health(self, snapshot: ChannelSnapshot) -> None:
        name = snapshot.name
        self._log_scalar(f"/monitor/published/{name}", 1.0)
        self._log_scalar(
            f"/monitor/dropped/{name}",
            float(snapshot.metadata.get("dropped_count", 0)),
        )
        self._log_scalar(
            f"/monitor/overwritten/{name}",
            float(snapshot.metadata.get("overwritten_count", 0)),
        )
        if name == "control":
            self._log_scalar(
                "/monitor/consumer_lag_cycles",
                float(snapshot.metadata.get("consumer_lag_cycles", 0)),
            )

    def _set_time(self, metadata: Mapping[str, Any], sequence: int) -> None:
        wall_timestamp = _timestamp_seconds(float(metadata.get("wall_timestamp", 0.0)))
        observation_timestamp = _timestamp_seconds(
            float(metadata.get("observation_timestamp", 0.0)),
            reference=wall_timestamp,
        )
        self._set_timeline("cycle", sequence=int(metadata.get("cycle_id", sequence)))
        self._set_timeline("observation", timestamp=observation_timestamp)
        self._set_timeline("wall_clock", timestamp=wall_timestamp)

    def _set_timeline(
        self,
        name: str,
        *,
        sequence: int | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Set a timeline across the old and current Rerun Python APIs."""

        if (sequence is None) == (timestamp is None):
            raise ValueError("exactly one of sequence or timestamp is required")
        setter = getattr(self.recording, "set_time", None)
        if callable(setter):
            if sequence is not None:
                setter(name, sequence=int(sequence))
            else:
                setter(name, timestamp=float(timestamp))
            return
        # Rerun 0.26.x used set_time_sequence for both sequence-like and
        # timestamp-like values. Keep this fallback for the documented legacy
        # compatibility range.
        legacy = getattr(self.recording, "set_time_sequence", None)
        if callable(legacy):
            value = int(sequence) if sequence is not None else int(max(0, round(float(timestamp) * 1_000_000)))
            legacy(name, value)

    def _log_image(self, path: str, value: Any) -> None:
        self._log(path, self.rr.Image(np.asarray(value)))

    def _log_depth(self, path: str, value: Any, scale: float) -> None:
        try:
            archetype = self.rr.DepthImage(np.asarray(value), meter=scale or 1.0)
        except TypeError:
            archetype = self.rr.DepthImage(np.asarray(value))
        self._log(path, archetype)

    def _log_points(self, path: str, points: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] not in (3, 6):
            return
        if points.shape[1] == 6:
            colors = np.clip(points[:, 3:6], 0.0, 1.0)
            self._log_points3d(path, points[:, :3], colors)
        else:
            self._log_points3d(path, points)

    def _log_points3d(self, path: str, positions: np.ndarray, colors: np.ndarray | None = None) -> None:
        kwargs = {"radii": float(self.config.display.point_radius)}
        if colors is not None:
            kwargs["colors"] = colors
        try:
            self._log(path, self.rr.Points3D(positions, **kwargs))
        except TypeError:
            self._log(path, self.rr.Points3D(positions, **({"colors": colors} if colors is not None else {})))

    def _log_action(self, prefix: str, values: Any) -> None:
        for index, value in enumerate(np.asarray(values).reshape(-1)):
            name = self._field_name("action_field_names", index)
            self._log_scalar(f"{prefix}/{name}", float(value))

    def _field_name(self, key: str, index: int) -> str:
        names = self.static_metadata.get(key, ())
        return str(names[index]) if index < len(names) else str(index)

    def _log_scalar(self, path: str, value: float, *, horizon_step: int | None = None, prediction_id: int | None = None) -> None:
        if horizon_step is not None:
            self._set_timeline("horizon_step", sequence=int(horizon_step))
            if prediction_id is not None:
                self._set_timeline("prediction_id", sequence=int(prediction_id))
        # Scalar was renamed to Scalars in the current Rerun SDK. Prefer the
        # current archetype while retaining compatibility with 0.26.x.
        scalar_type = getattr(self.rr, "Scalars", None) or getattr(self.rr, "Scalar", None)
        if scalar_type is None:
            raise RuntimeError("Rerun SDK provides neither Scalars nor Scalar")
        self._log(path, scalar_type(float(value)))

    def _log_text(self, path: str, text: str) -> None:
        archetype = getattr(self.rr, "TextLog", None)
        self._log(path, archetype(str(text)) if archetype is not None else str(text))

    def _log(self, path: str, archetype: Any) -> None:
        logger = getattr(self.recording, "log", None)
        if callable(logger):
            logger(path, archetype)
        else:
            self.rr.log(path, archetype, recording=self.recording)

    def close(self) -> None:
        if self.connected:
            try:
                self._log_text("/events", "viewer_disconnected")
                self._log_scalar("/monitor/viewer_connected", 0.0)
            except Exception:  # noqa: BLE001
                pass
        # Explicitly close sinks opened by spawn/connect_grpc. Relying on the
        # telemetry process exiting left attached native Viewer processes alive
        # after inference on rerun-sdk 0.34.1.
        if self.recording is not None:
            flusher = getattr(self.recording, "flush", None)
            if callable(flusher):
                try:
                    flusher(timeout_sec=1.0)
                except Exception:  # noqa: BLE001
                    pass
            disconnect = getattr(self.recording, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect()
                except Exception:  # noqa: BLE001
                    pass
            else:
                disconnect = getattr(self.rr, "disconnect", None)
                if callable(disconnect):
                    try:
                        disconnect(recording=self.recording)
                    except Exception:  # noqa: BLE001
                        pass
        if self.viewer_client is not None and not self.config.viewer.detach_process:
            close_viewer = getattr(self.viewer_client, "close", None)
            if callable(close_viewer):
                try:
                    close_viewer()
                except Exception:  # noqa: BLE001
                    pass
        self.viewer_client = None
        self.connected = False


def _jsonish(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)


def _ensure_viewer_on_path() -> None:
    """Make an env-local ``rerun`` CLI discoverable by rr.spawn()."""

    if shutil.which("rerun") is not None:
        return
    candidate = Path(sys.prefix) / "bin" / "rerun"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        os.environ["PATH"] = str(candidate.parent) + os.pathsep + os.environ.get("PATH", "")


def _as_grpc_proxy_url(url: str) -> str:
    """Accept older config values while satisfying current Rerun connect_grpc."""

    value = str(url).rstrip("/")
    return value if value.endswith("/proxy") else value + "/proxy"


def _local_port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.2):
            return True
    except OSError:
        return False


def _timestamp_seconds(value: float, *, reference: float | None = None) -> float:
    """Normalize device epoch timestamps expressed in s/ms/us/ns to seconds."""

    if not np.isfinite(value):
        return 0.0
    if reference is not None and np.isfinite(reference) and reference > 0.0:
        candidates = (value, value / 1_000.0, value / 1_000_000.0, value / 1_000_000_000.0)
        return min(candidates, key=lambda candidate: abs(candidate - reference))
    magnitude = abs(value)
    if magnitude >= 1.0e17:
        return value / 1_000_000_000.0
    if magnitude >= 1.0e14:
        return value / 1_000_000.0
    if magnitude >= 1.0e11:
        return value / 1_000.0
    return value


__all__ = ["RerunSink"]
