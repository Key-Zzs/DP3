"""Lightweight producer-side monitor client.

The client owns only preallocated NumPy scratch buffers and shared-memory
publication.  It never imports the Rerun SDK and never waits for a consumer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .config import MonitorConfig, TelemetryShapes
from .process import TelemetryProcess
from .schema import (
    FLAG_CAMERA,
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
    STATUS_SEND_ERROR,
    STATUS_SEND_SKIPPED,
    STATUS_SENT,
    STATUS_STOP_FILE,
    STATUS_TIMING_SKIP,
    TelemetrySchema,
)
from .shared_ring import SharedMemoryTelemetryBus

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CyclePlan:
    now: float
    control_due: bool
    camera_due: bool
    sampled_pointcloud_due: bool
    stage_pointcloud_due: bool
    health_due: bool


class MonitorClient:
    """Fail-open, latest-only producer API used from the inference process."""

    def __init__(
        self,
        config: MonitorConfig,
        shapes: TelemetryShapes | None = None,
        *,
        bus: SharedMemoryTelemetryBus | None = None,
        process: TelemetryProcess | None = None,
        static_metadata: Mapping[str, Any] | None = None,
        sink_kind: str | None = None,
    ) -> None:
        self.config = config
        self.enabled = bool(config.enabled and bus is not None)
        self.shapes = shapes
        self.bus = bus
        self.process = process
        self.static_metadata = dict(static_metadata or {})
        self.sink_kind = sink_kind or ("rerun" if config.backend == "rerun" else "null")
        self._closed = False
        self._next_due: dict[str, float] = {
            "control": 0.0,
            "camera": 0.0,
            "sampled_pointcloud": 0.0,
            "stage_pointcloud": 0.0,
            "health": 0.0,
        }
        self._last_plan: CyclePlan | None = None
        self._prediction_id = 0
        self._horizon_scratch = (
            np.zeros((int(shapes.policy_horizon), int(shapes.action_dim)), dtype=np.float32)
            if shapes is not None
            else None
        )
        self._stage_scratch = (
            {
                "raw": np.zeros((int(shapes.max_raw_points), int(shapes.point_dim)), dtype=np.float32),
                "cropped": np.zeros((int(shapes.max_cropped_points), int(shapes.point_dim)), dtype=np.float32),
            }
            if shapes is not None
            else {}
        )
        self._stats = {
            "plan_calls": 0,
            "publish_calls": 0,
            "publish_errors": 0,
            "bulk_skipped_slack": 0,
            "control_due": 0,
            "camera_due": 0,
            "sampled_due": 0,
            "stage_due": 0,
        }

    @classmethod
    def disabled(cls, config: MonitorConfig | None = None) -> "MonitorClient":
        return cls(config or MonitorConfig(enabled=False))

    @classmethod
    def create(
        cls,
        config: MonitorConfig,
        shapes: TelemetryShapes,
        *,
        static_metadata: Mapping[str, Any] | None = None,
        sink_kind: str | None = None,
        start_process: bool = True,
    ) -> "MonitorClient":
        if not config.enabled:
            return cls.disabled(config)
        schema = TelemetrySchema.from_shapes(shapes)
        bus = SharedMemoryTelemetryBus.create(schema, capacity=config.ring_capacity)
        process = TelemetryProcess(
            bus.descriptor(),
            config=config,
            static_metadata=static_metadata,
            sink_kind=sink_kind or config.backend,
        )
        client = cls(
            config,
            shapes,
            bus=bus,
            process=process,
            static_metadata=static_metadata,
            sink_kind=sink_kind,
        )
        if start_process:
            try:
                process.start(timeout=config.startup_timeout_sec)
            except Exception as exc:  # noqa: BLE001
                client._stats["publish_errors"] += 1
                LOGGER.warning("Telemetry child failed to start: %s", exc)
                if not config.fail_open:
                    client.close()
                    raise
                # Startup timeout/failure can leave a spawned child and its
                # shared-memory owner alive.  Fail-open means disable the
                # feature after bounded cleanup, not abandon those resources.
                client.close()
                return cls.disabled(config)
        return client

    def plan_cycle(self, now: float | None = None) -> CyclePlan:
        """Compute due flags only; no locks or payload copies are touched."""

        self._poll_process_messages()

        if not self.enabled:
            plan = CyclePlan(float(now if now is not None else time.monotonic()), False, False, False, False, False)
            self._last_plan = plan
            return plan
        now = float(time.monotonic() if now is None else now)
        due: dict[str, bool] = {}
        rates = {
            "control": self.config.rates.control_hz,
            "camera": self.config.rates.camera_hz,
            "sampled_pointcloud": self.config.rates.sampled_pointcloud_hz,
            "stage_pointcloud": self.config.rates.stage_pointcloud_hz,
            "health": self.config.rates.health_hz,
        }
        for key, rate in rates.items():
            interval = 1.0 / float(rate)
            # The producer loop and its sleep deadline are independent clocks;
            # tolerate a sub-millisecond scheduling edge instead of turning a
            # nominal 10 Hz channel into an accidental every-other-cycle gate.
            epsilon = min(0.001, interval * 0.01)
            is_due = now + epsilon >= self._next_due[key]
            due[key] = is_due
            if is_due:
                # Advance once, rather than accumulating a backlog after a slow
                # cycle.  The next call can publish only the current latest.
                scheduled_next = self._next_due[key] + interval
                self._next_due[key] = scheduled_next if scheduled_next > now else now + interval
                self._stats[f"{key if key in {'control', 'camera', 'health'} else key.replace('_pointcloud', '')}_due"] = (
                    self._stats.get(f"{key if key in {'control', 'camera', 'health'} else key.replace('_pointcloud', '')}_due", 0) + 1
                )
        self._stats["plan_calls"] += 1
        plan = CyclePlan(
            now,
            due["control"],
            due["camera"],
            due["sampled_pointcloud"],
            due["stage_pointcloud"],
            due["health"],
        )
        self._last_plan = plan
        return plan

    def publish_cycle(
        self,
        *,
        cycle_id: int,
        measured_state: Any | None = None,
        rgb: Any | None = None,
        depth: Any | None = None,
        depth_scale: float = 0.0,
        frame_index: int = -1,
        sampled_pointcloud: Any | None = None,
        pointcloud_meta: Mapping[str, Any] | None = None,
        stages: Mapping[str, Any] | None = None,
        policy_horizon: Any | None = None,
        prediction_id: int = -1,
        selected_raw_action: Any | None = None,
        filtered_action: Any | None = None,
        commanded_action: Any | None = None,
        commanded_valid: bool = False,
        send_status: str | int = "unknown",
        observation_timestamp: float = 0.0,
        monotonic_timestamp: float | None = None,
        wall_timestamp: float | None = None,
        timings: Mapping[str, float] | None = None,
        selected_action_index: int = -1,
        plan: CyclePlan | None = None,
        remaining_slack_ms: float | None = None,
    ) -> dict[str, Any]:
        """Best-effort publish after ``send_action`` has completed."""

        self._stats["publish_calls"] += 1
        if not self.enabled or self.bus is None:
            return {"enabled": False}
        plan = plan or self._last_plan or self.plan_cycle()
        mono = float(time.monotonic() if monotonic_timestamp is None else monotonic_timestamp)
        wall = float(time.time() if wall_timestamp is None else wall_timestamp)
        result: dict[str, Any] = {"enabled": True, "control": False, "bulk": {}}
        result["publish_timings_ms"] = {}
        try:
            control_started = time.perf_counter() if plan.control_due else None
            horizon_array: np.ndarray | None = None
            horizon_length = 0
            if plan.control_due and self.config.payloads.policy_horizon and policy_horizon is not None:
                source_horizon = _array(policy_horizon, np.float32)
                if source_horizon.ndim != 2:
                    raise ValueError(f"policy horizon must be 2D, got {source_horizon.shape}")
                horizon_length = int(source_horizon.shape[0])
                horizon_array = _fit_horizon(source_horizon, self.shapes, self._horizon_scratch)
            if plan.control_due:
                arrays: dict[str, Any] = {}
                flags = 0
                if self.config.payloads.state and measured_state is not None:
                    arrays["measured_state"] = _array(measured_state, np.float32)
                    flags |= FLAG_STATE
                if self.config.payloads.policy_horizon and horizon_array is not None:
                    arrays["policy_horizon"] = horizon_array
                    flags |= FLAG_HORIZON
                if self.config.payloads.selected_action and selected_raw_action is not None:
                    arrays["selected_raw_action"] = _array(selected_raw_action, np.float32)
                    flags |= FLAG_SELECTED_ACTION
                if self.config.payloads.filtered_action and filtered_action is not None:
                    arrays["filtered_action"] = _array(filtered_action, np.float32)
                    flags |= FLAG_FILTERED_ACTION
                if self.config.payloads.commanded_action and commanded_action is not None:
                    arrays["commanded_action"] = _array(commanded_action, np.float32)
                    flags |= FLAG_COMMANDED_ACTION
                for key, value in (timings or {}).items():
                    if key not in {field.name for field in self.bus.schema.channel("control").fields}:
                        continue
                    arrays[key] = np.asarray(value, dtype=np.float32).reshape(())
                if timings:
                    flags |= FLAG_TIMING
                publish = self.bus.publish(
                    "control",
                    arrays,
                    cycle_id=cycle_id,
                    observation_timestamp=observation_timestamp,
                    monotonic_timestamp=mono,
                    wall_timestamp=wall,
                    valid_flags=flags,
                    status=_status_value(send_status),
                    point_count=int((pointcloud_meta or {}).get("num_raw_points", 0) or 0),
                    point_count_2=int((pointcloud_meta or {}).get("num_cropped_points", 0) or 0),
                    prediction_id=prediction_id,
                    horizon_length=horizon_length,
                    selected_index=int(selected_action_index),
                    commanded_valid=commanded_valid,
                )
                assert control_started is not None
                result["publish_timings_ms"]["control"] = (time.perf_counter() - control_started) * 1000.0
                result["control"] = publish.committed
                result["control_sequence"] = publish.sequence

            bulk_allowed = remaining_slack_ms is None or float(remaining_slack_ms) >= self.config.min_bulk_slack_ms
            if not bulk_allowed:
                self._stats["bulk_skipped_slack"] += 1
                return result
            if plan.camera_due and (self.config.payloads.rgb or self.config.payloads.depth):
                started = time.perf_counter()
                arrays = {}
                flags = 0
                if self.config.payloads.rgb and rgb is not None:
                    arrays["rgb"] = _camera_rgb(rgb)
                    flags |= FLAG_RGB
                if self.config.payloads.depth and depth is not None:
                    arrays["depth"] = _camera_depth(depth)
                    flags |= FLAG_DEPTH
                if arrays:
                    p = self.bus.publish(
                        "camera",
                        arrays,
                        cycle_id=cycle_id,
                        observation_timestamp=observation_timestamp,
                        monotonic_timestamp=mono,
                        wall_timestamp=wall,
                        valid_flags=flags | FLAG_CAMERA,
                        frame_index=frame_index,
                        depth_scale=depth_scale,
                    )
                    result["publish_timings_ms"]["camera"] = (time.perf_counter() - started) * 1000.0
                    result["bulk"]["camera"] = p.committed
            if plan.sampled_pointcloud_due and self.config.payloads.sampled_pointcloud and sampled_pointcloud is not None:
                started = time.perf_counter()
                points = _sampled_points(sampled_pointcloud)
                p = self.bus.publish(
                    "sampled_pointcloud",
                    {"points": points},
                    cycle_id=cycle_id,
                    observation_timestamp=observation_timestamp,
                    monotonic_timestamp=mono,
                    wall_timestamp=wall,
                    valid_flags=FLAG_SAMPLED,
                    point_count=int((pointcloud_meta or {}).get("num_sampled_points", points.shape[0]) or points.shape[0]),
                )
                result["publish_timings_ms"]["sampled_pointcloud"] = (time.perf_counter() - started) * 1000.0
                result["bulk"]["sampled_pointcloud"] = p.committed
            if plan.stage_pointcloud_due and self.config.stages_enabled and stages is not None:
                started = time.perf_counter()
                arrays = {}
                flags = 0
                if self.config.payloads.raw_pointcloud and stages.get("raw") is not None:
                    arrays["raw_points"] = _stage_points(
                        stages["raw"],
                        self.shapes.max_raw_points if self.shapes else None,
                        scratch=self._stage_scratch.get("raw"),
                    )
                    flags |= FLAG_RAW
                if self.config.payloads.cropped_pointcloud and stages.get("cropped") is not None:
                    arrays["cropped_points"] = _stage_points(
                        stages["cropped"],
                        self.shapes.max_cropped_points if self.shapes else None,
                        scratch=self._stage_scratch.get("cropped"),
                    )
                    flags |= FLAG_CROPPED
                if arrays:
                    p = self.bus.publish(
                        "stage_pointcloud",
                        arrays,
                        cycle_id=cycle_id,
                        observation_timestamp=observation_timestamp,
                        monotonic_timestamp=mono,
                        wall_timestamp=wall,
                        valid_flags=flags,
                        point_count=int((pointcloud_meta or {}).get("num_raw_points", 0) or 0),
                        point_count_2=int((pointcloud_meta or {}).get("num_cropped_points", 0) or 0),
                    )
                    result["publish_timings_ms"]["stage_pointcloud"] = (time.perf_counter() - started) * 1000.0
                    result["bulk"]["stage_pointcloud"] = p.committed
        except Exception as exc:  # noqa: BLE001
            # Monitoring must never alter safety or send semantics.
            self._stats["publish_errors"] += 1
            LOGGER.debug("monitor publish failed", exc_info=True)
            result["error"] = str(exc)
        return result

    def publish_event(self, *, cycle_id: int, status: str, message: str) -> None:
        """Best-effort small control event for skip/error paths."""

        if not self.enabled or self.bus is None:
            return
        try:
            self.bus.publish(
                "control",
                cycle_id=cycle_id,
                monotonic_timestamp=time.monotonic(),
                wall_timestamp=time.time(),
                valid_flags=0,
                status=_status_value(status),
                event_message=message,
            )
        except Exception:  # noqa: BLE001
            self._stats["publish_errors"] += 1

    def stats(self) -> dict[str, Any]:
        self._poll_process_messages()
        result = dict(self._stats)
        result["bus"] = {} if self.bus is None else self.bus.stats()
        result["telemetry_alive"] = bool(self.process and self.process.is_alive())
        result["telemetry_heartbeat"] = None if self.process is None else self.process.last_heartbeat
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process is not None:
            try:
                self.process.close(timeout=self.config.shutdown_timeout_sec)
            except Exception:  # noqa: BLE001
                LOGGER.warning("Telemetry child cleanup failed", exc_info=True)
        if self.bus is not None:
            self.bus.close()
            self.bus.unlink()

    def _poll_process_messages(self) -> None:
        if self.process is None:
            return
        try:
            for message in self.process.poll_messages():
                kind = message.get("type")
                if kind in {"sink_error", "failed"}:
                    LOGGER.warning("Telemetry child %s: %s", kind, message.get("error", message))
        except Exception:  # noqa: BLE001
            self._stats["publish_errors"] += 1


def _array(value: Any, dtype: Any | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        # This path is intentionally not used for sampled policy input or
        # action horizons in the live launcher; both are already CPU NumPy.
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _camera_rgb(value: Any) -> np.ndarray:
    array = _array(value)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
            array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            array = np.asarray(array, dtype=np.uint8)
    return np.ascontiguousarray(array)


def _camera_depth(value: Any) -> np.ndarray:
    return np.ascontiguousarray(_array(value))


def _sampled_points(value: Any) -> np.ndarray:
    # The live inference path passes prepare_point_cloud()'s CPU NumPy output.
    # Do not add another Tensor.cpu() or device synchronization here.
    array = np.asarray(value, dtype=np.float32)
    return np.ascontiguousarray(array)


def _fit_horizon(
    value: np.ndarray,
    shapes: TelemetryShapes | None,
    scratch: np.ndarray | None = None,
) -> np.ndarray:
    if shapes is None:
        return np.ascontiguousarray(value)
    target = scratch
    if target is None:
        target = np.zeros((int(shapes.policy_horizon), int(shapes.action_dim)), dtype=np.float32)
    target.fill(0.0)
    if value.shape[1] != target.shape[1]:
        raise ValueError(f"policy horizon action dim {value.shape[1]} != {target.shape[1]}")
    count = min(value.shape[0], target.shape[0])
    target[:count] = value[:count]
    return target


def _stage_points(value: Any, max_points: int | None, *, scratch: np.ndarray | None = None) -> np.ndarray:
    array = _array(value, np.float32)
    if array.ndim != 2:
        raise ValueError(f"stage point cloud must be 2D, got {array.shape}")
    if max_points is not None and scratch is not None:
        if scratch.shape[1] != array.shape[1]:
            raise ValueError(f"stage point dimension {array.shape[1]} != scratch {scratch.shape[1]}")
        scratch.fill(0.0)
        if array.shape[0] > int(max_points):
            stride = max(1, int(np.ceil(array.shape[0] / int(max_points))))
            source = array[::stride][: int(max_points)]
        else:
            source = array
        np.copyto(scratch[: source.shape[0]], source, casting="unsafe")
        return scratch
    if max_points is not None and array.shape[0] > int(max_points):
        stride = max(1, int(np.ceil(array.shape[0] / int(max_points))))
        array = array[::stride][: int(max_points)]
    if max_points is not None and array.shape[0] < int(max_points):
        padded = np.zeros((int(max_points), array.shape[1]), dtype=np.float32)
        padded[: array.shape[0]] = array
        array = padded
    return np.ascontiguousarray(array)


def _status_value(value: str | int) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    key = str(value).strip().lower()
    return {
        "sent": STATUS_SENT,
        "send_error": STATUS_SEND_ERROR,
        "skipped_stop_file_after_inference": STATUS_STOP_FILE,
        "skipped_timing_safety": STATUS_TIMING_SKIP,
        "send_skipped": STATUS_SEND_SKIPPED,
        "pending": STATUS_SEND_SKIPPED,
    }.get(key, 0)


__all__ = ["CyclePlan", "MonitorClient"]
