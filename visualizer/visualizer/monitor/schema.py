"""Fixed-field telemetry schema shared by the producer and telemetry child."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

FLAG_STATE = 1 << 0
FLAG_CAMERA = 1 << 1
FLAG_SAMPLED = 1 << 2
FLAG_RAW = 1 << 3
FLAG_CROPPED = 1 << 4
FLAG_HORIZON = 1 << 5
FLAG_SELECTED_ACTION = 1 << 6
FLAG_FILTERED_ACTION = 1 << 7
FLAG_COMMANDED_ACTION = 1 << 8
FLAG_TIMING = 1 << 9
FLAG_RGB = 1 << 10
FLAG_DEPTH = 1 << 11

STATUS_UNKNOWN = 0
STATUS_SENT = 1
STATUS_SEND_SKIPPED = 2
STATUS_SEND_ERROR = 3
STATUS_TIMING_SKIP = 4
STATUS_STOP_FILE = 5

META_DTYPE = np.dtype(
    [
        ("sequence", "<i8"),
        ("consumed_sequence", "<i8"),
        ("cycle_id", "<i8"),
        ("observation_timestamp", "<f8"),
        ("monotonic_timestamp", "<f8"),
        ("wall_timestamp", "<f8"),
        ("valid_flags", "<u4"),
        ("status", "<i4"),
        ("event_text", "S256"),
        ("point_count", "<i8"),
        ("point_count_2", "<i8"),
        ("frame_index", "<i8"),
        ("prediction_id", "<i8"),
        ("horizon_length", "<i4"),
        ("selected_index", "<i4"),
        ("commanded_valid", "<i1"),
        ("depth_scale", "<f8"),
        ("dropped_count", "<i8"),
        ("overwritten_count", "<i8"),
    ],
    align=True,
)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    fields: tuple[FieldSpec, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "fields": [field.as_dict() for field in self.fields]}


@dataclass(frozen=True)
class TelemetrySchema:
    channels: tuple[ChannelSpec, ...]

    @classmethod
    def from_shapes(cls, shapes: Any) -> "TelemetrySchema":
        h, w = int(shapes.camera_height), int(shapes.camera_width)
        n, d = int(shapes.point_count), int(shapes.point_dim)
        state, action, horizon = int(shapes.state_dim), int(shapes.action_dim), int(shapes.policy_horizon)
        return cls(
            channels=(
                ChannelSpec(
                    "control",
                    tuple(
                        FieldSpec(name, shape, dtype)
                        for name, shape, dtype in (
                            ("measured_state", (state,), "float32"),
                            ("policy_horizon", (horizon, action), "float32"),
                            ("selected_raw_action", (action,), "float32"),
                            ("filtered_action", (action,), "float32"),
                            ("commanded_action", (action,), "float32"),
                            ("pointcloud_build_ms", (), "float32"),
                            ("policy_predict_ms", (), "float32"),
                            ("policy_latency_ms", (), "float32"),
                            ("action_age_ms", (), "float32"),
                            ("camera_frame_age_ms", (), "float32"),
                            ("send_duration_ms", (), "float32"),
                            ("cycle_time_ms", (), "float32"),
                            ("loop_overrun_ms", (), "float32"),
                        )
                    ),
                ),
                ChannelSpec(
                    "camera",
                    (
                        FieldSpec("rgb", (h, w, 3), "uint8"),
                        FieldSpec("depth", (h, w), str(shapes.depth_dtype)),
                    ),
                ),
                ChannelSpec(
                    "sampled_pointcloud",
                    (FieldSpec("points", (n, d), "float32"),),
                ),
                ChannelSpec(
                    "stage_pointcloud",
                    (
                        FieldSpec("raw_points", (int(shapes.max_raw_points), d), "float32"),
                        FieldSpec("cropped_points", (int(shapes.max_cropped_points), d), "float32"),
                    ),
                ),
            )
        )

    def channel(self, name: str) -> ChannelSpec:
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise KeyError(f"Unknown telemetry channel: {name}")

    def as_dict(self) -> dict[str, Any]:
        return {"channels": [channel.as_dict() for channel in self.channels]}


@dataclass
class ChannelSnapshot:
    name: str
    sequence: int
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


__all__ = [
    "ChannelSnapshot",
    "ChannelSpec",
    "FieldSpec",
    "META_DTYPE",
    "STATUS_SEND_ERROR",
    "STATUS_SEND_SKIPPED",
    "STATUS_SENT",
    "STATUS_STOP_FILE",
    "STATUS_TIMING_SKIP",
    "STATUS_UNKNOWN",
    "TelemetrySchema",
    "FLAG_CAMERA",
    "FLAG_COMMANDED_ACTION",
    "FLAG_CROPPED",
    "FLAG_DEPTH",
    "FLAG_FILTERED_ACTION",
    "FLAG_HORIZON",
    "FLAG_RAW",
    "FLAG_RGB",
    "FLAG_SAMPLED",
    "FLAG_SELECTED_ACTION",
    "FLAG_STATE",
    "FLAG_TIMING",
]
