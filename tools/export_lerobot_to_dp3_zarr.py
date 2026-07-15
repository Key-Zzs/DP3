#!/usr/bin/env python
"""Export a local LeRobot RGB-D dataset to the DP3 zarr replay-buffer format."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

DP3_ROOT = TOOLS_DIR.parent / "3D-Diffusion-Policy"
if str(DP3_ROOT) not in sys.path:
    sys.path.insert(0, str(DP3_ROOT))

import lerobot_rgbd_source as rgbd_source
from diffusion_policy_3d.common.flexiv_state_contract import (
    FLEXIV_ACTION_DIM,
    FLEXIV_LEGACY_STATE_SCHEMA,
    FLEXIV_LEGACY_STATE_DIM,
    FLEXIV_STATE_DIM,
    FLEXIV_STATE_ROTATION_REFERENCE,
    FLEXIV_STATE_ROTATION_REPRESENTATION,
    FLEXIV_STATE_SCHEMA,
    FLEXIV_ROTATION6D_CONVENTION,
    FLEXIV_ROTATION6D_ORDER,
    build_flexiv_state_schema,
    convert_legacy_abs_rotvec_state,
    flexiv_action_names,
    flexiv_legacy_state_names,
    flexiv_state_names,
    rotation_matrix_to_rot6d,
    validate_flexiv_state_rotation6d,
)


CAMERA_SPECS = rgbd_source.CAMERA_SPECS

STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"
STATE_DIM = FLEXIV_STATE_DIM
ACTION_DIM = FLEXIV_ACTION_DIM
STATE_FIELD_NAMES = tuple(flexiv_state_names())
LEGACY_STATE_FIELD_NAMES = tuple(flexiv_legacy_state_names())
ACTION_FIELD_NAMES = tuple(flexiv_action_names())
TARGET_STATE_SCHEMA = FLEXIV_STATE_SCHEMA
LEGACY_CONVERTER_NAME = "legacy_abs_rotvec_to_abs_rot6d"
DEFAULT_OUTPUT_ROOT = Path.home() / ".cache" / "dp3_zarr"
LEROBOT_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot"
EXPORT_STATUS_ATTR = "export_status"
EXPORT_STATUS_IN_PROGRESS = "in_progress"
EXPORT_STATUS_COMPLETE = "complete"
EXPECTED_FRAMES_ATTR = "expected_total_frames"
CONVERTED_FRAMES_ATTR = "converted_frames"
INTEGRITY_ATTR = "integrity"


@dataclass(frozen=True)
class SourceStateContract:
    schema: str
    transform: str
    state_dim: int
    state_names: tuple[str, ...]


def convert_legacy_abs_rotvec_to_v2(state: Any) -> np.ndarray:
    """Public, explicit offline converter for the supported Flexiv v1 source."""

    return convert_legacy_abs_rotvec_state(state)


def _feature_shape(feature: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(feature, dict) or "shape" not in feature:
        raise ValueError(f"LeRobot metadata is missing {label} feature shape")
    try:
        return tuple(int(value) for value in feature["shape"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LeRobot metadata has invalid {label} feature shape") from exc


def _feature_names(feature: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(feature, dict) or not isinstance(feature.get("names"), (list, tuple)):
        raise ValueError(
            f"LeRobot metadata is missing exact {label} names/order; "
            "dimension-only schema detection is forbidden"
        )
    names = tuple(str(name) for name in feature["names"])
    if any(not name for name in names):
        raise ValueError(f"LeRobot metadata contains an empty {label} field name")
    return names


def _state_schema_metadata(info: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for key in ("robot_state_schema", "state_schema_metadata"):
        value = info.get(key)
        if value is not None:
            if isinstance(value, dict):
                candidates.append(value)
            elif isinstance(value, str) and value.strip():
                candidates.append({"state_schema": value.strip()})
            else:
                raise ValueError(
                    f"LeRobot metadata {key!r} must be a mapping or schema string"
                )
    value = info.get("state_schema")
    if isinstance(value, dict):
        candidates.append(value)
    elif isinstance(value, str) and value.strip():
        candidates.append({"state_schema": value.strip()})
    elif value is not None:
        raise ValueError(
            "LeRobot metadata 'state_schema' must be a mapping or schema string"
        )
    top_level_contract = {
        key: info[key]
        for key in (
            "state_dim",
            "action_dim",
            "state_names",
            "action_names",
            "state_rotation_representation",
            "state_rotation_reference",
            "rotation6d_convention",
            "rotation6d_order",
            "action_rotation_representation",
        )
        if key in info
    }
    if top_level_contract:
        candidates.append(top_level_contract)
    return candidates


def _validate_state_schema_metadata(
    metadata: dict[str, Any],
    *,
    source_schema: str,
    state_names: tuple[str, ...],
    action_names: tuple[str, ...],
) -> None:
    declared_schema = metadata.get("state_schema")
    legacy_aliases = {
        FLEXIV_LEGACY_STATE_SCHEMA,
        "flexiv_abs_rotvec",
        "flexiv_physical_v1",
    }
    allowed_schemas = (
        {TARGET_STATE_SCHEMA}
        if source_schema == TARGET_STATE_SCHEMA
        else legacy_aliases
    )
    if declared_schema is not None and declared_schema not in allowed_schemas:
        raise ValueError(
            "LeRobot state schema metadata conflicts with feature names: "
            f"declared={declared_schema!r}, detected={source_schema!r}"
        )
    expected_names = STATE_FIELD_NAMES if source_schema == TARGET_STATE_SCHEMA else LEGACY_STATE_FIELD_NAMES
    if "state_names" in metadata and tuple(metadata["state_names"]) != expected_names:
        raise ValueError("LeRobot state schema metadata state_names/order is not exact")
    if "action_names" in metadata and tuple(metadata["action_names"]) != ACTION_FIELD_NAMES:
        raise ValueError("LeRobot state schema metadata action_names/order is not exact")
    if source_schema == TARGET_STATE_SCHEMA:
        for key, expected in (
            ("state_dim", STATE_DIM),
            ("action_dim", ACTION_DIM),
            ("state_rotation_representation", FLEXIV_STATE_ROTATION_REPRESENTATION),
            ("state_rotation_reference", FLEXIV_STATE_ROTATION_REFERENCE),
            ("rotation6d_convention", FLEXIV_ROTATION6D_CONVENTION),
            ("rotation6d_order", list(FLEXIV_ROTATION6D_ORDER)),
            ("action_rotation_representation", "rotvec"),
        ):
            if key in metadata and metadata[key] != expected:
                raise ValueError(
                    f"LeRobot state schema metadata {key}={metadata[key]!r} "
                    f"does not match {expected!r}"
                )
    else:
        for key, expected in (
            ("state_dim", FLEXIV_LEGACY_STATE_DIM),
            ("action_dim", ACTION_DIM),
            ("state_rotation_representation", "absolute_rotvec"),
            ("state_rotation_reference", FLEXIV_STATE_ROTATION_REFERENCE),
            ("action_rotation_representation", "rotvec"),
        ):
            if key in metadata and metadata[key] != expected:
                raise ValueError(
                    f"Legacy Flexiv metadata {key}={metadata[key]!r} "
                    f"does not match {expected!r}"
                )
        if "rotation6d_convention" in metadata:
            raise ValueError(
                "Legacy Flexiv metadata must not declare a rotation-6D convention"
            )
        if "rotation6d_order" in metadata:
            raise ValueError(
                "Legacy Flexiv metadata must not declare a rotation-6D order"
            )


def detect_source_state_contract(
    info: dict[str, Any],
    *,
    target_state_schema: str = TARGET_STATE_SCHEMA,
    allow_legacy_conversion: bool = False,
) -> SourceStateContract:
    """Detect v2 or explicitly identifiable v1 metadata before reading rows."""

    if target_state_schema != TARGET_STATE_SCHEMA:
        raise ValueError(
            f"Only target state schema {TARGET_STATE_SCHEMA!r} is supported, "
            f"got {target_state_schema!r}"
        )
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("LeRobot meta/info.json is missing the features mapping")
    state_feature = features.get(STATE_COLUMN)
    action_feature = features.get(ACTION_COLUMN)
    state_shape = _feature_shape(state_feature, label=STATE_COLUMN)
    action_shape = _feature_shape(action_feature, label=ACTION_COLUMN)
    state_names = _feature_names(state_feature, label=STATE_COLUMN)
    action_names = _feature_names(action_feature, label=ACTION_COLUMN)
    if action_shape != (ACTION_DIM,) or action_names != ACTION_FIELD_NAMES:
        raise ValueError(
            "LeRobot action schema mismatch: expected exact 14D Flexiv delta-rotvec "
            f"names/order, got shape={action_shape}, names={action_names!r}"
        )

    if state_names == STATE_FIELD_NAMES:
        detected_schema = TARGET_STATE_SCHEMA
        expected_dim = STATE_DIM
        transform = "passthrough_v2"
    elif state_names == LEGACY_STATE_FIELD_NAMES:
        detected_schema = FLEXIV_LEGACY_STATE_SCHEMA
        expected_dim = FLEXIV_LEGACY_STATE_DIM
        transform = LEGACY_CONVERTER_NAME
    else:
        raise ValueError(
            "Unknown Flexiv state schema: exact state names/order do not match either "
            f"v2 ({STATE_DIM}D) or the supported legacy v1 ({FLEXIV_LEGACY_STATE_DIM}D). "
            "The exporter will not infer semantics from array dimensions."
        )
    if state_shape != (expected_dim,):
        raise ValueError(
            f"Detected {detected_schema!r} by names but metadata shape is {state_shape}; "
            f"expected ({expected_dim},)"
        )
    for metadata in _state_schema_metadata(info):
        _validate_state_schema_metadata(
            metadata,
            source_schema=detected_schema,
            state_names=state_names,
            action_names=action_names,
        )
    if detected_schema == FLEXIV_LEGACY_STATE_SCHEMA and not allow_legacy_conversion:
        raise ValueError(
            "Detected the supported legacy Flexiv 28D absolute-rotvec schema, but legacy "
            "conversion was not explicitly enabled. Re-run with "
            "--allow-legacy-state-conversion; unknown 28D data is never guessed."
        )
    return SourceStateContract(
        schema=detected_schema,
        transform=transform,
        state_dim=expected_dim,
        state_names=state_names,
    )


def convert_source_state(
    state: Any,
    source_contract: SourceStateContract,
) -> np.ndarray:
    """Apply only the conversion selected by ``detect_source_state_contract``."""

    values = np.asarray(state, dtype=np.float32)
    if values.shape != (source_contract.state_dim,):
        raise ValueError(
            f"source state shape {values.shape} != ({source_contract.state_dim},) "
            f"for {source_contract.schema}"
        )
    if source_contract.schema == TARGET_STATE_SCHEMA:
        validate_flexiv_state_rotation6d(values, context="source v2 state")
        return values.copy()
    if source_contract.schema == FLEXIV_LEGACY_STATE_SCHEMA:
        if source_contract.transform != LEGACY_CONVERTER_NAME:
            raise ValueError(f"Unsupported legacy state transform {source_contract.transform!r}")
        return convert_legacy_abs_rotvec_to_v2(values)
    raise ValueError(f"Unsupported source state schema {source_contract.schema!r}")


def _with_v2_contract_attrs(attrs: dict[str, Any] | None) -> dict[str, Any]:
    """Fill and validate the mandatory v2 metadata for every DP3 output."""

    output = dict(attrs or {})
    required = build_flexiv_state_schema()
    required.update(
        {
            "state_schema": TARGET_STATE_SCHEMA,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "state_names": list(STATE_FIELD_NAMES),
            "action_names": list(ACTION_FIELD_NAMES),
            "state_rotation_representation": FLEXIV_STATE_ROTATION_REPRESENTATION,
            "state_rotation_reference": FLEXIV_STATE_ROTATION_REFERENCE,
            "rotation6d_convention": FLEXIV_ROTATION6D_CONVENTION,
            "rotation6d_order": list(FLEXIV_ROTATION6D_ORDER),
            "action_rotation_representation": "rotvec",
        }
    )
    for key, expected in required.items():
        if key in output and output[key] != expected:
            raise ValueError(
                f"DP3 v2 metadata {key}={output[key]!r} conflicts with {expected!r}"
            )
        output.setdefault(key, expected)

    source_schema = output.setdefault("source_state_schema", TARGET_STATE_SCHEMA)
    if source_schema not in {TARGET_STATE_SCHEMA, FLEXIV_LEGACY_STATE_SCHEMA}:
        raise ValueError(f"Unknown DP3 source_state_schema: {source_schema!r}")
    expected_transform = (
        "passthrough_v2"
        if source_schema == TARGET_STATE_SCHEMA
        else LEGACY_CONVERTER_NAME
    )
    expected_source_names = (
        STATE_FIELD_NAMES
        if source_schema == TARGET_STATE_SCHEMA
        else LEGACY_STATE_FIELD_NAMES
    )
    for key, expected in (
        ("state_transform", expected_transform),
        ("source_state_names", list(expected_source_names)),
    ):
        if key in output and output[key] != expected:
            raise ValueError(
                f"DP3 source metadata {key}={output[key]!r} conflicts with {expected!r}"
            )
        output.setdefault(key, expected)
    output.setdefault("source_fps", None)
    return output


def build_pointcloud_builder_config(
    realsense_calibration: dict[str, Any],
    *,
    camera: str,
    pointcloud_mode: str,
    num_points: int,
) -> dict[str, Any]:
    """Build the YAML mapping consumed by PointCloudBuilder.from_yaml."""

    if camera not in CAMERA_SPECS:
        raise ValueError(f"Unsupported camera: {camera}")
    if pointcloud_mode not in {"xyz", "xyzrgb"}:
        raise ValueError("pointcloud_mode must be 'xyz' or 'xyzrgb'")
    camera_key = CAMERA_SPECS[camera]["calibration_key"]
    cameras = realsense_calibration.get("cameras")
    if not isinstance(cameras, dict) or camera_key not in cameras:
        available = sorted(cameras.keys()) if isinstance(cameras, dict) else []
        raise KeyError(f"Missing calibration camera '{camera_key}'. Available: {available}")

    camera_calibration = cameras[camera_key]
    config: dict[str, Any] = {
        "camera": {
            "name": camera,
            "aligned_depth_to_color": False,
            "depth_scale": float(camera_calibration["depth_scale_m_per_unit"]),
            "depth_intrinsics": _extract_intrinsics(camera_calibration, "depth"),
            "color_intrinsics": _extract_intrinsics(camera_calibration, "color"),
        },
        "pointcloud": {
            "use_rgb": pointcloud_mode == "xyzrgb",
            "output_format": pointcloud_mode,
        },
        "sampling": {
            "enabled": True,
            "mode": "voxel_random",
            "num_points": int(num_points),
            "pad_mode": "repeat",
        },
    }
    if pointcloud_mode == "xyzrgb":
        config["camera"]["depth_to_color_extrinsics"] = _extract_depth_to_color_extrinsics(
            camera_calibration
        )
        config["pointcloud"].update(
            {
                "rgb_mapping": "project_depth_to_color",
                "rgb_sampling": "nearest",
                "xyz_frame": "depth",
            }
        )
    return config


def compute_episode_ends(
    *,
    episode_rows: list[dict[str, Any]] | None = None,
    episode_indices: list[int] | np.ndarray | None = None,
    total_frames: int,
    max_frames: int | None = None,
) -> np.ndarray:
    """Return DP3-style cumulative episode ends, clipped by max_frames."""

    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    target_frames = min(int(max_frames), total_frames) if max_frames is not None else total_frames
    if target_frames <= 0:
        raise ValueError("max_frames must leave at least one frame")

    if episode_rows:
        sorted_rows = sorted(episode_rows, key=lambda row: int(row["episode_index"]))
        if "dataset_to_index" in sorted_rows[0]:
            raw_ends = [int(row["dataset_to_index"]) for row in sorted_rows]
        else:
            cumulative = 0
            raw_ends = []
            for row in sorted_rows:
                cumulative += int(row["length"])
                raw_ends.append(cumulative)
        if raw_ends[-1] != total_frames:
            raise ValueError(
                f"Episode metadata ends at {raw_ends[-1]}, but data has {total_frames} frames"
            )
    elif episode_indices is not None:
        indices = np.asarray(episode_indices, dtype=np.int64)
        if indices.shape[0] != total_frames:
            raise ValueError(
                f"episode_indices length {indices.shape[0]} does not match total_frames {total_frames}"
            )
        raw_ends = []
        for idx in range(1, len(indices)):
            if indices[idx] != indices[idx - 1]:
                raw_ends.append(idx)
        raw_ends.append(total_frames)
    else:
        raw_ends = [total_frames]

    clipped: list[int] = []
    for end in raw_ends:
        if end < target_frames:
            clipped.append(int(end))
        else:
            clipped.append(target_frames)
            break
    if not clipped or clipped[-1] != target_frames:
        clipped.append(target_frames)
    return np.asarray(clipped, dtype=np.int64)


def write_dp3_zarr(
    output_zarr: str | Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    point_cloud: np.ndarray,
    episode_ends: np.ndarray,
    attrs: dict[str, Any],
    img: np.ndarray | None = None,
    overwrite: bool = False,
) -> None:
    """Write complete in-memory arrays to a DP3-compatible zarr store."""
    state = np.asarray(state, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        raise ValueError(f"state must have shape (T, {STATE_DIM}), got {state.shape}")
    if action.ndim != 2 or action.shape[1] != ACTION_DIM:
        raise ValueError(f"action must have shape (T, {ACTION_DIM}), got {action.shape}")
    if action.shape[0] != state.shape[0]:
        raise ValueError("state and action must have the same number of frames")
    if point_cloud.ndim != 3 or point_cloud.shape[0] != state.shape[0]:
        raise ValueError(
            "point_cloud must have shape (T, N, C) with the same T as state"
        )
    validate_flexiv_state_rotation6d(state, context="export state")
    _reject_nonfinite(action, "action", -1)
    _reject_nonfinite(point_cloud, "point_cloud", -1)
    output_path = Path(output_zarr).expanduser()
    _validate_output_target(output_path, overwrite=overwrite)
    work_path = _prepare_atomic_output(output_path, overwrite=overwrite)
    total_frames = int(state.shape[0])
    export_attrs = _with_v2_contract_attrs(attrs)
    export_attrs.update(
        {
            EXPORT_STATUS_ATTR: EXPORT_STATUS_IN_PROGRESS,
            EXPECTED_FRAMES_ATTR: total_frames,
        }
    )
    try:
        arrays = _create_output_arrays(
            work_path,
            total_frames=total_frames,
            num_points=int(point_cloud.shape[1]),
            pointcloud_dim=int(point_cloud.shape[2]),
            state_dim=int(state.shape[1]),
            action_dim=int(action.shape[1]),
            episode_ends=episode_ends,
            attrs=export_attrs,
            img_shape=tuple(img.shape[1:]) if img is not None else None,
            overwrite=False,
        )
        arrays["state"][:] = state
        arrays["action"][:] = action
        arrays["point_cloud"][:] = point_cloud
        if img is not None and "img" in arrays:
            arrays["img"][:] = img.astype(np.uint8, copy=False)
        integrity = _verify_written_arrays(
            work_path,
            expected_frames=total_frames,
            expected_hashes={
                "state": _numpy_sha256(state, dtype=np.float32),
                "action": _numpy_sha256(action, dtype=np.float32),
                "point_cloud": _numpy_sha256(point_cloud, dtype=np.float32),
            },
        )
        _mark_export_complete(work_path, converted_frames=total_frames, integrity=integrity)
        state_hash = _numpy_sha256(state, dtype=np.float32)
        source_schema = export_attrs["source_state_schema"]
        raw_source_hash = export_attrs.get("raw_source_state_sha256")
        if raw_source_hash is None:
            if source_schema == FLEXIV_LEGACY_STATE_SCHEMA:
                raise ValueError(
                    "write_dp3_zarr requires raw_source_state_sha256 when source_state_schema "
                    "is the legacy Flexiv schema"
                )
            raw_source_hash = state_hash
        if not isinstance(raw_source_hash, str) or len(raw_source_hash) != 64:
            raise ValueError("raw_source_state_sha256 must be a SHA-256 hex digest")
        try:
            int(raw_source_hash, 16)
        except ValueError as exc:
            raise ValueError("raw_source_state_sha256 must be a SHA-256 hex digest") from exc
        if source_schema == TARGET_STATE_SCHEMA and raw_source_hash != state_hash:
            raise ValueError(
                "v2 passthrough state requires raw_source_state_sha256 to match data/state"
            )
        _write_state_provenance(
            work_path,
            source_state_hash=raw_source_hash,
            derived_state_hash=state_hash,
        )
        _commit_atomic_output(work_path, output_path, overwrite=overwrite)
    except BaseException:
        _remove_path(work_path)
        raise


def default_output_zarr_path(
    lerobot_path: str | Path,
    info: dict[str, Any],
    *,
    camera: str,
    pointcloud_mode: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Return the default DP3 zarr path for a LeRobot dataset export."""

    repo_id = _lerobot_repo_id(lerobot_path, info)
    filename = (
        f"{_safe_filename_component(repo_id)}_{camera}_{pointcloud_mode}"
        f"_state_abs_rot6d_v2.zarr"
    )
    return Path(output_root).expanduser() / filename


def export_lerobot_to_dp3_zarr(args: argparse.Namespace) -> dict[str, Any]:
    lerobot_arg = Path(args.lerobot_path).expanduser()
    if not lerobot_arg.is_absolute():
        raise ValueError("--lerobot-path must be an absolute path")
    lerobot_path = lerobot_arg.resolve()
    if not lerobot_path.exists():
        raise FileNotFoundError(f"LeRobot dataset path does not exist: {lerobot_path}")
    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")

    info = _read_json(lerobot_path / "meta" / "info.json")
    source_contract = detect_source_state_contract(
        info,
        target_state_schema=getattr(args, "target_state_schema", TARGET_STATE_SCHEMA),
        allow_legacy_conversion=bool(
            getattr(args, "allow_legacy_state_conversion", False)
        ),
    )
    output_zarr = _resolve_output_zarr(args, lerobot_path=lerobot_path, info=info)
    _validate_output_target(output_zarr, overwrite=bool(args.overwrite))
    data_paths = _data_parquet_paths(lerobot_path)
    total_frames = _count_parquet_rows(data_paths)
    episode_rows = _read_episode_rows(lerobot_path)
    episode_ends = compute_episode_ends(
        episode_rows=episode_rows,
        total_frames=total_frames,
        max_frames=args.max_frames,
    )
    frames_to_export = int(episode_ends[-1])

    sidecar_source = rgbd_source.open_rgbd_sidecar_source(
        lerobot_path,
        source=getattr(args, "rgbd_sidecar_source", "auto"),
        info=info,
        parquet_row_count=total_frames,
        total_episodes=len(episode_rows) if episode_rows else None,
    )
    sidecar_source.validate_join(data_paths, camera=args.camera)
    realsense_calibration = sidecar_source.calibration
    builder_config_path, builder_config = _resolve_builder_config(
        args=args,
        output_zarr=output_zarr,
        realsense_calibration=realsense_calibration,
    )

    PointCloudBuilder = _import_pointcloud_builder()
    builder = PointCloudBuilder.from_yaml(builder_config_path)

    pointcloud_dim = 6 if args.pointcloud_mode == "xyzrgb" else 3
    img_shape = _image_shape_from_info(info, CAMERA_SPECS[args.camera]["video_key"]) if args.save_img else None
    attrs = _zarr_attrs(
        args=args,
        lerobot_path=lerobot_path,
        builder_config_path=builder_config_path,
        builder_config=builder_config,
        realsense_calibration=realsense_calibration,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        pointcloud_dim=pointcloud_dim,
        sidecar_source=sidecar_source,
        source_contract=source_contract,
        source_fps=info.get("fps"),
    )
    attrs.update(
        {
            EXPORT_STATUS_ATTR: EXPORT_STATUS_IN_PROGRESS,
            EXPECTED_FRAMES_ATTR: frames_to_export,
            "source_total_frames": total_frames,
            "raw_depth_scale_m_per_unit": sidecar_source.depth_scale_m_per_unit(args.camera),
            "pointcloud_builder_config_source": (
                "provided" if args.builder_config else "generated_from_calibration"
            ),
        }
    )
    work_zarr = _prepare_atomic_output(output_zarr, overwrite=args.overwrite)
    try:
        arrays = _create_output_arrays(
            work_zarr,
            total_frames=frames_to_export,
            num_points=args.num_points,
            pointcloud_dim=pointcloud_dim,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            episode_ends=episode_ends,
            attrs=attrs,
            img_shape=img_shape,
            overwrite=False,
        )

        camera_spec = CAMERA_SPECS[args.camera]
        need_rgb = args.pointcloud_mode == "xyzrgb" or args.save_img
        video_paths = _video_paths(lerobot_path, camera_spec["video_key"]) if need_rgb else []
        rgb_iter = iter_video_frames(video_paths) if need_rgb else None
        columns = [
            STATE_COLUMN,
            ACTION_COLUMN,
            "global_frame_index",
            camera_spec["timestamp_column"],
            camera_spec["reused_column"],
            "episode_index",
            "frame_index",
            "index",
        ]

        reused_count = 0
        converted = 0
        source_hashers = {
            "state": hashlib.sha256(),
            "action": hashlib.sha256(),
            "point_cloud": hashlib.sha256(),
        }
        raw_source_state_hasher = hashlib.sha256()
        for source_frame in sidecar_source.iter_frames(
            data_paths,
            camera=args.camera,
            columns=columns,
            max_frames=frames_to_export,
        ):
            row = source_frame.row
            source_path = source_frame.source_path
            raw_state = _as_vector(
                row[STATE_COLUMN],
                source_contract.state_dim,
                STATE_COLUMN,
                source_path,
            )
            action = _as_vector(row[ACTION_COLUMN], ACTION_DIM, ACTION_COLUMN, source_path)
            _reject_nonfinite(raw_state, STATE_COLUMN, converted)
            raw_source_state_hasher.update(np.ascontiguousarray(raw_state).tobytes())
            state = convert_source_state(raw_state, source_contract)
            _reject_nonfinite(state, "derived_state", converted)
            _reject_nonfinite(action, ACTION_COLUMN, converted)
            depth = source_frame.depth
            rgb = next(rgb_iter) if rgb_iter is not None else None

            frame = {
                "depth": depth,
                "timestamp": row[camera_spec["timestamp_column"]],
                "global_frame_index": row["global_frame_index"],
            }
            if rgb is not None:
                frame["rgb"] = rgb
            pc_tensor, _meta = builder.from_recorded_frame(frame)
            point_cloud = pc_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            _validate_point_cloud(point_cloud, args.num_points, pointcloud_dim, converted)
            _reject_nonfinite(point_cloud, "point_cloud", converted)

            arrays["state"][converted] = state
            arrays["action"][converted] = action
            arrays["point_cloud"][converted] = point_cloud
            source_hashers["state"].update(np.ascontiguousarray(state).tobytes())
            source_hashers["action"].update(np.ascontiguousarray(action).tobytes())
            source_hashers["point_cloud"].update(np.ascontiguousarray(point_cloud).tobytes())
            if args.save_img:
                if rgb is None:
                    raise RuntimeError("--save-img requested but RGB video frame was not decoded")
                arrays["img"][converted] = rgb

            reused_count += int(bool(row[camera_spec["reused_column"]]))
            converted += 1
            if args.verbose and (
                converted == 1 or converted % 25 == 0 or converted == frames_to_export
            ):
                print(
                    f"[export] {converted}/{frames_to_export} frames, "
                    f"pc={point_cloud.shape}, reused={reused_count}"
                )

        if converted != frames_to_export:
            raise RuntimeError(f"Converted {converted} frames but expected {frames_to_export}")
        integrity = _verify_written_arrays(
            work_zarr,
            expected_frames=frames_to_export,
            expected_hashes={name: hasher.hexdigest() for name, hasher in source_hashers.items()},
        )
        _mark_export_complete(
            work_zarr,
            converted_frames=converted,
            integrity=integrity,
        )
        _write_state_provenance(
            work_zarr,
            source_state_hash=raw_source_state_hasher.hexdigest(),
            derived_state_hash=source_hashers["state"].hexdigest(),
        )
        _commit_atomic_output(work_zarr, output_zarr, overwrite=args.overwrite)
    except BaseException:
        _remove_path(work_zarr)
        raise

    summary = {
        "total_frames": converted,
        "episodes": int(episode_ends.shape[0]),
        "point_cloud_shape": tuple(arrays["point_cloud"].shape),
        "state_shape": tuple(arrays["state"].shape),
        "action_shape": tuple(arrays["action"].shape),
        "reused_frames": reused_count,
        "reused_ratio": reused_count / converted,
        "rgbd_sidecar_storage": sidecar_source.storage,
        "output_zarr": str(output_zarr),
        "builder_config_path": str(builder_config_path),
        "source_state_schema": source_contract.schema,
        "state_transform": source_contract.transform,
    }
    _print_summary(summary)
    return summary


def verify_dp3_zarr(zarr_path: str | Path) -> dict[str, Any]:
    """Verify completion metadata, shapes, finiteness, and stored array hashes."""

    try:
        import zarr
    except ImportError as exc:
        raise ImportError("zarr is required to verify DP3 replay buffers") from exc

    path = Path(zarr_path).expanduser()
    root = zarr.open(str(path), mode="r")
    status = root.attrs.get(EXPORT_STATUS_ATTR)
    if status != EXPORT_STATUS_COMPLETE:
        raise ValueError(
            f"Zarr export is not complete: {EXPORT_STATUS_ATTR}={status!r}, "
            f"expected {EXPORT_STATUS_COMPLETE!r}"
        )
    expected_frames = int(root.attrs.get(EXPECTED_FRAMES_ATTR, -1))
    converted_frames = int(root.attrs.get(CONVERTED_FRAMES_ATTR, -1))
    if expected_frames <= 0 or converted_frames != expected_frames:
        raise ValueError(
            f"Invalid export frame metadata: expected={expected_frames}, "
            f"converted={converted_frames}"
        )
    _validate_v2_zarr_metadata(root.attrs)
    stored_integrity = root.attrs.get(INTEGRITY_ATTR)
    if not isinstance(stored_integrity, dict):
        raise ValueError(f"Missing zarr integrity metadata: {INTEGRITY_ATTR}")
    for key in ("derived_state", "raw_source_state"):
        if key not in stored_integrity:
            raise ValueError(f"Zarr integrity metadata is missing {key}")
    if stored_integrity["derived_state"] != stored_integrity.get("state"):
        raise ValueError("Zarr derived_state integrity does not match data/state")
    if stored_integrity["raw_source_state"] != root.attrs.get("raw_source_state_sha256"):
        raise ValueError("Zarr raw_source_state integrity does not match provenance")
    if root.attrs.get("derived_state_sha256") != stored_integrity["derived_state"]:
        raise ValueError("Zarr derived_state_sha256 does not match integrity metadata")
    if root.attrs.get("exported_state_sha256") != stored_integrity.get("state"):
        raise ValueError("Zarr exported_state_sha256 does not match data/state")
    _verify_written_arrays(
        path,
        expected_frames=expected_frames,
        expected_hashes={
            name: str(stored_integrity[name])
            for name in ("state", "action", "point_cloud")
        },
    )
    return {str(key): str(value) for key, value in stored_integrity.items()}


def iter_lerobot_rows(
    data_paths: list[Path],
    *,
    columns: list[str],
    max_frames: int,
    batch_size: int = 8,
) -> Iterator[tuple[dict[str, Any], Path]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to read LeRobot parquet files") from exc

    remaining = max_frames
    for path in data_paths:
        parquet_file = pq.ParquetFile(path)
        missing = sorted(set(columns) - set(parquet_file.schema_arrow.names))
        if missing:
            raise KeyError(f"Missing parquet columns in {path}: {missing}")
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            data = batch.to_pydict()
            for idx in range(batch.num_rows):
                yield {column: data[column][idx] for column in columns}, path
                remaining -= 1
                if remaining <= 0:
                    return


def iter_video_frames(video_paths: list[Path]) -> Iterator[np.ndarray]:
    if not video_paths:
        raise FileNotFoundError("No RGB video files found")
    try:
        import av

        for path in video_paths:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                for frame in container.decode(stream):
                    yield frame.to_ndarray(format="rgb24")
        return
    except ImportError:
        pass

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Either PyAV (av) or OpenCV (cv2) is required to decode RGB videos") from exc

    for path in video_paths:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise OSError(f"Failed to open video: {path}")
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()


def _resolve_builder_config(
    *,
    args: argparse.Namespace,
    output_zarr: Path,
    realsense_calibration: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    if args.builder_config:
        path = Path(args.builder_config).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Builder config does not exist: {path}")
        return path, _read_yaml(path)

    config = build_pointcloud_builder_config(
        realsense_calibration,
        camera=args.camera,
        pointcloud_mode=args.pointcloud_mode,
        num_points=args.num_points,
    )
    config_path = output_zarr.with_suffix(".pointcloud_builder.yaml").expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return config_path.resolve(), config


def _resolve_output_zarr(
    args: argparse.Namespace,
    *,
    lerobot_path: Path,
    info: dict[str, Any],
) -> Path:
    if args.output_zarr:
        return Path(args.output_zarr).expanduser()
    return default_output_zarr_path(
        lerobot_path,
        info,
        camera=args.camera,
        pointcloud_mode=args.pointcloud_mode,
    )


def _validate_output_target(output_zarr: Path, *, overwrite: bool) -> None:
    """Never replace a pre-v2 store through the new exporter."""

    if not overwrite or not output_zarr.exists():
        return
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("zarr is required to inspect an existing output target") from exc
    try:
        root = zarr.open(str(output_zarr), mode="r")
        existing_schema = root.attrs.get("state_schema")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Refusing to overwrite existing output that is not a validated v2 Zarr: {output_zarr}"
        ) from exc
    if existing_schema != TARGET_STATE_SCHEMA:
        raise ValueError(
            f"Refusing to overwrite non-v2 Zarr {output_zarr}: "
            f"state_schema={existing_schema!r}"
        )


def _create_output_arrays(
    output_zarr: Path,
    *,
    total_frames: int,
    num_points: int,
    pointcloud_dim: int,
    state_dim: int,
    action_dim: int,
    episode_ends: np.ndarray,
    attrs: dict[str, Any],
    img_shape: tuple[int, int, int] | None,
    overwrite: bool,
) -> dict[str, Any]:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("zarr is required to write DP3 replay buffers") from exc

    if int(state_dim) != STATE_DIM:
        raise ValueError(f"DP3 data/state dimension must be {STATE_DIM}, got {state_dim}")
    if int(action_dim) != ACTION_DIM:
        raise ValueError(f"DP3 data/action dimension must be {ACTION_DIM}, got {action_dim}")
    output_zarr = output_zarr.expanduser()
    if output_zarr.exists():
        if not overwrite:
            raise FileExistsError(f"Output zarr already exists: {output_zarr}")
        shutil.rmtree(output_zarr)
    output_zarr.parent.mkdir(parents=True, exist_ok=True)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    root = zarr.group(str(output_zarr))
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    root.attrs.update(_jsonable(attrs))

    time_chunk = max(1, min(128, total_frames))
    arrays = {
        "state": data_group.create_dataset(
            "state",
            shape=(total_frames, state_dim),
            chunks=(time_chunk, state_dim),
            dtype="float32",
            compressor=compressor,
        ),
        "action": data_group.create_dataset(
            "action",
            shape=(total_frames, action_dim),
            chunks=(time_chunk, action_dim),
            dtype="float32",
            compressor=compressor,
        ),
        "point_cloud": data_group.create_dataset(
            "point_cloud",
            shape=(total_frames, num_points, pointcloud_dim),
            chunks=(max(1, min(32, total_frames)), num_points, pointcloud_dim),
            dtype="float32",
            compressor=compressor,
        ),
    }
    meta_group.create_dataset(
        "episode_ends",
        data=np.asarray(episode_ends, dtype=np.int64),
        chunks=(max(1, min(128, episode_ends.shape[0])),),
        dtype="int64",
        compressor=compressor,
    )
    if img_shape is not None:
        arrays["img"] = data_group.create_dataset(
            "img",
            shape=(total_frames, *img_shape),
            chunks=(1, *img_shape),
            dtype="uint8",
            compressor=compressor,
        )
    return arrays


def _prepare_atomic_output(output_zarr: Path, *, overwrite: bool) -> Path:
    output_zarr = output_zarr.expanduser()
    if output_zarr.exists() and not overwrite:
        raise FileExistsError(f"Output zarr already exists: {output_zarr}")
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    work_path = output_zarr.with_name(f".{output_zarr.name}.incomplete-{os.getpid()}")
    _remove_path(work_path)
    return work_path


def _commit_atomic_output(work_path: Path, output_path: Path, *, overwrite: bool) -> None:
    if not work_path.is_dir():
        raise FileNotFoundError(f"Incomplete export directory does not exist: {work_path}")
    output_path = output_path.expanduser()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output zarr already exists: {output_path}")

    backup_path = output_path.with_name(f".{output_path.name}.backup-{os.getpid()}")
    _remove_path(backup_path)
    if output_path.exists():
        output_path.rename(backup_path)
    try:
        work_path.rename(output_path)
    except BaseException:
        if backup_path.exists() and not output_path.exists():
            backup_path.rename(output_path)
        raise
    else:
        _remove_path(backup_path)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _numpy_sha256(array: np.ndarray, *, dtype: Any) -> str:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _zarr_array_sha256(array: Any) -> str:
    hasher = hashlib.sha256()
    rows_per_chunk = int(array.chunks[0]) if array.chunks else 128
    for start in range(0, int(array.shape[0]), rows_per_chunk):
        chunk = np.ascontiguousarray(array[start : start + rows_per_chunk])
        hasher.update(chunk.tobytes())
    return hasher.hexdigest()


def _verify_written_arrays(
    zarr_path: Path,
    *,
    expected_frames: int,
    expected_hashes: dict[str, str],
) -> dict[str, str]:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("zarr is required to verify DP3 replay buffers") from exc

    root = zarr.open(str(zarr_path), mode="r")
    for group_name in ("data", "meta"):
        if group_name not in root:
            raise ValueError(f"Missing zarr group: {group_name}")
    for name in ("state", "action", "point_cloud"):
        if name not in root["data"]:
            raise ValueError(f"Missing zarr array: data/{name}")
        array = root["data"][name]
        if int(array.shape[0]) != int(expected_frames):
            raise ValueError(
                f"data/{name} has {array.shape[0]} frames, expected {expected_frames}"
            )
        if name == "state" and tuple(array.shape[1:]) != (STATE_DIM,):
            raise ValueError(f"data/state must have shape (T, {STATE_DIM}), got {array.shape}")
        if name == "action" and tuple(array.shape[1:]) != (ACTION_DIM,):
            raise ValueError(f"data/action must have shape (T, {ACTION_DIM}), got {array.shape}")
        if name == "point_cloud" and len(array.shape) != 3:
            raise ValueError(f"data/point_cloud must be T x N x C, got {array.shape}")
        if name == "state":
            validate_flexiv_state_rotation6d(
                np.asarray(array[:]),
                context="Zarr data/state",
            )
        actual_hash = _zarr_array_sha256(array)
        expected_hash = expected_hashes.get(name)
        if actual_hash != expected_hash:
            raise ValueError(
                f"data/{name} checksum mismatch: actual={actual_hash}, expected={expected_hash}"
            )

    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if episode_ends.ndim != 1 or episode_ends.size == 0:
        raise ValueError("meta/episode_ends must be a non-empty vector")
    if not np.all(np.diff(episode_ends) > 0):
        raise ValueError("meta/episode_ends must be strictly increasing")
    if int(episode_ends[-1]) != int(expected_frames):
        raise ValueError(
            f"episode_ends[-1]={episode_ends[-1]} does not match {expected_frames} frames"
        )
    return {name: expected_hashes[name] for name in ("state", "action", "point_cloud")}


def _mark_export_complete(
    zarr_path: Path,
    *,
    converted_frames: int,
    integrity: dict[str, str],
) -> None:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("zarr is required to finalize DP3 replay buffers") from exc

    root = zarr.open(str(zarr_path), mode="a")
    root.attrs.update(
        {
            CONVERTED_FRAMES_ATTR: int(converted_frames),
            INTEGRITY_ATTR: _jsonable(integrity),
            EXPORT_STATUS_ATTR: EXPORT_STATUS_COMPLETE,
        }
    )


def _write_state_provenance(
    zarr_path: Path,
    *,
    source_state_hash: str,
    derived_state_hash: str,
) -> None:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("zarr is required to write DP3 replay buffers") from exc

    root = zarr.open(str(zarr_path), mode="a")
    root.attrs.update(
        {
            "raw_source_state_sha256": str(source_state_hash),
            "derived_state_sha256": str(derived_state_hash),
            "exported_state_sha256": str(derived_state_hash),
        }
    )
    integrity = root.attrs.get(INTEGRITY_ATTR)
    if not isinstance(integrity, dict):
        raise ValueError("Cannot write state provenance without integrity metadata")
    integrity.update(
        {
            "raw_source_state": str(source_state_hash),
            "derived_state": str(derived_state_hash),
        }
    )
    root.attrs[INTEGRITY_ATTR] = _jsonable(integrity)


def _validate_v2_zarr_metadata(attrs: Any) -> None:
    expected = build_flexiv_state_schema()
    expected.update({"action_rotation_representation": "rotvec"})
    for key, value in expected.items():
        actual = attrs.get(key)
        if actual != value:
            raise ValueError(
                f"Zarr v2 metadata {key}={actual!r} does not match {value!r}"
            )
    if attrs.get("state_transform") not in {
        "passthrough_v2",
        LEGACY_CONVERTER_NAME,
    }:
        raise ValueError(
            "Zarr state_transform must be passthrough_v2 or "
            f"{LEGACY_CONVERTER_NAME}"
        )
    source_schema = attrs.get("source_state_schema")
    if source_schema not in {TARGET_STATE_SCHEMA, FLEXIV_LEGACY_STATE_SCHEMA}:
        raise ValueError(f"Unknown Zarr source_state_schema: {source_schema!r}")
    expected_transform = (
        "passthrough_v2"
        if source_schema == TARGET_STATE_SCHEMA
        else LEGACY_CONVERTER_NAME
    )
    if attrs.get("state_transform") != expected_transform:
        raise ValueError(
            "Zarr source_state_schema/state_transform pairing is invalid: "
            f"schema={source_schema!r}, transform={attrs.get('state_transform')!r}, "
            f"expected={expected_transform!r}"
        )
    source_names = attrs.get("source_state_names")
    expected_source_names = (
        STATE_FIELD_NAMES
        if source_schema == TARGET_STATE_SCHEMA
        else LEGACY_STATE_FIELD_NAMES
    )
    if tuple(source_names or ()) != expected_source_names:
        raise ValueError("Zarr source_state_names/order does not match source_state_schema")
    if "source_fps" not in attrs:
        raise ValueError("Zarr is missing source_fps metadata")
    for key in ("raw_source_state_sha256", "derived_state_sha256", "exported_state_sha256"):
        digest = attrs.get(key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Zarr is missing valid {key} provenance")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"Zarr has invalid {key} provenance") from exc


def _zarr_attrs(
    *,
    args: argparse.Namespace,
    lerobot_path: Path,
    builder_config_path: Path,
    builder_config: dict[str, Any],
    realsense_calibration: dict[str, Any],
    state_dim: int,
    action_dim: int,
    pointcloud_dim: int,
    sidecar_source: rgbd_source.RGBDSidecarSource,
    source_contract: SourceStateContract,
    source_fps: Any,
) -> dict[str, Any]:
    attrs = {
        "source_lerobot_path": str(lerobot_path),
        "camera": args.camera,
        "pointcloud_mode": args.pointcloud_mode,
        "num_points": int(args.num_points),
        "pointcloud_builder_config": builder_config,
        "pointcloud_builder_config_path": str(builder_config_path),
        "realsense_calibration": realsense_calibration,
        "depth_source": "native_depth",
        "aligned_depth_to_color": False,
        "use_rgb": args.pointcloud_mode == "xyzrgb",
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "pointcloud_dim": int(pointcloud_dim),
        "state_schema": TARGET_STATE_SCHEMA,
        "state_names": list(STATE_FIELD_NAMES),
        "action_names": list(ACTION_FIELD_NAMES),
        "state_rotation_representation": FLEXIV_STATE_ROTATION_REPRESENTATION,
        "state_rotation_reference": FLEXIV_STATE_ROTATION_REFERENCE,
        "rotation6d_convention": FLEXIV_ROTATION6D_CONVENTION,
        "rotation6d_order": list(FLEXIV_ROTATION6D_ORDER),
        "action_rotation_representation": "rotvec",
        "source_state_schema": source_contract.schema,
        "source_state_names": list(source_contract.state_names),
        "state_transform": source_contract.transform,
        "source_fps": source_fps,
    }
    attrs.update(sidecar_source.provenance)
    return attrs


def _import_pointcloud_builder() -> Any:
    try:
        from pointcloud_builder import PointCloudBuilder

        return PointCloudBuilder
    except ImportError:
        repo_root = Path(__file__).resolve().parents[1]
        candidate = repo_root / "PointCloudBuilder"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            from pointcloud_builder import PointCloudBuilder

            return PointCloudBuilder
        raise


def _extract_intrinsics(camera_calibration: dict[str, Any], stream_name: str) -> dict[str, Any]:
    stream = camera_calibration.get("streams", {}).get(stream_name)
    if not isinstance(stream, dict):
        raise KeyError(f"Missing RealSense stream calibration: {stream_name}")
    intrinsics = stream.get("intrinsics", stream)
    if not isinstance(intrinsics, dict):
        raise KeyError(f"Missing intrinsics for stream: {stream_name}")
    matrix = intrinsics.get("K")
    if matrix is None:
        matrix = intrinsics.get("k")
    fx, fy, cx, cy = _intrinsic_values_from_mapping(intrinsics, matrix)
    return {
        "width": int(_first_present(intrinsics, stream, "width")),
        "height": int(_first_present(intrinsics, stream, "height")),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def _intrinsic_values_from_mapping(
    intrinsics: dict[str, Any],
    matrix: Any,
) -> tuple[float, float, float, float]:
    if all(key in intrinsics for key in ["fx", "fy", "cx", "cy"]):
        return (
            float(intrinsics["fx"]),
            float(intrinsics["fy"]),
            float(intrinsics["cx"]),
            float(intrinsics["cy"]),
        )
    if matrix is None:
        raise KeyError("Intrinsics must provide fx/fy/cx/cy or K")
    if isinstance(matrix[0], list):
        return (
            float(matrix[0][0]),
            float(matrix[1][1]),
            float(matrix[0][2]),
            float(matrix[1][2]),
        )
    return (
        float(matrix[0]),
        float(matrix[4]),
        float(matrix[2]),
        float(matrix[5]),
    )


def _extract_depth_to_color_extrinsics(camera_calibration: dict[str, Any]) -> dict[str, Any]:
    extrinsics = camera_calibration.get("extrinsics", {}).get("depth_to_color")
    if not isinstance(extrinsics, dict):
        raise KeyError("Missing RealSense extrinsics.depth_to_color")
    rotation = extrinsics.get("rotation_matrix_row_major")
    if rotation is None:
        flat_rotation = extrinsics.get("rotation")
        if flat_rotation is None:
            raise KeyError("Missing depth_to_color rotation")
        rotation = [flat_rotation[0:3], flat_rotation[3:6], flat_rotation[6:9]]
    translation = extrinsics.get("translation_m", extrinsics.get("translation"))
    if translation is None:
        raise KeyError("Missing depth_to_color translation")
    return {
        "rotation": rotation,
        "translation": translation,
    }


def _first_present(primary: dict[str, Any], secondary: dict[str, Any], key: str) -> Any:
    if key in primary:
        return primary[key]
    if key in secondary:
        return secondary[key]
    raise KeyError(f"Missing intrinsics key: {key}")


def _data_parquet_paths(root: Path) -> list[Path]:
    paths = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet files found under {root / 'data'}")
    return paths


def _video_paths(root: Path, video_key: str) -> list[Path]:
    paths = sorted((root / "videos" / video_key).glob("chunk-*/file-*.mp4"))
    if not paths:
        raise FileNotFoundError(f"No RGB video files found for {video_key}")
    return paths


def _count_parquet_rows(paths: list[Path]) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to inspect LeRobot parquet files") from exc
    total = 0
    for path in paths:
        total += pq.ParquetFile(path).metadata.num_rows
    if total <= 0:
        raise ValueError("LeRobot dataset contains no frames")
    return total


def _read_episode_rows(root: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to read LeRobot episode metadata") from exc
    paths = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        return []
    rows: list[dict[str, Any]] = []
    for path in paths:
        table = pq.read_table(path)
        for row_idx in range(table.num_rows):
            row = {}
            for key in ["episode_index", "length", "dataset_from_index", "dataset_to_index"]:
                if key in table.column_names:
                    row[key] = table[key][row_idx].as_py()
            rows.append(row)
    return rows


def _image_shape_from_info(info: dict[str, Any], video_key: str) -> tuple[int, int, int]:
    feature = info.get("features", {}).get(video_key)
    if not isinstance(feature, dict) or "shape" not in feature:
        raise KeyError(f"Missing video shape in meta/info.json for {video_key}")
    shape = tuple(int(x) for x in feature["shape"])
    if len(shape) != 3 or shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB shape for {video_key}, got {shape}")
    return shape


def _as_vector(value: Any, dim: int, column: str, source_path: Path) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (dim,):
        raise ValueError(f"{source_path}: {column} shape {array.shape} != ({dim},)")
    return array


def _as_depth(value: Any, column: str, source_path: Path) -> np.ndarray:
    array = np.asarray(value, dtype=np.uint16)
    if array.ndim != 2:
        raise ValueError(f"{source_path}: {column} must be HxW, got {array.shape}")
    return array


def _validate_point_cloud(point_cloud: np.ndarray, num_points: int, pointcloud_dim: int, frame_idx: int) -> None:
    expected = (num_points, pointcloud_dim)
    if point_cloud.shape != expected:
        raise ValueError(f"frame {frame_idx}: point_cloud shape {point_cloud.shape} != {expected}")


def _reject_nonfinite(array: np.ndarray, name: str, frame_idx: int) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"frame {frame_idx}: {name} contains NaN or Inf")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def _lerobot_repo_id(lerobot_path: str | Path, info: dict[str, Any]) -> str:
    repo_id = info.get("repo_id")
    if isinstance(repo_id, str) and repo_id.strip():
        return repo_id.strip()

    path = Path(lerobot_path).expanduser().resolve()
    try:
        return path.relative_to(LEROBOT_CACHE_ROOT).as_posix()
    except ValueError:
        return path.name


def _safe_filename_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not safe:
        raise ValueError("LeRobot repo_id produced an empty output filename component")
    return safe


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _print_summary(summary: dict[str, Any]) -> None:
    print("Export summary")
    print(f"  total_frames: {summary['total_frames']}")
    print(f"  episodes: {summary['episodes']}")
    print(f"  point_cloud_shape: {summary['point_cloud_shape']}")
    print(f"  state_shape: {summary['state_shape']}")
    print(f"  action_shape: {summary['action_shape']}")
    print(f"  reused_frames: {summary['reused_frames']} ({summary['reused_ratio']:.4%})")
    print(f"  rgbd_sidecar_storage: {summary['rgbd_sidecar_storage']}")
    print(f"  output_zarr: {summary['output_zarr']}")
    print(f"  builder_config_path: {summary['builder_config_path']}")


def _print_failure_diagnostics(args: argparse.Namespace, exc: BaseException) -> None:
    print(f"[export] failed: {exc}", file=sys.stderr)
    root = Path(args.lerobot_path).expanduser()
    print(f"[export] lerobot_path: {root}", file=sys.stderr)
    print(
        f"[export] rgbd_sidecar_source: {getattr(args, 'rgbd_sidecar_source', 'auto')}",
        file=sys.stderr,
    )
    print(
        f"[export] raw sidecar manifest: {root / rgbd_source.RAW_MANIFEST_RELATIVE_PATH}",
        file=sys.stderr,
    )
    data_paths = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    video_key = CAMERA_SPECS.get(args.camera, CAMERA_SPECS["head"])["video_key"]
    video_paths = sorted((root / "videos" / video_key).glob("chunk-*/file-*.mp4"))
    print(f"[export] data parquet files: {[str(p) for p in data_paths]}", file=sys.stderr)
    print(f"[export] attempted video files: {[str(p) for p in video_paths]}", file=sys.stderr)
    try:
        import pyarrow.parquet as pq

        for path in data_paths[:1]:
            print(f"[export] parquet schema for {path}:", file=sys.stderr)
            print(pq.ParquetFile(path).schema_arrow, file=sys.stderr)
    except Exception as schema_exc:
        print(f"[export] parquet schema unavailable: {schema_exc}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-path", required=True, help="Absolute path to a local LeRobot dataset")
    parser.add_argument(
        "--output-zarr",
        help=(
            "Output DP3 zarr path. Defaults to "
            f"{DEFAULT_OUTPUT_ROOT}/<lerobot_repo_id>_<camera>_<pointcloud-mode>"
            "_state_abs_rot6d_v2.zarr"
        ),
    )
    parser.add_argument(
        "--target-state-schema",
        choices=[TARGET_STATE_SCHEMA],
        default=TARGET_STATE_SCHEMA,
        help="Target Flexiv state contract. Only the v2 rotation-6D contract is writable.",
    )
    parser.add_argument(
        "--allow-legacy-state-conversion",
        action="store_true",
        help=(
            "Explicitly enable conversion of an exact Flexiv v1 28D absolute-rotvec "
            f"source using {LEGACY_CONVERTER_NAME}."
        ),
    )
    parser.add_argument("--camera", choices=sorted(CAMERA_SPECS), default="head")
    parser.add_argument(
        "--rgbd-sidecar-source",
        choices=["auto", "zarr", "parquet"],
        default="auto",
        help=(
            "RGB-D/IR source layout. auto requires and validates raw Zarr when "
            "meta/rgbd_sidecar.json exists; otherwise it uses legacy Parquet."
        ),
    )
    parser.add_argument("--pointcloud-mode", choices=["xyz", "xyzrgb"], default="xyz")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--builder-config", help="Optional PointCloudBuilder YAML path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output zarr if it exists")
    parser.add_argument("--max-frames", type=int, help="Only convert the first N frames")
    parser.add_argument("--save-img", action="store_true", help="Also save RGB frames to data/img")
    parser.add_argument("--verbose", action="store_true", help="Print per-frame progress")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    return args


def main() -> None:
    args = parse_args()
    try:
        export_lerobot_to_dp3_zarr(args)
    except Exception as exc:
        _print_failure_diagnostics(args, exc)
        raise


if __name__ == "__main__":
    main()
