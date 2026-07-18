#!/usr/bin/env python
"""Export a local LeRobot RGB-D dataset to the DP3 zarr replay-buffer format."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
from collections.abc import Iterator, Mapping
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
    FLEXIV_LEGACY_TO_V2_TRANSFORM,
    FLEXIV_RAW_FORCE_DROPPED_STATE_NAMES,
    FLEXIV_RAW_FORCE_STATE_DIM,
    FLEXIV_RAW_FORCE_STATE_NAMES,
    FLEXIV_RAW_FORCE_STATE_SCHEMA,
    FLEXIV_RAW_FORCE_TO_V2_TRANSFORM,
    FLEXIV_STATE_DIM,
    FLEXIV_STATE_ROTATION_REFERENCE,
    FLEXIV_STATE_ROTATION_REPRESENTATION,
    FLEXIV_STATE_SCHEMA,
    FLEXIV_ROTATION6D_CONVENTION,
    FLEXIV_ROTATION6D_ORDER,
    FlexivSourceStateContract,
    build_flexiv_raw_force_state_schema,
    build_flexiv_state_schema,
    convert_legacy_abs_rotvec_state,
    detect_flexiv_source_state_contract,
    flexiv_action_names,
    flexiv_legacy_state_names,
    flexiv_state_names,
    project_flexiv_source_state_to_v2,
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
RAW_FORCE_STATE_FIELD_NAMES = tuple(FLEXIV_RAW_FORCE_STATE_NAMES)
ACTION_FIELD_NAMES = tuple(flexiv_action_names())
TARGET_STATE_SCHEMA = FLEXIV_STATE_SCHEMA
LEGACY_CONVERTER_NAME = FLEXIV_LEGACY_TO_V2_TRANSFORM
RAW_FORCE_CONVERTER_NAME = FLEXIV_RAW_FORCE_TO_V2_TRANSFORM
DEFAULT_OUTPUT_ROOT = Path.home() / ".cache" / "dp3_zarr"
LEROBOT_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot"
EXPORT_STATUS_ATTR = "export_status"
EXPORT_STATUS_IN_PROGRESS = "in_progress"
EXPORT_STATUS_COMPLETE = "complete"
EXPECTED_FRAMES_ATTR = "expected_total_frames"
CONVERTED_FRAMES_ATTR = "converted_frames"
INTEGRITY_ATTR = "integrity"
DEPTH_SOURCES = ("native_depth", "ffs_stereo")
FFS_BACKENDS = (
    "pytorch",
    "tensorrt_single",
    "tensorrt_two_stage",
    "tensorrt_plugin",
)
FFS_DEFAULT_PRECISION = "fp16"
FFS_DEFAULT_BUILDER_OPTIMIZATION_LEVEL = 3
FFS_DEFAULT_WORKSPACE_GIB = 8.0
FFS_DEFAULT_MAX_DISP = 416
FFS_DEFAULT_VALID_ITERS = 8
FFS_PATH_KEYS = (
    "checkpoint_path",
    "model_config_path",
    "engine_path",
    "feature_engine_path",
    "post_engine_path",
    "plugin_library_path",
    "manifest_path",
    "calibration_path",
    "config_path",
)


@dataclass(frozen=True)
class BuilderConfigResolution:
    """Runtime builder config plus the provenance needed by the exporter."""

    config_path: Path
    config: dict[str, Any]
    depth_source: str
    ffs_provenance: dict[str, Any] | None = None
    generated_config_path: bool = False


@dataclass(frozen=True)
class BuilderConfigContract:
    """The source and output contract owned by one Builder YAML."""

    path: Path
    camera: str
    pointcloud_mode: str
    num_points: int
    depth_source: str


SourceStateContract = FlexivSourceStateContract


def convert_legacy_abs_rotvec_to_v2(state: Any) -> np.ndarray:
    """Public, explicit offline converter for the supported Flexiv v1 source."""

    return convert_legacy_abs_rotvec_state(state)


def detect_source_state_contract(
    info: dict[str, Any],
    *,
    target_state_schema: str = TARGET_STATE_SCHEMA,
    allow_legacy_conversion: bool = False,
) -> SourceStateContract:
    """Compatibility wrapper for the shared source-state detector."""

    if target_state_schema != TARGET_STATE_SCHEMA:
        raise ValueError(
            f"Only target state schema {TARGET_STATE_SCHEMA!r} is supported, "
            f"got {target_state_schema!r}"
        )
    return detect_flexiv_source_state_contract(
        info,
        state_column=STATE_COLUMN,
        action_column=ACTION_COLUMN,
        allow_legacy_conversion=allow_legacy_conversion,
    )


def convert_source_state(
    state: Any,
    source_contract: SourceStateContract,
) -> np.ndarray:
    """Compatibility wrapper for the shared source-state projection helper."""

    return project_flexiv_source_state_to_v2(state, source_contract)


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
    if source_schema not in {
        TARGET_STATE_SCHEMA,
        FLEXIV_LEGACY_STATE_SCHEMA,
        FLEXIV_RAW_FORCE_STATE_SCHEMA,
    }:
        raise ValueError(f"Unknown DP3 source_state_schema: {source_schema!r}")
    expected_transform = (
        "passthrough_v2"
        if source_schema == TARGET_STATE_SCHEMA
        else LEGACY_CONVERTER_NAME
        if source_schema == FLEXIV_LEGACY_STATE_SCHEMA
        else RAW_FORCE_CONVERTER_NAME
    )
    expected_source_names = (
        STATE_FIELD_NAMES
        if source_schema == TARGET_STATE_SCHEMA
        else LEGACY_STATE_FIELD_NAMES
        if source_schema == FLEXIV_LEGACY_STATE_SCHEMA
        else RAW_FORCE_STATE_FIELD_NAMES
    )
    expected_source_dim = (
        STATE_DIM
        if source_schema == TARGET_STATE_SCHEMA
        else FLEXIV_LEGACY_STATE_DIM
        if source_schema == FLEXIV_LEGACY_STATE_SCHEMA
        else FLEXIV_RAW_FORCE_STATE_DIM
    )
    for key, expected in (
        ("state_transform", expected_transform),
        ("source_state_dim", expected_source_dim),
        ("source_state_names", list(expected_source_names)),
    ):
        if key in output and output[key] != expected:
            raise ValueError(
                f"DP3 source metadata {key}={output[key]!r} conflicts with {expected!r}"
            )
        output.setdefault(key, expected)
    if source_schema == FLEXIV_RAW_FORCE_STATE_SCHEMA:
        if "dropped_state_names" in output and tuple(output["dropped_state_names"]) != FLEXIV_RAW_FORCE_DROPPED_STATE_NAMES:
            raise ValueError("DP3 v3 source metadata dropped_state_names/order is not exact")
        output.setdefault("dropped_state_names", list(FLEXIV_RAW_FORCE_DROPPED_STATE_NAMES))
    elif output.get("dropped_state_names") not in (None, [], ()):
        raise ValueError("Only the v3 raw-force source may declare dropped_state_names")
    output.setdefault("source_fps", None)
    output.setdefault("depth_source", "native_depth")
    output.setdefault("native_depth_used_for_builder", True)
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
            if source_schema in {
                FLEXIV_LEGACY_STATE_SCHEMA,
                FLEXIV_RAW_FORCE_STATE_SCHEMA,
            }:
                raise ValueError(
                    "write_dp3_zarr requires raw_source_state_sha256 when source_state_schema "
                    "is a converted Flexiv source schema"
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
    _apply_builder_config_contract(args)
    lerobot_arg = Path(args.lerobot_path).expanduser()
    if not lerobot_arg.is_absolute():
        raise ValueError("--lerobot-path must be an absolute path")
    lerobot_path = lerobot_arg.resolve()
    if not lerobot_path.exists():
        raise FileNotFoundError(f"LeRobot dataset path does not exist: {lerobot_path}")
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
    builder_resolution = _resolve_builder_config_for_export(
        args=args,
        output_zarr=output_zarr,
        realsense_calibration=realsense_calibration,
    )
    builder_config_path = builder_resolution.config_path
    builder_config = builder_resolution.config

    PointCloudBuilder = _import_pointcloud_builder()
    try:
        builder = PointCloudBuilder.from_yaml(builder_config_path)
    except BaseException:
        if builder_resolution.depth_source == "ffs_stereo" and builder_resolution.generated_config_path:
            _remove_path(builder_config_path)
        raise

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
        depth_source=builder_resolution.depth_source,
        ffs_provenance=builder_resolution.ffs_provenance,
    )
    attrs.update(
        {
            EXPORT_STATUS_ATTR: EXPORT_STATUS_IN_PROGRESS,
            EXPECTED_FRAMES_ATTR: frames_to_export,
            "source_total_frames": total_frames,
            "raw_depth_scale_m_per_unit": sidecar_source.depth_scale_m_per_unit(args.camera),
            "pointcloud_builder_config_source": "provided",
            "native_depth_used_for_builder": builder_resolution.depth_source == "native_depth",
        }
    )
    if builder_resolution.ffs_provenance is not None:
        attrs.update(_flatten_ffs_provenance(builder_resolution.ffs_provenance))
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
        builder_meta_summary: dict[str, Any] = {
            "frames": 0,
            "last": None,
            "timing_ms_sum": {},
            "timing_ms_mean": {},
            "count_fields": {},
        }
        for source_frame in sidecar_source.iter_frames(
            data_paths,
            camera=args.camera,
            columns=columns,
            max_frames=frames_to_export,
            include_ir=builder_resolution.depth_source == "ffs_stereo",
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
            rgb = next(rgb_iter) if rgb_iter is not None else None
            frame = _builder_frame_from_source_frame(
                source_frame,
                camera=args.camera,
                depth_source=builder_resolution.depth_source,
                timestamp_column=camera_spec["timestamp_column"],
                rgb=rgb,
            )
            pc_tensor, _meta = builder.from_recorded_frame(frame)
            point_cloud = pc_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            _validate_point_cloud(point_cloud, args.num_points, pointcloud_dim, converted)
            _reject_nonfinite(point_cloud, "point_cloud", converted)
            builder_meta_summary = _record_builder_meta(
                builder_meta_summary,
                _meta,
                depth_source=builder_resolution.depth_source,
            )

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
        _update_zarr_attrs(
            work_zarr,
            {
                "pointcloud_builder_metadata": _jsonable(builder_meta_summary),
            },
        )
        _write_state_provenance(
            work_zarr,
            source_state_hash=raw_source_state_hasher.hexdigest(),
            derived_state_hash=source_hashers["state"].hexdigest(),
        )
        _commit_atomic_output(work_zarr, output_zarr, overwrite=args.overwrite)
    except BaseException:
        _remove_path(work_zarr)
        if builder_resolution.depth_source == "ffs_stereo" and builder_resolution.generated_config_path:
            _remove_path(builder_config_path)
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
        "depth_source": builder_resolution.depth_source,
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
    _validate_depth_source_metadata(root.attrs)
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


def _depth_source_mode(args: argparse.Namespace) -> str:
    value = str(getattr(args, "depth_source", "native_depth")).strip().lower()
    if value not in DEPTH_SOURCES:
        raise ValueError(f"depth_source must be one of {DEPTH_SOURCES}, got {value!r}")
    return value


def _read_builder_config_contract(path_value: str | Path | None) -> BuilderConfigContract:
    """Read all point-cloud runtime choices from the explicit Builder YAML."""

    if path_value is None:
        raise ValueError("--builder-config is required; point-cloud settings come from that YAML")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Builder config does not exist: {path}")
    raw = _read_yaml(path)

    camera = raw.get("camera")
    if not isinstance(camera, Mapping):
        raise ValueError("Builder YAML must contain a camera mapping")
    camera_name = str(camera.get("name", "")).strip().lower()
    if camera_name not in CAMERA_SPECS:
        raise ValueError(
            f"Builder YAML camera.name must be one of {sorted(CAMERA_SPECS)}, got {camera_name!r}"
        )

    pointcloud = raw.get("pointcloud")
    if not isinstance(pointcloud, Mapping):
        raise ValueError("Builder YAML must contain a pointcloud mapping")
    pointcloud_mode = str(pointcloud.get("output_format", "")).strip().lower()
    if pointcloud_mode not in {"xyz", "xyzrgb"}:
        raise ValueError(
            "Builder YAML pointcloud.output_format must be 'xyz' or 'xyzrgb', "
            f"got {pointcloud_mode!r}"
        )
    expected_use_rgb = pointcloud_mode == "xyzrgb"
    if bool(pointcloud.get("use_rgb", expected_use_rgb)) != expected_use_rgb:
        raise ValueError(
            "Builder YAML pointcloud.use_rgb must agree with pointcloud.output_format"
        )

    sampling = raw.get("sampling")
    if sampling is None:
        num_points = 1024
    elif isinstance(sampling, Mapping):
        num_points = int(sampling.get("num_points", 1024))
    else:
        raise ValueError("Builder YAML sampling must be a mapping when provided")
    if num_points <= 0:
        raise ValueError("Builder YAML sampling.num_points must be positive")

    depth_source = raw.get("depth_source")
    if depth_source is None:
        depth_mode = "native_depth"
    elif isinstance(depth_source, Mapping):
        declared_mode = str(depth_source.get("mode", "frame")).strip().lower()
        depth_mode = "native_depth" if declared_mode == "frame" else declared_mode
    else:
        raise ValueError("Builder YAML depth_source must be a mapping when provided")
    if depth_mode not in DEPTH_SOURCES:
        raise ValueError(
            f"Builder YAML depth_source.mode must be one of {DEPTH_SOURCES}, got {depth_mode!r}"
        )

    return BuilderConfigContract(
        path=path,
        camera=camera_name,
        pointcloud_mode=pointcloud_mode,
        num_points=num_points,
        depth_source=depth_mode,
    )


def _apply_builder_config_contract(args: argparse.Namespace) -> BuilderConfigContract:
    """Populate internal call fields from the only authoritative Builder YAML."""

    contract = _read_builder_config_contract(getattr(args, "builder_config", None))
    args.builder_config = str(contract.path)
    args.camera = contract.camera
    args.pointcloud_mode = contract.pointcloud_mode
    args.num_points = contract.num_points
    args.depth_source = contract.depth_source
    return contract


def _resolve_builder_config(
    *,
    args: argparse.Namespace,
    output_zarr: Path,
    realsense_calibration: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Compatibility wrapper returning only the config path and mapping."""

    resolution = _resolve_builder_config_for_export(
        args=args,
        output_zarr=output_zarr,
        realsense_calibration=realsense_calibration,
    )
    return resolution.config_path, resolution.config


def _resolve_builder_config_for_export(
    *,
    args: argparse.Namespace,
    output_zarr: Path,
    realsense_calibration: dict[str, Any],
) -> BuilderConfigResolution:
    _apply_builder_config_contract(args)
    depth_source = _depth_source_mode(args)
    path = Path(args.builder_config).expanduser().resolve()
    config = _read_yaml(path)
    if depth_source == "native_depth":
        return BuilderConfigResolution(
            config_path=path,
            config=config,
            depth_source=depth_source,
        )

    return _resolve_ffs_builder_config(
        args=args,
        output_zarr=output_zarr,
        realsense_calibration=realsense_calibration,
    )


def _resolve_ffs_builder_config(
    *,
    args: argparse.Namespace,
    output_zarr: Path,
    realsense_calibration: dict[str, Any],
) -> BuilderConfigResolution:
    original_path = Path(args.builder_config).expanduser().resolve()
    if not original_path.is_file():
        raise FileNotFoundError(f"FFS Builder config does not exist: {original_path}")
    raw = _read_yaml(original_path)
    resolved = copy.deepcopy(raw)
    depth_raw = resolved.get("depth_source")
    if not isinstance(depth_raw, dict) or str(depth_raw.get("mode", "")).lower() != "ffs_stereo":
        raise ValueError(
            "FFS Builder YAML must explicitly declare depth_source.mode=ffs_stereo"
        )
    ffs_raw = depth_raw.get("ffs")
    if not isinstance(ffs_raw, dict):
        raise ValueError("FFS Builder YAML must contain depth_source.ffs")

    backend = str(ffs_raw.get("backend", "")).strip().lower()
    if backend not in FFS_BACKENDS:
        raise ValueError(f"FFS backend must be one of {FFS_BACKENDS}, got {backend!r}")
    ffs_raw["backend"] = backend
    artifact_id = str(ffs_raw.get("artifact_id", "")).strip()
    if not artifact_id:
        raise ValueError("FFS Builder YAML must declare depth_source.ffs.artifact_id")
    ffs_raw["artifact_id"] = artifact_id
    precision = str(ffs_raw.get("precision", FFS_DEFAULT_PRECISION)).strip().lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"FFS precision must be fp16 or fp32, got {precision!r}")
    ffs_raw["precision"] = precision
    optimization_level = int(
        ffs_raw.get("builder_optimization_level", FFS_DEFAULT_BUILDER_OPTIMIZATION_LEVEL)
    )
    if not 0 <= optimization_level <= 5:
        raise ValueError("FFS builder_optimization_level must be between 0 and 5")
    ffs_raw["builder_optimization_level"] = optimization_level
    workspace_gib = float(ffs_raw.get("workspace_gib", FFS_DEFAULT_WORKSPACE_GIB))
    if not np.isfinite(workspace_gib) or workspace_gib <= 0.0:
        raise ValueError("FFS workspace_gib must be finite and positive")
    ffs_raw["workspace_gib"] = workspace_gib

    for key, default in (("width", 640), ("height", 480)):
        ffs_raw[key] = int(ffs_raw.get(key, default))
    if (int(ffs_raw["height"]), int(ffs_raw["width"])) != (480, 640):
        raise ValueError(
            "FFS Builder YAML must use the fixed IR shape height=480,width=640"
        )
    ffs_raw["max_disp"] = int(ffs_raw.get("max_disp", FFS_DEFAULT_MAX_DISP))
    ffs_raw["valid_iters"] = int(ffs_raw.get("valid_iters", FFS_DEFAULT_VALID_ITERS))
    if ffs_raw["max_disp"] <= 0 or ffs_raw["max_disp"] % 4 != 0:
        raise ValueError("FFS max_disp must be positive and divisible by 4")
    if ffs_raw["valid_iters"] <= 0:
        raise ValueError("FFS valid_iters must be positive")
    if str(ffs_raw.get("left_key", "left_ir")) != "left_ir":
        raise ValueError("FFS Builder left_key must be 'left_ir' for the exporter contract")
    if str(ffs_raw.get("right_key", "right_ir")) != "right_ir":
        raise ValueError("FFS Builder right_key must be 'right_ir' for the exporter contract")
    ffs_raw["left_key"] = "left_ir"
    ffs_raw["right_key"] = "right_ir"

    _resolve_ffs_declared_paths(ffs_raw, original_path.parent)
    _fill_ffs_artifact_defaults(ffs_raw, original_path.parent, backend, str(artifact_id))
    _resolve_ffs_declared_paths(ffs_raw, original_path.parent)
    _apply_recorded_calibration_to_ffs_config(
        resolved,
        ffs_raw,
        realsense_calibration,
        camera=args.camera,
        calibration_path=Path(
            _require_source_calibration_path(args, realsense_calibration)
        ).resolve(),
    )
    _validate_builder_pointcloud_contract(resolved, args=args)
    artifact_provenance = _preflight_ffs_artifacts(
        ffs_raw,
        config_dir=original_path.parent,
        backend=backend,
        artifact_id=str(artifact_id),
    )

    portable_config = _portable_builder_config(resolved, original_path.parent)
    portable_yaml = yaml.safe_dump(portable_config, sort_keys=False)
    runtime_yaml = yaml.safe_dump(resolved, sort_keys=False)
    resolved_path = output_zarr.with_suffix(".resolved.pointcloud_builder.yaml").resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(runtime_yaml, encoding="utf-8")
    calibration_path = Path(str(ffs_raw["calibration_path"])).resolve()
    config_provenance = {
        "source_yaml_relative_path": _portable_path(original_path, original_path.parent),
        "resolved_yaml_file_name": resolved_path.name,
        "resolved_config": portable_config,
        "resolved_config_sha256": _sha256_bytes(portable_yaml.encode("utf-8")),
        "runtime_config_sha256": _sha256_bytes(runtime_yaml.encode("utf-8")),
        "config_base_relative_to_source_yaml": ".",
    }
    ffs_provenance = {
        "depth_source": "ffs_stereo",
        "ffs_backend": backend,
        "artifact_id": str(artifact_id),
        "precision": precision,
        "max_disp": int(ffs_raw["max_disp"]),
        "valid_iters": int(ffs_raw["valid_iters"]),
        "builder_optimization_level": optimization_level,
        "workspace_gib": workspace_gib,
        "normalization_contract": _ffs_normalization_contract(backend),
        "calibration_relative_path": _portable_path(
            calibration_path,
            Path(args.lerobot_path).expanduser().resolve(),
        ),
        "calibration_sha256": _sha256_file(calibration_path),
        "rectification_mode": str(ffs_raw.get("rectification_mode", "auto")),
        "native_depth_used_for_builder": False,
        "builder_config": config_provenance,
        "artifacts": artifact_provenance,
    }
    return BuilderConfigResolution(
        config_path=resolved_path,
        config=resolved,
        depth_source="ffs_stereo",
        ffs_provenance=ffs_provenance,
        generated_config_path=True,
    )


def _resolve_ffs_declared_paths(ffs: dict[str, Any], config_dir: Path) -> None:
    for key in FFS_PATH_KEYS:
        value = ffs.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = config_dir / candidate
        ffs[key] = str(candidate.resolve())


def _fill_ffs_artifact_defaults(
    ffs: dict[str, Any],
    config_dir: Path,
    backend: str,
    artifact_id: str,
) -> None:
    candidates = [
        Path(str(ffs[key])).expanduser()
        for key in FFS_PATH_KEYS
        if isinstance(ffs.get(key), str) and ffs[key]
    ]
    artifact_dir = next((path.parent for path in candidates if path.parent.is_dir()), None)
    if artifact_dir is None:
        artifact_dir = config_dir / "artifacts"
    artifact_dir = artifact_dir.resolve()
    if backend == "pytorch":
        ffs.setdefault("checkpoint_path", str(artifact_dir / "model_best_bp2_serialize.pth"))
        ffs.setdefault("model_config_path", str(artifact_dir / "cfg.yaml"))
        ffs.setdefault("manifest_path", str(artifact_dir / f"artifact_manifest_{artifact_id}.json"))
        return
    prefix = backend
    if backend == "tensorrt_two_stage":
        ffs.setdefault("feature_engine_path", str(artifact_dir / f"{prefix}_feature_{artifact_id}.engine"))
        ffs.setdefault("post_engine_path", str(artifact_dir / f"{prefix}_post_{artifact_id}.engine"))
    else:
        ffs.setdefault("engine_path", str(artifact_dir / f"{prefix}_{artifact_id}.engine"))
    ffs.setdefault("manifest_path", str(artifact_dir / f"{prefix}_{artifact_id}.manifest.json"))
    ffs.setdefault("config_path", str(artifact_dir / f"{prefix}_{artifact_id}.yaml"))
    if backend == "tensorrt_plugin":
        ffs.setdefault("plugin_library_path", str(artifact_dir.parent / "build" / "libffs_gwc_plugin.so"))


def _require_source_calibration_path(
    args: argparse.Namespace,
    realsense_calibration: dict[str, Any],
) -> Path:
    del realsense_calibration
    dataset_root = Path(args.lerobot_path).expanduser().resolve()
    path = dataset_root / "meta" / "realsense_calibration.json"
    if not path.is_file():
        raise FileNotFoundError(f"FFS requires the recorded calibration file: {path}")
    return path


def _apply_recorded_calibration_to_ffs_config(
    config: dict[str, Any],
    ffs: dict[str, Any],
    calibration: dict[str, Any],
    *,
    camera: str,
    calibration_path: Path,
) -> None:
    camera_key = CAMERA_SPECS[camera]["calibration_key"]
    cameras = calibration.get("cameras")
    if not isinstance(cameras, dict) or not isinstance(cameras.get(camera_key), dict):
        raise KeyError(f"Calibration has no camera {camera_key!r} for FFS")
    camera_calibration = cameras[camera_key]
    streams = camera_calibration.get("streams")
    if not isinstance(streams, dict):
        raise ValueError(f"Calibration camera {camera_key!r} has no streams")
    for stream_name in ("infrared1", "infrared2"):
        stream = streams.get(stream_name)
        if not isinstance(stream, dict):
            raise KeyError(f"Calibration camera {camera_key!r} has no {stream_name} stream")
        shape = _extract_intrinsics(camera_calibration, stream_name)
        if (int(shape["height"]), int(shape["width"])) != (480, 640):
            raise ValueError(
                f"FFS calibration {camera}/{stream_name} must be 480x640, got "
                f"{shape['height']}x{shape['width']}"
            )
    baseline = camera_calibration.get("baseline")
    if not isinstance(baseline, dict):
        raise KeyError(f"Calibration camera {camera_key!r} is missing baseline")
    recorded_baseline = float(
        baseline.get("recommended_baseline_m", baseline.get("baseline_m_abs_x", 0.0))
    )
    configured_baseline = float(ffs.get("baseline_m", 0.0) or 0.0)
    if configured_baseline > 0.0 and abs(configured_baseline - recorded_baseline) > 1e-5:
        raise ValueError(
            f"FFS YAML baseline_m={configured_baseline} disagrees with recorded calibration "
            f"{recorded_baseline} for camera={camera}"
        )
    ffs["baseline_m"] = recorded_baseline
    declared_calibration = ffs.get("calibration_path")
    if declared_calibration and Path(str(declared_calibration)).resolve() != calibration_path:
        raise ValueError(
            "FFS YAML calibration_path must refer to the dataset calibration used by the sidecar"
        )
    declared_camera = str(ffs.get("calibration_camera", camera))
    if declared_camera != camera:
        raise ValueError(
            f"FFS YAML calibration_camera={declared_camera!r} does not match --camera={camera!r}"
        )
    ffs["calibration_path"] = str(calibration_path)
    ffs["calibration_camera"] = camera

    config_camera = config.setdefault("camera", {})
    if not isinstance(config_camera, dict):
        raise ValueError("FFS Builder YAML camera must be a mapping")
    config_camera["aligned_depth_to_color"] = False
    config_camera["depth_scale"] = 1.0
    config_camera["depth_intrinsics"] = _extract_intrinsics(camera_calibration, "infrared1")
    config_camera["color_intrinsics"] = _extract_intrinsics(camera_calibration, "color")
    ir_to_color = _extract_ir_to_color_extrinsics(camera_calibration)
    if ir_to_color is not None:
        config_camera["depth_to_color_extrinsics"] = ir_to_color


def _validate_builder_pointcloud_contract(
    config: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> None:
    pointcloud = config.get("pointcloud")
    if not isinstance(pointcloud, dict):
        raise ValueError("FFS Builder YAML must contain a pointcloud mapping")
    expected_mode = str(args.pointcloud_mode)
    actual_mode = str(pointcloud.get("output_format", expected_mode)).lower()
    if actual_mode != expected_mode:
        raise ValueError(
            f"FFS Builder pointcloud.output_format={actual_mode!r} conflicts with "
            f"the Builder config contract={expected_mode!r}"
        )
    expected_rgb = expected_mode == "xyzrgb"
    if bool(pointcloud.get("use_rgb", expected_rgb)) != expected_rgb:
        raise ValueError("FFS Builder pointcloud.use_rgb conflicts with its output_format")
    sampling = config.get("sampling")
    if isinstance(sampling, dict) and sampling.get("enabled", True):
        configured_points = sampling.get("num_points")
        if configured_points is not None and int(configured_points) != int(args.num_points):
            raise ValueError(
                f"FFS Builder sampling.num_points={configured_points} conflicts with "
                f"the Builder config contract={args.num_points}"
            )


def _preflight_ffs_artifacts(
    ffs: dict[str, Any],
    *,
    config_dir: Path,
    backend: str,
    artifact_id: str,
) -> dict[str, Any]:
    del config_dir
    required_keys = {
        "pytorch": ("checkpoint_path", "model_config_path", "manifest_path"),
        "tensorrt_single": ("engine_path", "manifest_path", "config_path"),
        "tensorrt_two_stage": (
            "feature_engine_path",
            "post_engine_path",
            "manifest_path",
            "config_path",
        ),
        "tensorrt_plugin": (
            "engine_path",
            "plugin_library_path",
            "manifest_path",
            "config_path",
        ),
    }[backend]
    paths: dict[str, Path] = {}
    for key in required_keys:
        value = ffs.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"FFS backend {backend} requires depth_source.ffs.{key}")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"FFS {key} does not exist: {path}")
        paths[key] = path
    manifest_path = paths["manifest_path"]
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"FFS manifest must be a mapping: {manifest_path}")
    route = manifest
    if isinstance(manifest.get("routes"), dict):
        route_value = manifest["routes"].get(backend)
        if isinstance(route_value, dict):
            route = route_value
        elif backend != "pytorch":
            raise ValueError(f"FFS aggregate manifest has no route for backend={backend!r}")
    _validate_ffs_manifest_contract(
        route,
        backend=backend,
        artifact_id=artifact_id,
        ffs=ffs,
    )
    manifest_artifacts = _manifest_file_records(manifest, route, manifest_path.parent)
    for record in manifest_artifacts:
        artifact_path = record["path"]
        if not artifact_path.is_file():
            raise FileNotFoundError(f"FFS manifest artifact is missing: {artifact_path}")
        actual_sha = _sha256_file(artifact_path)
        if actual_sha != record["sha256"]:
            raise ValueError(
                f"FFS artifact SHA-256 mismatch for {artifact_path}: "
                f"actual={actual_sha}, manifest={record['sha256']}"
            )
    hashes: dict[str, str] = {}
    for key, path in paths.items():
        digest = _manifest_hash_for_path(path, manifest_artifacts)
        if key == "config_path":
            _validate_ffs_engine_config(path, ffs=ffs, backend=backend, artifact_id=artifact_id)
            digest = _sha256_file(path)
        if digest is None:
            digest = _sha256_file(path)
        hashes[key] = digest
    hashes["manifest_path"] = _sha256_file(manifest_path)
    config_path = paths.get("config_path")
    if config_path is not None:
        hashes["config_path"] = _sha256_file(config_path)
    return {
        "manifest_relative_path": manifest_path.name,
        "manifest_file_name": manifest_path.name,
        "manifest_sha256": hashes["manifest_path"],
        "contract": {
            "backend": backend,
            "artifact_id": artifact_id,
            "height": int(ffs["height"]),
            "width": int(ffs["width"]),
            "max_disp": int(ffs["max_disp"]),
            "valid_iters": int(ffs["valid_iters"]),
            "precision": str(ffs["precision"]),
            "normalization_contract": _ffs_normalization_contract(backend),
            "builder_optimization_level": int(ffs["builder_optimization_level"]),
            "workspace_gib": float(ffs["workspace_gib"]),
        },
        "files": {
            key: {
                "file_name": path.name,
                "relative_path": _portable_path(path, manifest_path.parent),
                "sha256": digest,
            }
            for key, (path, digest) in (
                (key, (paths[key], hashes[key])) for key in paths
            )
        },
        "manifest": _portable_value(manifest),
    }


def _validate_ffs_manifest_contract(
    manifest: dict[str, Any],
    *,
    backend: str,
    artifact_id: str,
    ffs: dict[str, Any],
) -> None:
    expected = {
        "backend": backend,
        "artifact_id": artifact_id,
        "height": int(ffs["height"]),
        "width": int(ffs["width"]),
        "max_disp": int(ffs["max_disp"]),
        "valid_iters": int(ffs["valid_iters"]),
        "precision": str(ffs["precision"]),
        "builder_optimization_level": int(ffs["builder_optimization_level"]),
        "workspace_gib": float(ffs["workspace_gib"]),
    }
    expected["normalization_contract"] = _ffs_normalization_contract(backend)
    for key, value in expected.items():
        actual = manifest.get(key)
        if actual is None and backend == "pytorch" and key in {"backend", "normalization_contract"}:
            continue
        if actual != value:
            raise ValueError(
                f"FFS manifest mismatch for {key}: manifest={actual!r}, configured={value!r}"
            )


def _validate_ffs_engine_config(
    path: Path,
    *,
    ffs: dict[str, Any],
    backend: str,
    artifact_id: str,
) -> None:
    config = _read_yaml(path)
    expected = {
        "backend": backend,
        "artifact_id": artifact_id,
        "height": int(ffs["height"]),
        "width": int(ffs["width"]),
        "max_disp": int(ffs["max_disp"]),
        "valid_iters": int(ffs["valid_iters"]),
        "precision": str(ffs["precision"]),
        "builder_optimization_level": int(ffs["builder_optimization_level"]),
        "workspace_gib": float(ffs["workspace_gib"]),
        "normalization_contract": _ffs_normalization_contract(backend),
    }
    for key, value in expected.items():
        if key in config and config[key] != value:
            raise ValueError(
                f"FFS engine/config mismatch for {key}: config={config[key]!r}, configured={value!r}"
            )


def _manifest_file_records(
    manifest: dict[str, Any],
    route: dict[str, Any],
    manifest_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    raw_records = route.get("artifacts")
    if isinstance(raw_records, list):
        for item in raw_records:
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                raise ValueError("FFS manifest artifacts must contain path and sha256")
            records.append(
                {
                    "path": (manifest_dir / str(item["path"])).resolve(),
                    "sha256": str(item["sha256"]),
                }
            )
    for record_name in ("checkpoint", "model_config"):
        item = route.get(record_name) or manifest.get(record_name)
        if isinstance(item, dict) and item.get("path") and item.get("sha256"):
            records.append(
                {
                    "path": (manifest_dir / str(item["path"])).resolve(),
                    "sha256": str(item["sha256"]),
                }
            )
    unique: dict[Path, dict[str, Any]] = {}
    for record in records:
        previous = unique.get(record["path"])
        if previous is not None and previous["sha256"] != record["sha256"]:
            raise ValueError(f"FFS manifest has conflicting hashes for {record['path']}")
        unique[record["path"]] = record
    return list(unique.values())


def _manifest_hash_for_path(path: Path, records: list[dict[str, Any]]) -> str | None:
    resolved = path.expanduser().resolve()
    for record in records:
        if record["path"] == resolved or record["path"].name == resolved.name:
            return str(record["sha256"])
    return None


def _ffs_normalization_contract(backend: str) -> str:
    return (
        "external_imagenet_0_255"
        if backend == "tensorrt_single"
        else "internal_imagenet_0_255"
    )


def _portable_builder_config(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    output = copy.deepcopy(config)
    depth_source = output.get("depth_source")
    if isinstance(depth_source, dict) and isinstance(depth_source.get("ffs"), dict):
        ffs = depth_source["ffs"]
        for key in FFS_PATH_KEYS:
            value = ffs.get(key)
            if isinstance(value, str) and value:
                ffs[key] = _portable_path(Path(value), base_dir)
    return _portable_value(output)


def _portable_path(path: Path, base_dir: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(base_dir.expanduser().resolve()).as_posix()
    except ValueError:
        return resolved.name


def _portable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _portable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_value(item) for item in value]
    if isinstance(value, Path):
        return value.name
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        RAW_FORCE_CONVERTER_NAME,
    }:
        raise ValueError(
            "Zarr state_transform must be passthrough_v2, "
            f"{LEGACY_CONVERTER_NAME}, or {RAW_FORCE_CONVERTER_NAME}"
        )
    source_schema = attrs.get("source_state_schema")
    if source_schema not in {
        TARGET_STATE_SCHEMA,
        FLEXIV_LEGACY_STATE_SCHEMA,
        FLEXIV_RAW_FORCE_STATE_SCHEMA,
    }:
        raise ValueError(f"Unknown Zarr source_state_schema: {source_schema!r}")
    expected_transform = (
        "passthrough_v2"
        if source_schema == TARGET_STATE_SCHEMA
        else LEGACY_CONVERTER_NAME
        if source_schema == FLEXIV_LEGACY_STATE_SCHEMA
        else RAW_FORCE_CONVERTER_NAME
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
        if source_schema == FLEXIV_LEGACY_STATE_SCHEMA
        else RAW_FORCE_STATE_FIELD_NAMES
    )
    if tuple(source_names or ()) != expected_source_names:
        raise ValueError("Zarr source_state_names/order does not match source_state_schema")
    expected_source_dim = (
        STATE_DIM
        if source_schema == TARGET_STATE_SCHEMA
        else FLEXIV_LEGACY_STATE_DIM
        if source_schema == FLEXIV_LEGACY_STATE_SCHEMA
        else FLEXIV_RAW_FORCE_STATE_DIM
    )
    if attrs.get("source_state_dim") != expected_source_dim:
        raise ValueError(
            "Zarr source_state_dim does not match source_state_schema: "
            f"expected={expected_source_dim}, got={attrs.get('source_state_dim')!r}"
        )
    if source_schema == FLEXIV_RAW_FORCE_STATE_SCHEMA:
        if tuple(attrs.get("dropped_state_names") or ()) != FLEXIV_RAW_FORCE_DROPPED_STATE_NAMES:
            raise ValueError("Zarr v3 dropped_state_names/order does not match the source contract")
    elif attrs.get("dropped_state_names") not in (None, [], ()):
        raise ValueError("Only a v3 raw-force source may declare dropped_state_names")
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


def _validate_depth_source_metadata(attrs: Any) -> None:
    depth_source = attrs.get("depth_source", "native_depth")
    if depth_source == "native_depth":
        if attrs.get("native_depth_used_for_builder", True) is not True:
            raise ValueError("native_depth export must declare native_depth_used_for_builder=true")
        return
    if depth_source != "ffs_stereo":
        raise ValueError(f"Unsupported Zarr depth_source: {depth_source!r}")
    if attrs.get("native_depth_used_for_builder") is not False:
        raise ValueError("FFS export must declare native_depth_used_for_builder=false")
    required = {
        "ffs_backend": FFS_BACKENDS,
        "artifact_id": None,
        "precision": {"fp16", "fp32"},
        "max_disp": None,
        "valid_iters": None,
        "builder_optimization_level": None,
        "workspace_gib": None,
        "normalization_contract": None,
        "calibration_sha256": None,
        "ffs_manifest_sha256": None,
    }
    for key, allowed in required.items():
        value = attrs.get(key)
        if value is None or value == "":
            raise ValueError(f"FFS Zarr provenance is missing {key}")
        if allowed is not None and value not in allowed:
            raise ValueError(f"FFS Zarr provenance {key}={value!r} is invalid")
    for key in ("calibration_sha256", "ffs_manifest_sha256"):
        value = str(attrs[key])
        if len(value) != 64:
            raise ValueError(f"FFS Zarr provenance {key} must be a SHA-256 digest")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"FFS Zarr provenance {key} must be hexadecimal") from exc
    for key in ("max_disp", "valid_iters"):
        if int(attrs[key]) <= 0:
            raise ValueError(f"FFS Zarr provenance {key} must be positive")
    if not 0 <= int(attrs["builder_optimization_level"]) <= 5:
        raise ValueError("FFS Zarr provenance builder_optimization_level must be 0..5")
    if not (0.0 < float(attrs["workspace_gib"]) and np.isfinite(float(attrs["workspace_gib"]))):
        raise ValueError("FFS Zarr provenance workspace_gib must be finite and positive")
    resolved = attrs.get("ffs_builder_resolved_config")
    if not isinstance(resolved, dict):
        raise ValueError("FFS Zarr provenance is missing resolved Builder config")
    config_hash = resolved.get("resolved_config_sha256")
    if not isinstance(config_hash, str) or len(config_hash) != 64:
        raise ValueError("FFS Zarr provenance is missing resolved Builder config hash")
    artifacts = attrs.get("ffs_artifact_provenance")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("files"), dict):
        raise ValueError("FFS Zarr provenance is missing artifact file hashes")


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
    depth_source: str = "native_depth",
    ffs_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attrs = {
        "source_lerobot_path": str(lerobot_path),
        "camera": args.camera,
        "pointcloud_mode": args.pointcloud_mode,
        "num_points": int(args.num_points),
        "pointcloud_builder_config": builder_config,
        "pointcloud_builder_config_path": str(builder_config_path),
        "realsense_calibration": realsense_calibration,
        "depth_source": depth_source,
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
        "source_state_dim": int(source_contract.state_dim),
        "source_state_names": list(source_contract.state_names),
        "state_transform": source_contract.transform,
        "source_fps": source_fps,
        "native_depth_used_for_builder": depth_source == "native_depth",
    }
    if source_contract.dropped_state_names:
        attrs["dropped_state_names"] = list(source_contract.dropped_state_names)
    attrs.update(sidecar_source.provenance)
    if ffs_provenance is not None:
        attrs.update(_flatten_ffs_provenance(ffs_provenance))
    return attrs


def _flatten_ffs_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Expose stable top-level FFS fields while retaining the nested record."""

    result = {
        "depth_source": "ffs_stereo",
        "ffs_backend": provenance.get("ffs_backend"),
        "artifact_id": provenance.get("artifact_id"),
        "ffs_artifact_id": provenance.get("artifact_id"),
        "precision": provenance.get("precision"),
        "ffs_precision": provenance.get("precision"),
        "max_disp": provenance.get("max_disp"),
        "valid_iters": provenance.get("valid_iters"),
        "builder_optimization_level": provenance.get("builder_optimization_level"),
        "workspace_gib": provenance.get("workspace_gib"),
        "normalization_contract": provenance.get("normalization_contract"),
        "calibration_sha256": provenance.get("calibration_sha256"),
        "ffs_rectification_mode": provenance.get("rectification_mode"),
        "native_depth_used_for_builder": False,
        "ffs_builder_resolved_config": provenance.get("builder_config"),
        "ffs_builder_resolved_config_sha256": (
            provenance.get("builder_config", {}).get("resolved_config_sha256")
            if isinstance(provenance.get("builder_config"), dict)
            else None
        ),
        "ffs_builder_runtime_config_sha256": (
            provenance.get("builder_config", {}).get("runtime_config_sha256")
            if isinstance(provenance.get("builder_config"), dict)
            else None
        ),
        "ffs_artifact_provenance": provenance.get("artifacts"),
    }
    artifacts = provenance.get("artifacts")
    if isinstance(artifacts, dict):
        result["ffs_manifest_sha256"] = artifacts.get("manifest_sha256")
        result["ffs_manifest_relative_path"] = artifacts.get("manifest_relative_path")
        files = artifacts.get("files")
        if isinstance(files, dict):
            for key, record in files.items():
                if not isinstance(record, dict):
                    continue
                result[f"ffs_{key}_sha256"] = record.get("sha256")
    return result


def _builder_frame_from_source_frame(
    source_frame: rgbd_source.RGBDSourceFrame,
    *,
    camera: str,
    depth_source: str,
    timestamp_column: str,
    rgb: np.ndarray | None,
) -> dict[str, Any]:
    row = source_frame.row
    context = _source_frame_context(source_frame, camera)
    if "global_frame_index" not in row or timestamp_column not in row:
        raise KeyError(f"{context}: missing timestamp/global_frame_index join fields")
    timestamp = float(row[timestamp_column])
    global_frame_index = int(row["global_frame_index"])
    if not np.isfinite(timestamp):
        raise ValueError(f"{context}: timestamp is not finite")
    if depth_source == "native_depth":
        frame: dict[str, Any] = {
            "depth": _as_depth(source_frame.depth, "native depth", source_frame.source_path),
            "timestamp": timestamp,
            "global_frame_index": global_frame_index,
        }
    else:
        pair = source_frame.ir_pair
        if pair is None:
            raise ValueError(f"{context}: FFS requires source_frame.ir_pair")
        _validate_ffs_ir_pair(
            source_frame,
            camera=camera,
            timestamp_column=timestamp_column,
        )
        frame = {
            "left_ir": pair.left_ir,
            "right_ir": pair.right_ir,
            "timestamp": timestamp,
            "global_frame_index": global_frame_index,
        }
    if rgb is not None:
        frame["rgb"] = rgb
    return frame


def _validate_ffs_ir_pair(
    source_frame: rgbd_source.RGBDSourceFrame,
    *,
    camera: str,
    timestamp_column: str,
) -> None:
    row = source_frame.row
    context = _source_frame_context(source_frame, camera)
    pair = source_frame.ir_pair
    if pair is None:
        raise ValueError(f"{context}: FFS requires source_frame.ir_pair")
    if not isinstance(pair.calibration_sha256, str) or len(pair.calibration_sha256) != 64:
        raise ValueError(f"{context}: IR calibration SHA-256 is missing or invalid")
    try:
        int(pair.calibration_sha256, 16)
    except ValueError as exc:
        raise ValueError(f"{context}: IR calibration SHA-256 is not hexadecimal") from exc
    if pair.calibration_path is None or not Path(pair.calibration_path).is_file():
        raise ValueError(f"{context}: IR calibration path is missing or does not exist")
    left = np.asarray(pair.left_ir)
    right = np.asarray(pair.right_ir)
    for name, array in (("left_ir", left), ("right_ir", right)):
        if array.dtype != np.dtype("uint8"):
            try:
                numeric = np.asarray(array)
                if not np.issubdtype(numeric.dtype, np.number):
                    raise TypeError
                if not np.isfinite(numeric).all() or np.any(numeric < 0) or np.any(numeric > 255):
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError(f"{context}: {name} must be uint8 or finite values in [0,255]")
    if left.shape != (480, 640) or right.shape != (480, 640):
        raise ValueError(
            f"{context}: FFS IR shape must be (480,640), got left={left.shape}, right={right.shape}"
        )
    if left.shape != right.shape:
        raise ValueError(f"{context}: left_ir/right_ir shape mismatch: {left.shape} != {right.shape}")
    row_timestamp = float(row[timestamp_column])
    if not np.isfinite(float(pair.timestamp)) or not np.isfinite(row_timestamp):
        raise ValueError(f"{context}: IR timestamp is not finite")
    if not np.isclose(float(pair.timestamp), row_timestamp, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"{context}: IR timestamp {pair.timestamp!r} != row timestamp {row_timestamp!r}"
        )
    try:
        int(row["global_frame_index"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: global_frame_index is not an integer") from exc


def _source_frame_context(source_frame: rgbd_source.RGBDSourceFrame, camera: str) -> str:
    row = source_frame.row
    return (
        f"camera={camera}, row={source_frame.row_index}, "
        f"episode={row.get('episode_index', '?')}, frame={row.get('frame_index', '?')}, "
        f"global={row.get('global_frame_index', '?')}"
    )


def _record_builder_meta(
    summary: dict[str, Any],
    meta: Any,
    *,
    depth_source: str,
) -> dict[str, Any]:
    if not isinstance(meta, Mapping):
        raise ValueError(f"PointCloudBuilder metadata must be a mapping, got {type(meta)!r}")
    summary = copy.deepcopy(summary)
    summary["frames"] = int(summary.get("frames", 0)) + 1
    selected = {
        key: meta[key]
        for key in (
            "stage",
            "mode",
            "depth_source",
            "effective_depth_scale",
            "num_raw_points",
            "num_cropped_points",
            "num_sampled_points",
            "crop_enabled",
            "crop_empty",
            "sampling_enabled",
            "sampling_mode",
            "target_num_points",
            "padded",
            "pad_mode",
            "device",
            "timestamp",
            "global_frame_index",
            "camera_name",
            "intrinsics",
            "ffs",
        )
        if key in meta
    }
    selected.setdefault("depth_source", depth_source)
    summary["last"] = _portable_value(selected)
    count_fields = summary.setdefault("count_fields", {})
    for key in ("num_raw_points", "num_cropped_points", "num_sampled_points", "target_num_points"):
        value = meta.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            record = count_fields.setdefault(key, {"min": value, "max": value, "last": value})
            record["min"] = min(record["min"], value)
            record["max"] = max(record["max"], value)
            record["last"] = value
    ffs_meta = meta.get("ffs")
    timing = ffs_meta.get("timing_ms") if isinstance(ffs_meta, Mapping) else None
    if isinstance(timing, Mapping):
        timing_sum = summary.setdefault("timing_ms_sum", {})
        for key, value in timing.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                timing_sum[key] = float(timing_sum.get(key, 0.0)) + float(value)
        summary["timing_ms_mean"] = {
            key: value / summary["frames"] for key, value in timing_sum.items()
        }
    return summary


def _update_zarr_attrs(path: Path, attrs: dict[str, Any]) -> None:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("zarr is required to update DP3 replay-buffer provenance") from exc
    root = zarr.open(str(path), mode="a")
    root.attrs.update(_jsonable(attrs))


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


def _extract_ir_to_color_extrinsics(camera_calibration: dict[str, Any]) -> dict[str, Any] | None:
    extrinsics = camera_calibration.get("extrinsics", {}).get("infrared1_to_color")
    if not isinstance(extrinsics, dict):
        return None
    rotation = extrinsics.get("rotation_matrix_row_major")
    if rotation is None:
        rotation = extrinsics.get("rotation")
    translation = extrinsics.get("translation_m", extrinsics.get("translation"))
    if rotation is None or translation is None:
        raise KeyError("Incomplete infrared1_to_color extrinsics")
    if len(rotation) == 9:
        rotation = [rotation[0:3], rotation[3:6], rotation[6:9]]
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
    camera = getattr(args, "camera", "head")
    video_key = CAMERA_SPECS.get(camera, CAMERA_SPECS["head"])["video_key"]
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
            f"{DEFAULT_OUTPUT_ROOT}/<lerobot_repo_id>_<builder-config-contract>"
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
    parser.add_argument(
        "--rgbd-sidecar-source",
        choices=["auto", "zarr", "parquet"],
        default="auto",
        help=(
            "RGB-D/IR source layout. auto requires and validates raw Zarr when "
            "meta/rgbd_sidecar.json exists; otherwise it uses legacy Parquet."
        ),
    )
    parser.add_argument(
        "--builder-config",
        required=True,
        help="PointCloudBuilder YAML; camera, point-cloud, sampling, and depth settings come from it.",
    )
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
