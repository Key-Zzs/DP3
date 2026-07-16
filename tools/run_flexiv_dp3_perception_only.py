#!/usr/bin/env python3
"""Validate live Flexiv DP3 RGB-D and point clouds without connecting robots."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
import site
import sys
import time
import types
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

USER_SITE = site.getusersitepackages()
sys.path = [path for path in sys.path if Path(path).resolve() != Path(USER_SITE).resolve()]

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
DP3_ROOT = REPO_ROOT / "3D-Diffusion-Policy"
POINTCLOUD_BUILDER_ROOT = REPO_ROOT / "PointCloudBuilder"
TOOLS_DIR = Path(__file__).resolve().parent
FLEXIV_INTERFACE_DIR = (
    REPO_ROOT / "third_party" / "real" / "dual_flexiv_rizon4s" / "interface"
)
DEFAULT_CONFIG = DP3_ROOT / "diffusion_policy_3d/config/dp3_inference_config.yaml"

for path in (REPO_ROOT, DP3_ROOT, POINTCLOUD_BUILDER_ROOT):
    if path.exists():
        sys.path.insert(0, str(path))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import export_lerobot_to_dp3_zarr as exporter  # noqa: E402
from pointcloud_builder import PointCloudBuilder  # noqa: E402
from pointcloud_builder.config import load_config as load_pointcloud_config  # noqa: E402
from tools.flexiv_dp3_live_viewer import (  # noqa: E402
    LiveVisualizationPublisher,
    ViewerConfig,
)

LOGGER = logging.getLogger("flexiv_dp3_perception_only")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture head-camera RGB-D, build raw/cropped/sampled DP3 point clouds, "
            "and report quality without importing Flexiv RDK or connecting either arm."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frames", type=int, default=300, help="Measured frames after warmup.")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=None,
        help="Discarded camera frames; defaults to robot.camera_warmup_frames.",
    )
    parser.add_argument("--stability-window", type=int, default=None)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=None)
    parser.add_argument("--max-valid-depth-ratio-range", type=float, default=None)
    parser.add_argument("--camera-timeout-ms", type=int, default=None)
    parser.add_argument("--pointcloud-device", default=None)
    parser.add_argument(
        "--builder-config",
        type=Path,
        default=None,
        help="Override pointcloud.config with an explicit PointCloudBuilder YAML.",
    )
    parser.add_argument(
        "--depth-source",
        choices=exporter.DEPTH_SOURCES,
        default="native_depth",
        help="Depth input for PointCloudBuilder (default: native_depth).",
    )
    parser.add_argument("--ffs-backend", choices=exporter.FFS_BACKENDS)
    parser.add_argument("--ffs-artifact-id")
    parser.add_argument("--ffs-precision", choices=("fp16", "fp32"))
    parser.add_argument(
        "--ffs-builder-optimization-level",
        type=int,
        choices=range(6),
        metavar="0..5",
    )
    parser.add_argument("--ffs-workspace-gib", type=float)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show the existing non-blocking depth/raw/cropped/sampled viewer.",
    )
    parser.add_argument("--viewer-rate-hz", type=float, default=None)
    parser.add_argument(
        "--fail-on-quality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return exit code 2 when the final quality gate fails.",
    )
    args = parser.parse_args(argv)
    _validate_cli_args(args)
    return args


def _validate_cli_args(args: argparse.Namespace) -> None:
    for name in ("frames", "print_every"):
        value = getattr(args, name)
        if isinstance(value, bool) or int(value) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be a positive integer")
    for name in ("warmup_frames", "stability_window", "camera_timeout_ms"):
        value = getattr(args, name)
        if value is not None and (isinstance(value, bool) or int(value) <= 0):
            raise SystemExit(f"--{name.replace('_', '-')} must be a positive integer")
    for name in ("min_valid_depth_ratio", "max_valid_depth_ratio_range"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.viewer_rate_hz is not None and float(args.viewer_rate_hz) <= 0.0:
        raise SystemExit("--viewer-rate-hz must be positive")
    if args.ffs_workspace_gib is not None and args.ffs_workspace_gib <= 0:
        raise SystemExit("--ffs-workspace-gib must be positive")
    if args.depth_source == "native_depth" and any(
        value is not None
        for value in (
            args.ffs_backend,
            args.ffs_artifact_id,
            args.ffs_precision,
            args.ffs_builder_optimization_level,
            args.ffs_workspace_gib,
        )
    ):
        raise SystemExit("FFS options require --depth-source=ffs_stereo")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runtime = _load_runtime(args)
    camera = _make_camera(runtime)
    builder = _make_builder(runtime)
    _validate_camera_builder_contract(runtime, builder)
    viewer = _make_viewer(runtime, builder)

    runtime.log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = runtime.log_dir / f"flexiv_dp3_perception_only_{timestamp}.jsonl"
    summary_path = runtime.log_dir / f"flexiv_dp3_perception_only_{timestamp}_summary.json"

    print("[perception-only] robot connection: DISABLED")
    print("[perception-only] robot commands: DISABLED")
    print(f"[perception-only] camera serial: {runtime.camera_serial}")
    print(f"[perception-only] config: {runtime.config_path}")
    print(f"[perception-only] pointcloud config: {runtime.pointcloud_config_path}")
    print(f"[perception-only] depth source: {runtime.depth_source}")
    print(f"[perception-only] log: {log_path}")

    records: list[dict[str, Any]] = []
    interrupted = False
    try:
        camera.connect(warmup=False)
        _discard_warmup_frames(camera, runtime)
        if viewer is not None:
            viewer.start()
        with log_path.open("w", encoding="utf-8") as log_file:
            for index in range(runtime.frames):
                capture_started = time.perf_counter()
                frame = camera.read_rgbd_ir(timeout_ms=runtime.camera_timeout_ms)
                capture_ms = (time.perf_counter() - capture_started) * 1000.0

                build_started = time.perf_counter()
                sampled, meta, stages = builder.from_live_frame_with_stages(frame)
                if builder.device.type == "cuda":
                    torch.cuda.synchronize(builder.device)
                build_ms = (time.perf_counter() - build_started) * 1000.0

                record = _frame_record(
                    index=index,
                    frame=frame,
                    sampled=sampled,
                    stages=stages,
                    meta=meta,
                    builder=builder,
                    capture_ms=capture_ms,
                    build_ms=build_ms,
                )
                records.append(record)
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()

                if index < 5 or (index + 1) % runtime.print_every == 0:
                    _print_record(record)
                if viewer is not None:
                    viewer.maybe_publish(
                        step_idx=index,
                        depth=frame["depth"],
                        stages=stages,
                        sampled_point_cloud=_to_numpy(sampled),
                        pointcloud_meta=meta,
                    )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[perception-only] interrupted by user")
    finally:
        if viewer is not None:
            viewer.close()
        if getattr(camera, "is_connected", False):
            camera.disconnect()

    summary = _summarize_records(records, runtime, interrupted=interrupted)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _print_summary(summary, summary_path)
    if not records:
        return 1
    if runtime.fail_on_quality and not summary["quality_gate"]["passed"]:
        return 2
    return 0


class Runtime:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _load_runtime(args: argparse.Namespace) -> Runtime:
    config_path = args.config.expanduser().resolve()
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, Mapping):
        raise SystemExit(f"Inference config must be a mapping: {config_path}")

    robot_section = _mapping(config, "robot")
    inference_section = _mapping(config, "inference")
    pointcloud_section = _mapping(config, "pointcloud")
    visualization_section = _mapping(config, "visualization", required=False)

    robot_config_path = _resolve_path(robot_section["config"])
    robot_config = _load_yaml(robot_config_path)
    robot_raw = _mapping(robot_config, "robot")
    cameras_raw = _mapping(robot_config, "cameras")

    camera_name = str(robot_section.get("camera_name", "head_rgb"))
    if camera_name != "head_rgb":
        raise SystemExit(
            f"perception-only currently requires robot.camera_name=head_rgb, got {camera_name!r}"
        )
    camera_serial = str(cameras_raw.get("head_cam_serial", "")).strip()
    if not camera_serial:
        raise SystemExit(f"Missing cameras.head_cam_serial in {robot_config_path}")

    warmup_frames = args.warmup_frames or int(robot_raw.get("camera_warmup_frames", 60))
    stability_window = args.stability_window or int(
        robot_raw.get("camera_warmup_stability_window", 15)
    )
    if stability_window > args.frames:
        raise SystemExit("--stability-window must not exceed --frames")
    minimum_ratio = (
        float(args.min_valid_depth_ratio)
        if args.min_valid_depth_ratio is not None
        else float(robot_raw.get("camera_min_valid_depth_ratio", 0.75))
    )
    maximum_range = (
        float(args.max_valid_depth_ratio_range)
        if args.max_valid_depth_ratio_range is not None
        else float(robot_raw.get("camera_max_valid_depth_ratio_range", 0.08))
    )
    camera_timeout_ms = args.camera_timeout_ms or int(
        robot_raw.get("camera_read_timeout_ms", 2000)
    )
    visualize = (
        bool(args.visualize)
        if args.visualize is not None
        else bool(visualization_section.get("enabled", True))
    )
    viewer_rate_hz = (
        float(args.viewer_rate_hz)
        if args.viewer_rate_hz is not None
        else float(visualization_section.get("rate_hz", 2.0))
    )
    pointcloud_config_path = (
        args.builder_config.expanduser().resolve()
        if args.builder_config is not None
        else _resolve_path(pointcloud_section["config"])
    )
    depth_source = _resolve_live_depth_source(args, pointcloud_config_path)
    return Runtime(
        config_path=config_path,
        robot_config_path=robot_config_path,
        pointcloud_config_path=pointcloud_config_path,
        depth_source=depth_source,
        pointcloud_device=args.pointcloud_device or pointcloud_section.get("device", "auto"),
        camera_serial=camera_serial,
        camera_fps=int(robot_section.get("camera_fps", 30)),
        camera_width=int(cameras_raw.get("width", 640)),
        camera_height=int(cameras_raw.get("height", 480)),
        camera_timeout_ms=camera_timeout_ms,
        warmup_frames=warmup_frames,
        stability_window=stability_window,
        min_valid_depth_ratio=minimum_ratio,
        max_valid_depth_ratio_range=maximum_range,
        frames=int(args.frames),
        print_every=int(args.print_every),
        log_dir=args.log_dir.expanduser().resolve(),
        visualize=visualize,
        viewer_rate_hz=viewer_rate_hz,
        visualization_max_raw_points=int(visualization_section.get("max_raw_points", 30000)),
        visualization_max_cropped_points=int(
            visualization_section.get("max_cropped_points", 30000)
        ),
        visualization_point_size=float(visualization_section.get("point_size", 3.0)),
        fail_on_quality=bool(args.fail_on_quality),
        gpu_id=int(inference_section.get("gpu_id", 0)),
    )


def _make_camera(runtime: Runtime) -> Any:
    camera_mod = _camera_module()
    return camera_mod.RealSenseCamera(
        camera_mod.RealSenseCameraConfig(
            serial_number_or_name=runtime.camera_serial,
            fps=runtime.camera_fps,
            width=runtime.camera_width,
            height=runtime.camera_height,
            color_mode=camera_mod.ColorMode.RGB,
            use_depth=True,
            use_ir=getattr(runtime, "depth_source", "native_depth") == "ffs_stereo",
            rotation=camera_mod.Cv2Rotation.NO_ROTATION,
        )
    )


def _camera_module() -> Any:
    package_name = "_dp3_perception_only_flexiv_interface"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(FLEXIV_INTERFACE_DIR)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.realsense_camera")


def _make_builder(runtime: Runtime) -> PointCloudBuilder:
    config = load_pointcloud_config(runtime.pointcloud_config_path)
    config = replace(config, device=str(runtime.pointcloud_device))
    return PointCloudBuilder(config)


def _resolve_live_depth_source(args: argparse.Namespace, builder_path: Path) -> str:
    """Validate the live Builder YAML without changing the file on disk."""

    raw = exporter._read_yaml(builder_path)
    declared = raw.get("depth_source")
    if isinstance(declared, Mapping):
        declared_mode = str(declared.get("mode", "")).strip().lower()
    elif isinstance(declared, str):
        declared_mode = declared.strip().lower()
    else:
        declared_mode = ""
    if declared_mode == "frame":
        declared_mode = "native_depth"
    if declared_mode and declared_mode not in exporter.DEPTH_SOURCES:
        raise SystemExit(f"Unsupported Builder depth_source.mode={declared_mode!r}")
    if declared_mode and declared_mode != args.depth_source:
        raise SystemExit(
            f"--depth-source={args.depth_source!r} conflicts with Builder YAML "
            f"depth_source.mode={declared_mode!r}"
        )
    if args.depth_source == "native_depth":
        return "native_depth"
    if args.builder_config is None:
        raise SystemExit("--depth-source=ffs_stereo requires --builder-config")
    if not isinstance(declared, Mapping) or not isinstance(declared.get("ffs"), Mapping):
        raise SystemExit("FFS Builder YAML must contain depth_source.ffs")

    ffs = copy.deepcopy(dict(declared["ffs"]))
    backend = str(ffs.get("backend", "")).strip().lower()
    if backend not in exporter.FFS_BACKENDS:
        raise SystemExit(f"FFS Builder YAML backend must be one of {exporter.FFS_BACKENDS}")
    artifact_id = str(ffs.get("artifact_id", "")).strip()
    if not artifact_id:
        raise SystemExit("FFS Builder YAML must declare depth_source.ffs.artifact_id")
    _check_live_ffs_cli_contract(args, ffs, "backend", backend)
    _check_live_ffs_cli_contract(args, ffs, "artifact_id", artifact_id)
    precision = str(ffs.get("precision", exporter.FFS_DEFAULT_PRECISION)).strip().lower()
    _check_live_ffs_cli_contract(args, ffs, "precision", precision)
    optimization_level = int(
        ffs.get("builder_optimization_level", exporter.FFS_DEFAULT_BUILDER_OPTIMIZATION_LEVEL)
    )
    _check_live_ffs_cli_contract(
        args,
        ffs,
        "builder_optimization_level",
        optimization_level,
    )
    workspace_gib = float(ffs.get("workspace_gib", exporter.FFS_DEFAULT_WORKSPACE_GIB))
    _check_live_ffs_cli_contract(args, ffs, "workspace_gib", workspace_gib)
    ffs.update(
        {
            "backend": backend,
            "artifact_id": artifact_id,
            "precision": precision,
            "builder_optimization_level": optimization_level,
            "workspace_gib": workspace_gib,
            "width": int(ffs.get("width", 640)),
            "height": int(ffs.get("height", 480)),
            "max_disp": int(ffs.get("max_disp", exporter.FFS_DEFAULT_MAX_DISP)),
            "valid_iters": int(ffs.get("valid_iters", exporter.FFS_DEFAULT_VALID_ITERS)),
        }
    )
    if (ffs["height"], ffs["width"]) != (480, 640):
        raise SystemExit("FFS Builder YAML must use fixed height=480,width=640")
    if ffs["max_disp"] <= 0 or ffs["max_disp"] % 4 != 0:
        raise SystemExit("FFS max_disp must be positive and divisible by 4")
    if ffs["valid_iters"] <= 0:
        raise SystemExit("FFS valid_iters must be positive")
    exporter._resolve_ffs_declared_paths(ffs, builder_path.parent)
    exporter._fill_ffs_artifact_defaults(ffs, builder_path.parent, backend, artifact_id)
    exporter._resolve_ffs_declared_paths(ffs, builder_path.parent)
    try:
        exporter._preflight_ffs_artifacts(
            ffs,
            config_dir=builder_path.parent,
            backend=backend,
            artifact_id=artifact_id,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(f"FFS live artifact preflight failed: {exc}") from exc
    return "ffs_stereo"


def _check_live_ffs_cli_contract(
    args: argparse.Namespace,
    ffs: Mapping[str, Any],
    key: str,
    yaml_value: Any,
) -> None:
    cli_key = {
        "backend": "ffs_backend",
        "artifact_id": "ffs_artifact_id",
        "precision": "ffs_precision",
        "builder_optimization_level": "ffs_builder_optimization_level",
        "workspace_gib": "ffs_workspace_gib",
    }[key]
    cli_value = getattr(args, cli_key, None)
    if cli_value is None:
        return
    if key in {"builder_optimization_level", "workspace_gib"}:
        if float(cli_value) != float(yaml_value):
            raise SystemExit(
                f"FFS CLI/YAML conflict for {key}: cli={cli_value!r}, yaml={yaml_value!r}"
            )
    elif str(cli_value).strip().lower() != str(yaml_value).strip().lower():
        raise SystemExit(
            f"FFS CLI/YAML conflict for {key}: cli={cli_value!r}, yaml={yaml_value!r}"
        )


def _validate_camera_builder_contract(runtime: Runtime, builder: PointCloudBuilder) -> None:
    if runtime.camera_width != builder.camera.width or runtime.camera_height != builder.camera.height:
        raise SystemExit(
            f"Camera {runtime.camera_width}x{runtime.camera_height} does not match "
            f"PointCloudBuilder {builder.camera.width}x{builder.camera.height}"
        )


def _make_viewer(runtime: Runtime, builder: PointCloudBuilder) -> LiveVisualizationPublisher | None:
    if not runtime.visualize:
        return None
    intrinsics = builder.camera.active_intrinsics
    return LiveVisualizationPublisher(
        rate_hz=runtime.viewer_rate_hz,
        max_raw_points=runtime.visualization_max_raw_points,
        max_cropped_points=runtime.visualization_max_cropped_points,
        viewer_config=ViewerConfig(
            title="DP3 Perception Only | No Robot Connection",
            camera_width=int(builder.camera.width),
            camera_height=int(builder.camera.height),
            camera_fx=float(intrinsics.fx),
            camera_fy=float(intrinsics.fy),
            camera_cx=float(intrinsics.cx),
            camera_cy=float(intrinsics.cy),
            depth_scale=float(builder.camera.depth_scale),
            point_size=runtime.visualization_point_size,
        ),
    )


def _discard_warmup_frames(camera: Any, runtime: Runtime) -> None:
    ratios: list[float] = []
    print(f"[perception-only] discarding {runtime.warmup_frames} warmup frames")
    for index in range(runtime.warmup_frames):
        frame = camera.read_rgbd_ir(timeout_ms=runtime.camera_timeout_ms)
        stats = _depth_stats(frame["depth"], depth_scale=1.0)
        ratios.append(float(stats["valid_ratio"]))
        if index < 3 or (index + 1) % runtime.print_every == 0:
            print(
                f"[perception-only] warmup={index + 1}/{runtime.warmup_frames} "
                f"valid_ratio={ratios[-1]:.3f}"
            )
    recent = np.asarray(ratios[-min(runtime.stability_window, len(ratios)) :])
    print(
        "[perception-only] warmup valid ratio "
        f"median={np.median(recent):.3f} min={np.min(recent):.3f} "
        f"max={np.max(recent):.3f} range={np.ptp(recent):.3f}"
    )


def _frame_record(
    *,
    index: int,
    frame: Mapping[str, Any],
    sampled: Any,
    stages: Mapping[str, Any],
    meta: Mapping[str, Any],
    builder: PointCloudBuilder,
    capture_ms: float,
    build_ms: float,
) -> dict[str, Any]:
    depth = np.asarray(frame["depth"])
    rgb = np.asarray(frame["rgb"])
    return {
        "frame": int(index),
        "depth_source": str(meta.get("depth_source", "native_depth")),
        "wall_time": time.time(),
        "camera_timestamp_ms": _optional_float(frame.get("timestamp")),
        "camera_frame_index": _optional_int(frame.get("frame_index")),
        "capture_ms": float(capture_ms),
        "pointcloud_build_ms": float(build_ms),
        "buffer": {
            "depth_owns_data": bool(depth.flags.owndata),
            "depth_c_contiguous": bool(depth.flags.c_contiguous),
            "rgb_owns_data": bool(rgb.flags.owndata),
            "rgb_c_contiguous": bool(rgb.flags.c_contiguous),
        },
        "depth": _depth_stats(depth, depth_scale=float(builder.camera.depth_scale)),
        "point_cloud": {
            "num_raw_points": int(meta["num_raw_points"]),
            "num_cropped_points": int(meta["num_cropped_points"]),
            "num_sampled_points": int(meta["num_sampled_points"]),
            "padded": bool(meta["padded"]),
            "raw_xyz": _point_stats(stages["raw"]),
            "cropped_xyz": _point_stats(stages["cropped"]),
            "sampled_xyz": _point_stats(sampled),
        },
    }


def _depth_stats(depth: Any, *, depth_scale: float) -> dict[str, Any]:
    array = np.asarray(depth)
    valid = np.isfinite(array) & (array > 0)
    count = int(np.count_nonzero(valid))
    total = int(array.size)
    values_m = array[valid].astype(np.float64, copy=False) * float(depth_scale)
    if values_m.size:
        q05, median, q95 = np.quantile(values_m, [0.05, 0.5, 0.95])
        minimum = float(np.min(values_m))
        maximum = float(np.max(values_m))
    else:
        minimum = maximum = q05 = median = q95 = None
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "valid_count": count,
        "invalid_count": total - count,
        "valid_ratio": float(count / total) if total else 0.0,
        "min_m": minimum,
        "q05_m": None if q05 is None else float(q05),
        "median_m": None if median is None else float(median),
        "q95_m": None if q95 is None else float(q95),
        "max_m": maximum,
    }


def _point_stats(points: Any) -> dict[str, Any]:
    if hasattr(points, "detach"):
        xyz = points[:, :3]
        if int(xyz.shape[0]) == 0:
            return {"count": 0, "min": None, "max": None, "mean": None}
        return {
            "count": int(xyz.shape[0]),
            "min": xyz.amin(dim=0).detach().cpu().tolist(),
            "max": xyz.amax(dim=0).detach().cpu().tolist(),
            "mean": xyz.mean(dim=0).detach().cpu().tolist(),
        }
    xyz_np = np.asarray(points)[:, :3]
    if not len(xyz_np):
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": int(len(xyz_np)),
        "min": xyz_np.min(axis=0).tolist(),
        "max": xyz_np.max(axis=0).tolist(),
        "mean": xyz_np.mean(axis=0).tolist(),
    }


def _summarize_records(
    records: Sequence[Mapping[str, Any]],
    runtime: Runtime,
    *,
    interrupted: bool,
) -> dict[str, Any]:
    ratios = np.asarray([record["depth"]["valid_ratio"] for record in records], dtype=float)
    raw_counts = np.asarray(
        [record["point_cloud"]["num_raw_points"] for record in records], dtype=float
    )
    cropped_counts = np.asarray(
        [record["point_cloud"]["num_cropped_points"] for record in records], dtype=float
    )
    if not len(records):
        return {
            "frames": 0,
            "interrupted": interrupted,
            "depth_source": getattr(runtime, "depth_source", "native_depth"),
            "quality_gate": {"passed": False, "failures": ["no measured frames"]},
        }
    recent = ratios[-min(runtime.stability_window, len(ratios)) :]
    ratio_median = float(np.median(recent))
    ratio_range = float(np.ptp(recent))
    failures: list[str] = []
    if ratio_median < runtime.min_valid_depth_ratio:
        failures.append(
            f"depth valid ratio median {ratio_median:.3f} < "
            f"{runtime.min_valid_depth_ratio:.3f}"
        )
    if ratio_range > runtime.max_valid_depth_ratio_range:
        failures.append(
            f"depth valid ratio range {ratio_range:.3f} > "
            f"{runtime.max_valid_depth_ratio_range:.3f}"
        )
    if any(bool(record["point_cloud"]["padded"]) for record in records):
        failures.append("at least one sampled point cloud was padded")
    if not all(bool(record["buffer"]["depth_owns_data"]) for record in records):
        failures.append("at least one depth array did not own its memory")
    if not all(bool(record["buffer"]["depth_c_contiguous"]) for record in records):
        failures.append("at least one depth array was not C-contiguous")

    return {
        "frames": len(records),
        "interrupted": interrupted,
        "depth_source": getattr(runtime, "depth_source", "native_depth"),
        "camera_serial": runtime.camera_serial,
        "thresholds": {
            "stability_window": runtime.stability_window,
            "min_valid_depth_ratio": runtime.min_valid_depth_ratio,
            "max_valid_depth_ratio_range": runtime.max_valid_depth_ratio_range,
        },
        "depth_valid_ratio": _numeric_summary(ratios),
        "num_raw_points": _numeric_summary(raw_counts),
        "num_cropped_points": _numeric_summary(cropped_counts),
        "quality_gate": {
            "passed": not failures,
            "recent_valid_ratio_median": ratio_median,
            "recent_valid_ratio_range": ratio_range,
            "failures": failures,
        },
    }


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _print_record(record: Mapping[str, Any]) -> None:
    depth = record["depth"]
    pc = record["point_cloud"]
    print(
        f"[perception-only] frame={record['frame']} valid={depth['valid_ratio']:.3f} "
        f"raw={pc['num_raw_points']} crop={pc['num_cropped_points']} "
        f"sampled={pc['num_sampled_points']} padded={pc['padded']} "
        f"capture_ms={record['capture_ms']:.1f} build_ms={record['pointcloud_build_ms']:.1f}"
    )


def _print_summary(summary: Mapping[str, Any], summary_path: Path) -> None:
    gate = summary["quality_gate"]
    print(f"[perception-only] summary: {summary_path}")
    print(f"[perception-only] quality gate: {'PASS' if gate['passed'] else 'FAIL'}")
    for failure in gate.get("failures", []):
        print(f"[perception-only] failure: {failure}")


def _mapping(container: Mapping[str, Any], key: str, *, required: bool = True) -> Mapping[str, Any]:
    value = container.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise SystemExit(f"Expected mapping at {key!r}")
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise SystemExit(f"YAML must contain a mapping: {path}")
    return value


def _resolve_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value).copy()


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


if __name__ == "__main__":
    raise SystemExit(main())
