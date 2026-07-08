#!/usr/bin/env python
"""Visualize one source LeRobot RGB-D frame through the DP3 point-cloud stages."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import export_lerobot_to_dp3_zarr as exporter


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lerobot-path",
        required=True,
        help="Absolute path to the source LeRobot dataset.",
    )
    parser.add_argument(
        "--frame-index",
        type=_positive_int,
        required=True,
        help="Zero-based dataset/export row index to inspect.",
    )
    parser.add_argument("--camera", choices=sorted(exporter.CAMERA_SPECS), default="head")
    parser.add_argument("--pointcloud-mode", choices=["xyz", "xyzrgb"], default="xyz")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--builder-config", help="Optional PointCloudBuilder YAML path.")
    parser.add_argument("--window-width", type=int, default=1800)
    parser.add_argument("--window-height", type=int, default=760)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--no-show", action="store_true", help="Process and print stats without Open3D GUI.")
    args = parser.parse_args()
    if args.num_points is not None and args.num_points <= 0:
        parser.error("--num-points must be positive")
    if args.window_width <= 0 or args.window_height <= 0:
        parser.error("--window-width and --window-height must be positive")
    if args.point_size <= 0:
        parser.error("--point-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    return run_debug(args)


def run_debug(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="dp3_pc_debug_") as tmp_dir:
        resolved = _resolve_debug_inputs(args, Path(tmp_dir))
        stages, meta, row, builder_config_path = _build_frame_stages(resolved)
        _print_summary(resolved, row, builder_config_path, stages, meta)
        if not args.no_show:
            _show_stages_open3d(
                stages,
                title=(
                    f"{resolved.lerobot_path.name} frame {resolved.frame_index} "
                    f"({resolved.camera}, {resolved.pointcloud_mode})"
                ),
                width=args.window_width,
                height=args.window_height,
                point_size=args.point_size,
            )
    return 0


class DebugInputs(argparse.Namespace):
    lerobot_path: Path
    frame_index: int
    camera: str
    pointcloud_mode: str
    num_points: int
    builder_config: str | None
    dp3_zarr: Path | None
    temp_config_path: Path | None


def _resolve_debug_inputs(args: argparse.Namespace, tmp_dir: Path) -> DebugInputs:
    lerobot_path = Path(args.lerobot_path).expanduser()
    if not lerobot_path.is_absolute():
        raise ValueError("--lerobot-path must be an absolute path")
    lerobot_path = lerobot_path.resolve()
    if not lerobot_path.exists():
        raise FileNotFoundError(f"LeRobot dataset path does not exist: {lerobot_path}")

    camera = args.camera
    if camera not in exporter.CAMERA_SPECS:
        raise ValueError(f"Unsupported camera: {camera}")
    pointcloud_mode = args.pointcloud_mode
    if pointcloud_mode not in {"xyz", "xyzrgb"}:
        raise ValueError("--pointcloud-mode must be 'xyz' or 'xyzrgb'")
    num_points = int(args.num_points)
    if num_points <= 0:
        raise ValueError("--num-points must be positive")

    builder_config = args.builder_config
    temp_config_path: Path | None = None
    if builder_config is None:
        temp_config_path = tmp_dir / "pointcloud_builder_debug.yaml"

    resolved = DebugInputs()
    resolved.lerobot_path = lerobot_path
    resolved.frame_index = int(args.frame_index)
    resolved.camera = camera
    resolved.pointcloud_mode = pointcloud_mode
    resolved.num_points = num_points
    resolved.builder_config = builder_config
    dp3_zarr = getattr(args, "dp3_zarr", None)
    resolved.dp3_zarr = Path(dp3_zarr).expanduser().resolve() if dp3_zarr else None
    resolved.temp_config_path = temp_config_path
    return resolved


def _build_frame_stages(
    args: DebugInputs,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    data_paths = exporter._data_parquet_paths(args.lerobot_path)
    total_frames = exporter._count_parquet_rows(data_paths)
    if args.frame_index >= total_frames:
        raise IndexError(f"--frame-index {args.frame_index} is outside dataset length {total_frames}")

    realsense_calibration = exporter._read_json(
        args.lerobot_path / "meta" / "realsense_calibration.json"
    )
    builder_config_path, builder_config = _resolve_builder_config_for_debug(
        args=args,
        realsense_calibration=realsense_calibration,
    )

    PointCloudBuilder = exporter._import_pointcloud_builder()
    builder = PointCloudBuilder.from_yaml(builder_config_path)

    camera_spec = exporter.CAMERA_SPECS[args.camera]
    need_rgb = args.pointcloud_mode == "xyzrgb"
    columns = [
        camera_spec["depth_column"],
        "global_frame_index",
        camera_spec["timestamp_column"],
        camera_spec["reused_column"],
        "episode_index",
        "frame_index",
        "index",
    ]
    row, source_path = _read_row_at_index(data_paths, columns=columns, frame_index=args.frame_index)
    depth = exporter._as_depth(row[camera_spec["depth_column"]], camera_spec["depth_column"], source_path)

    frame: dict[str, Any] = {
        "depth": depth,
        "timestamp": row[camera_spec["timestamp_column"]],
        "global_frame_index": row["global_frame_index"],
    }
    if need_rgb:
        video_paths = exporter._video_paths(args.lerobot_path, camera_spec["video_key"])
        frame["rgb"] = _read_rgb_at_index(video_paths, frame_index=args.frame_index)

    stages, meta = builder.build_stages(frame)
    sampled = stages["sampled"].detach().cpu().numpy().astype(np.float32, copy=False)
    pointcloud_dim = 6 if args.pointcloud_mode == "xyzrgb" else 3
    exporter._validate_point_cloud(sampled, args.num_points, pointcloud_dim, args.frame_index)
    exporter._reject_nonfinite(sampled, "point_cloud", args.frame_index)

    row["_source_parquet"] = str(source_path)
    row["_builder_config"] = builder_config
    return stages, meta, row, builder_config_path


def _resolve_builder_config_for_debug(
    *,
    args: DebugInputs,
    realsense_calibration: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    if args.builder_config:
        path = Path(args.builder_config).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Builder config does not exist: {path}")
        return path, exporter._read_yaml(path)

    config = exporter.build_pointcloud_builder_config(
        realsense_calibration,
        camera=args.camera,
        pointcloud_mode=args.pointcloud_mode,
        num_points=args.num_points,
    )
    if args.temp_config_path is None:
        raise RuntimeError("Internal error: temp_config_path was not prepared")
    config_path = args.temp_config_path
    _write_yaml(config_path, config)
    return config_path.resolve(), config


def _read_row_at_index(
    data_paths: list[Path],
    *,
    columns: list[str],
    frame_index: int,
) -> tuple[dict[str, Any], Path]:
    row: dict[str, Any] | None = None
    source_path: Path | None = None
    for idx, (candidate, path) in enumerate(
        exporter.iter_lerobot_rows(data_paths, columns=columns, max_frames=frame_index + 1)
    ):
        if idx == frame_index:
            row = candidate
            source_path = path
            break
    if row is None or source_path is None:
        raise IndexError(f"Failed to read frame {frame_index}")
    return row, source_path


def _read_rgb_at_index(video_paths: list[Path], *, frame_index: int) -> np.ndarray:
    for idx, rgb in enumerate(exporter.iter_video_frames(video_paths)):
        if idx == frame_index:
            return rgb
    raise IndexError(f"RGB video ended before frame {frame_index}")


def _show_stages_open3d(
    stages: dict[str, Any],
    *,
    title: str,
    width: int,
    height: int,
    point_size: float,
) -> None:
    import open3d as o3d  # type: ignore[import-not-found]
    from open3d.visualization import gui, rendering  # type: ignore[import-not-found]

    app = gui.Application.instance
    app.initialize()
    window = app.create_window(title, width, height)

    labels: list[Any] = []
    scenes: list[Any] = []
    stage_names = ["raw", "cropped", "sampled"]
    for name in stage_names:
        point_cloud_np = _tensor_to_numpy(stages[name])
        geometry = _to_open3d_point_cloud(o3d, point_cloud_np, name)
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)
        scene.scene.set_background([1.0, 1.0, 1.0, 1.0])
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = point_size
        scene.scene.add_geometry(name, geometry, material)
        _setup_camera(o3d, scene, geometry)

        label = gui.Label(_stage_label(name, point_cloud_np))
        labels.append(label)
        scenes.append(scene)
        window.add_child(label)
        window.add_child(scene)

    def on_layout(layout_context: Any) -> None:
        content = window.content_rect
        em = window.theme.font_size
        gap = int(0.5 * em)
        label_h = int(3.5 * em)
        panel_w = max(1, int((content.width - gap * (len(stage_names) - 1)) / len(stage_names)))
        for idx, (label, scene) in enumerate(zip(labels, scenes, strict=True)):
            x = content.x + idx * (panel_w + gap)
            label.frame = gui.Rect(x, content.y, panel_w, label_h)
            scene.frame = gui.Rect(
                x,
                content.y + label_h,
                panel_w,
                max(1, content.height - label_h),
            )

    window.set_on_layout(on_layout)
    app.run()


def _setup_camera(o3d: Any, scene: Any, geometry: Any) -> None:
    bounds = geometry.get_axis_aligned_bounding_box()
    extent = np.asarray(bounds.get_extent(), dtype=np.float64)
    center = np.asarray(bounds.get_center(), dtype=np.float64)
    if geometry.is_empty() or not np.isfinite(extent).all() or float(np.max(extent)) <= 1e-9:
        if not np.isfinite(center).all():
            center = np.zeros(3, dtype=np.float64)
        bounds = o3d.geometry.AxisAlignedBoundingBox(center - 0.05, center + 0.05)
    scene.setup_camera(60.0, bounds, bounds.get_center())


def _tensor_to_numpy(point_cloud: Any) -> np.ndarray:
    pc = point_cloud.detach().to("cpu").numpy()
    if pc.ndim != 2 or pc.shape[1] not in {3, 6}:
        raise ValueError(f"Expected point cloud shape N x 3 or N x 6, got {pc.shape}")
    return pc.astype(np.float64, copy=False)


def _to_open3d_point_cloud(o3d: Any, point_cloud: np.ndarray, stage: str) -> Any:
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    if point_cloud.shape[1] == 6:
        geometry.colors = o3d.utility.Vector3dVector(np.clip(point_cloud[:, 3:6], 0.0, 1.0))
    else:
        geometry.paint_uniform_color(_stage_color(stage))
    return geometry


def _stage_color(stage: str) -> list[float]:
    return {
        "raw": [0.55, 0.55, 0.55],
        "cropped": [0.1, 0.35, 0.85],
        "sampled": [0.95, 0.45, 0.1],
    }[stage]


def _stage_label(stage: str, point_cloud: np.ndarray) -> str:
    if point_cloud.shape[0] == 0:
        return f"{stage}\npoints: 0\nxyz: empty"
    mins = point_cloud[:, :3].min(axis=0)
    maxs = point_cloud[:, :3].max(axis=0)
    return (
        f"{stage}\n"
        f"points: {point_cloud.shape[0]}  dim: {point_cloud.shape[1]}\n"
        f"x[{mins[0]:.3f}, {maxs[0]:.3f}] y[{mins[1]:.3f}, {maxs[1]:.3f}] "
        f"z[{mins[2]:.3f}, {maxs[2]:.3f}]"
    )


def _print_summary(
    args: DebugInputs,
    row: dict[str, Any],
    builder_config_path: Path,
    stages: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    print("Point-cloud debug summary")
    print(f"  lerobot_path: {args.lerobot_path}")
    if args.dp3_zarr is not None:
        print(f"  dp3_zarr: {args.dp3_zarr}")
    print(f"  frame_index: {args.frame_index}")
    print(f"  parquet: {row['_source_parquet']}")
    print(f"  global_frame_index: {row.get('global_frame_index')}")
    print(f"  episode_index: {row.get('episode_index')}")
    print(f"  episode_frame_index: {row.get('frame_index')}")
    print(f"  index: {row.get('index')}")
    print(f"  camera: {args.camera}")
    print(f"  pointcloud_mode: {args.pointcloud_mode}")
    print(f"  num_points: {args.num_points}")
    print(f"  builder_config_path: {builder_config_path}")
    print(f"  crop_enabled: {meta.get('crop_enabled')}")
    print(f"  crop_range: {meta.get('crop_range')}")
    print(f"  sampling_mode: {meta.get('sampling_mode')}")
    print(f"  sampling: {meta.get('sampling')}")
    for name in ("raw", "cropped", "sampled"):
        point_cloud = _tensor_to_numpy(stages[name])
        print(f"  {name}_shape: {tuple(point_cloud.shape)}")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


if __name__ == "__main__":
    raise SystemExit(main())
