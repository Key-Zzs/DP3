"""Visualize one point-cloud frame from a DP3 zarr dataset with Open3D."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import zarr


def open_point_cloud_array(path: str | Path) -> Any:
    """Open either a zarr root or a direct data/point_cloud zarr array."""

    zarr_path = Path(path).expanduser()
    if not zarr_path.is_absolute():
        raise ValueError(f"--zarr-path must be an absolute path: {zarr_path}")
    if not zarr_path.exists():
        raise FileNotFoundError(zarr_path)

    node = zarr.open(str(zarr_path), mode="r")
    if hasattr(node, "shape"):
        return node
    if "data" in node and "point_cloud" in node["data"]:
        return node["data"]["point_cloud"]
    if "point_cloud" in node:
        return node["point_cloud"]
    raise KeyError(
        f"Could not find point_cloud array in {zarr_path}. "
        "Pass either the zarr root or the data/point_cloud array path."
    )


def load_point_cloud_frame(path: str | Path, frame: int) -> np.ndarray:
    """Load one frame and validate that it is an Nx3 or Nx6 point cloud."""

    array = open_point_cloud_array(path)
    if len(array.shape) != 3:
        raise ValueError(f"point_cloud array must have shape T x N x C, got {array.shape}")
    if frame < 0 or frame >= int(array.shape[0]):
        raise IndexError(f"frame {frame} out of range [0, {int(array.shape[0]) - 1}]")

    point_cloud = np.asarray(array[frame], dtype=np.float32)
    if point_cloud.ndim != 2 or point_cloud.shape[1] not in {3, 6}:
        raise ValueError(f"point cloud frame must have shape N x 3 or N x 6, got {point_cloud.shape}")
    if not np.isfinite(point_cloud).all():
        raise ValueError(f"point cloud frame {frame} contains NaN or Inf")
    return point_cloud


def make_open3d_point_cloud(
    point_cloud: np.ndarray,
    *,
    max_points: int | None = None,
) -> Any:
    """Create an Open3D point cloud, auto-detecting XYZ vs XYZRGB."""

    import open3d as o3d

    pc = _subsample(point_cloud, max_points)
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(pc[:, :3].astype(np.float64))
    if pc.shape[1] == 6:
        geometry.colors = o3d.utility.Vector3dVector(_normalize_rgb(pc[:, 3:6]))
    else:
        geometry.colors = o3d.utility.Vector3dVector(_colorize_by_z(pc[:, :3]))
    return geometry


def show_open3d_point_cloud(
    geometry: Any,
    *,
    window_name: str,
    point_size: float,
    background: tuple[float, float, float],
) -> None:
    """Display an Open3D point cloud with stable render options."""

    import open3d as o3d

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name=window_name)
    visualizer.add_geometry(geometry)
    render_option = visualizer.get_render_option()
    render_option.point_size = float(point_size)
    render_option.background_color = np.asarray(background, dtype=np.float64)
    visualizer.run()
    visualizer.destroy_window()


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb_float = np.asarray(rgb, dtype=np.float32)
    if rgb_float.size > 0 and float(np.nanmax(rgb_float)) > 1.0:
        rgb_float = rgb_float / 255.0
    return np.clip(rgb_float, 0.0, 1.0).astype(np.float64)


def _colorize_by_z(xyz: np.ndarray) -> np.ndarray:
    z = xyz[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    scale = z_max - z_min
    if scale <= 1e-12:
        t = np.zeros_like(z, dtype=np.float32)
    else:
        t = ((z - z_min) / scale).astype(np.float32)
    colors = np.stack(
        [
            t,
            0.25 + 0.5 * (1.0 - np.abs(t - 0.5) * 2.0),
            1.0 - t,
        ],
        axis=1,
    )
    return np.clip(colors, 0.0, 1.0).astype(np.float64)


def _subsample(point_cloud: np.ndarray, max_points: int | None) -> np.ndarray:
    if max_points is None or max_points <= 0 or point_cloud.shape[0] <= max_points:
        return point_cloud
    indices = np.linspace(0, point_cloud.shape[0] - 1, max_points, dtype=np.int64)
    return point_cloud[indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr-path",
        required=True,
        help="Absolute path to a DP3 zarr root or its data/point_cloud array.",
    )
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize.")
    parser.add_argument("--point-size", type=float, default=3.0, help="Open3D point size.")
    parser.add_argument(
        "--background",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("R", "G", "B"),
        help="Open3D background color in [0,1].",
    )
    parser.add_argument("--max-points", type=int, help="Optional display-only deterministic subsample.")
    parser.add_argument("--no-show", action="store_true", help="Load and validate without opening Open3D.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zarr_path = Path(args.zarr_path).expanduser()
    point_cloud = load_point_cloud_frame(zarr_path, args.frame)
    mode = "xyzrgb" if point_cloud.shape[1] == 6 else "xyz"
    print(f"Loaded frame {args.frame}: shape={point_cloud.shape}, mode={mode}")
    if args.max_points is not None and args.max_points > 0:
        print(f"Display subsample: max_points={args.max_points}")
    if args.no_show:
        return 0

    geometry = make_open3d_point_cloud(point_cloud, max_points=args.max_points)
    window_name = f"{zarr_path.name} frame={args.frame} {mode}"
    show_open3d_point_cloud(
        geometry,
        window_name=window_name,
        point_size=args.point_size,
        background=tuple(args.background),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
