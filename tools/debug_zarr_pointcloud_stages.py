#!/usr/bin/env python
"""Use exported DP3 zarr attrs to debug the source LeRobot point-cloud stages."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import debug_lerobot_pointcloud_stages as lerobot_debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp3-zarr", required=True, help="Exported DP3 zarr path.")
    parser.add_argument(
        "--frame-index",
        type=lerobot_debug._positive_int,
        required=True,
        help="Zero-based exported/source frame index to inspect.",
    )
    parser.add_argument(
        "--lerobot-path",
        help="Override source_lerobot_path stored in zarr attrs.",
    )
    parser.add_argument("--camera", choices=sorted(lerobot_debug.exporter.CAMERA_SPECS), default=None)
    parser.add_argument("--pointcloud-mode", choices=["xyz", "xyzrgb"], default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--builder-config", help="Override PointCloudBuilder YAML path.")
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
    attrs = _read_dp3_attrs(Path(args.dp3_zarr).expanduser())
    with tempfile.TemporaryDirectory(prefix="dp3_zarr_pc_debug_") as tmp_dir:
        lerobot_args = _lerobot_args_from_zarr_attrs(args, attrs, Path(tmp_dir))
        return lerobot_debug.run_debug(lerobot_args)


def _lerobot_args_from_zarr_attrs(
    args: argparse.Namespace,
    attrs: dict[str, Any],
    tmp_dir: Path,
) -> argparse.Namespace:
    lerobot_path = args.lerobot_path or attrs.get("source_lerobot_path")
    if not lerobot_path:
        raise ValueError("zarr attrs do not contain source_lerobot_path; pass --lerobot-path")

    camera = args.camera or attrs.get("camera") or "head"
    pointcloud_mode = args.pointcloud_mode or attrs.get("pointcloud_mode") or "xyz"
    num_points = int(args.num_points or attrs.get("num_points") or 1024)
    builder_config = _resolve_builder_config(args, attrs, tmp_dir)

    return argparse.Namespace(
        lerobot_path=lerobot_path,
        frame_index=args.frame_index,
        camera=camera,
        pointcloud_mode=pointcloud_mode,
        num_points=num_points,
        builder_config=builder_config,
        window_width=args.window_width,
        window_height=args.window_height,
        point_size=args.point_size,
        no_show=args.no_show,
        dp3_zarr=args.dp3_zarr,
    )


def _resolve_builder_config(
    args: argparse.Namespace,
    attrs: dict[str, Any],
    tmp_dir: Path,
) -> str | None:
    if args.builder_config:
        return args.builder_config

    attr_config = attrs.get("pointcloud_builder_config")
    if isinstance(attr_config, dict):
        config_path = tmp_dir / "pointcloud_builder_from_zarr_attrs.yaml"
        _write_yaml(config_path, attr_config)
        return str(config_path)

    attr_config_path = attrs.get("pointcloud_builder_config_path")
    if attr_config_path and Path(attr_config_path).expanduser().exists():
        return str(Path(attr_config_path).expanduser().resolve())

    return None


def _read_dp3_attrs(path: Path) -> dict[str, Any]:
    attrs_path = path / ".zattrs"
    if not attrs_path.exists():
        raise FileNotFoundError(f"DP3 zarr attrs not found: {attrs_path}")
    with attrs_path.open("r", encoding="utf-8") as f:
        attrs = json.load(f)
    if not isinstance(attrs, dict):
        raise ValueError(f"DP3 zarr attrs must be a JSON mapping: {attrs_path}")
    return attrs


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


if __name__ == "__main__":
    raise SystemExit(main())
