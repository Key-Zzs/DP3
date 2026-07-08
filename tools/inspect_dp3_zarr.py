#!/usr/bin/env python
"""Inspect a DP3 replay-buffer zarr exported from LeRobot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_DATA_KEYS = ["state", "action", "point_cloud"]


def inspect_dp3_zarr(zarr_path: str | Path) -> dict[str, Any]:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("zarr is required to inspect DP3 replay buffers") from exc

    path = Path(zarr_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    root = zarr.open(str(path), mode="r")
    if "data" not in root:
        raise KeyError("Missing zarr group: data")
    if "meta" not in root:
        raise KeyError("Missing zarr group: meta")
    data = root["data"]
    meta = root["meta"]

    for key in REQUIRED_DATA_KEYS:
        if key not in data:
            raise KeyError(f"Missing zarr array: data/{key}")
    if "episode_ends" not in meta:
        raise KeyError("Missing zarr array: meta/episode_ends")

    state = data["state"]
    action = data["action"]
    point_cloud = data["point_cloud"]
    episode_ends = meta["episode_ends"][:]
    if point_cloud.ndim != 3:
        raise ValueError(f"data/point_cloud must be T x N x C, got {point_cloud.shape}")
    if not np.all(np.diff(episode_ends) > 0):
        raise ValueError("meta/episode_ends must be strictly increasing")
    if int(episode_ends[-1]) != int(state.shape[0]):
        raise ValueError(
            f"episode_ends[-1] ({episode_ends[-1]}) does not equal T ({state.shape[0]})"
        )
    for key, array in [("state", state), ("action", action), ("point_cloud", point_cloud)]:
        if not _array_is_finite(array):
            raise ValueError(f"data/{key} contains NaN or Inf")

    summary = {
        "path": str(path),
        "state": _array_summary(state),
        "action": _array_summary(action),
        "point_cloud": _array_summary(point_cloud),
        "episode_ends": {
            "shape": tuple(episode_ends.shape),
            "dtype": str(episode_ends.dtype),
            "values": episode_ends.tolist(),
        },
        "fixed_size_point_cloud": True,
        "attrs": dict(root.attrs),
    }
    if "img" in data:
        img = data["img"]
        summary["img"] = {
            "shape": tuple(img.shape),
            "dtype": str(img.dtype),
            "min": int(np.min(img[:])),
            "max": int(np.max(img[:])),
        }
    _print_summary(summary)
    return summary


def _array_summary(array: Any) -> dict[str, Any]:
    return {
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "min": float(np.min(array[:])),
        "max": float(np.max(array[:])),
    }


def _array_is_finite(array: Any) -> bool:
    return bool(np.isfinite(array[:]).all())


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"zarr_path: {summary['path']}")
    for key in ["state", "action", "point_cloud"]:
        item = summary[key]
        print(
            f"data/{key}: shape={item['shape']} dtype={item['dtype']} "
            f"min={item['min']:.6g} max={item['max']:.6g}"
        )
    if "img" in summary:
        item = summary["img"]
        print(
            f"data/img: shape={item['shape']} dtype={item['dtype']} "
            f"min={item['min']} max={item['max']}"
        )
    episode = summary["episode_ends"]
    print(
        f"meta/episode_ends: shape={episode['shape']} dtype={episode['dtype']} "
        f"values={episode['values']}"
    )
    print(f"fixed_size_point_cloud: {summary['fixed_size_point_cloud']}")
    print("attrs:")
    print(json.dumps(summary["attrs"], indent=2, ensure_ascii=False)[:8000])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-path", required=True, help="Path to a DP3 zarr replay buffer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_dp3_zarr(args.zarr_path)


if __name__ == "__main__":
    main()
