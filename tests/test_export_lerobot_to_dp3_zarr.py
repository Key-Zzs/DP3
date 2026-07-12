from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "export_lerobot_to_dp3_zarr.py"
SPEC = importlib.util.spec_from_file_location("export_lerobot_to_dp3_zarr", MODULE_PATH)
exporter = importlib.util.module_from_spec(SPEC)
sys.modules["export_lerobot_to_dp3_zarr"] = exporter
assert SPEC.loader is not None
SPEC.loader.exec_module(exporter)


def _calibration() -> dict:
    camera = {
        "depth_scale_m_per_unit": 0.001,
        "streams": {
            "depth": {
                "intrinsics": {
                    "width": 640,
                    "height": 480,
                    "fx": 392.5,
                    "fy": 392.5,
                    "cx": 316.8,
                    "cy": 235.8,
                }
            },
            "color": {
                "intrinsics": {
                    "width": 640,
                    "height": 480,
                    "fx": 606.1,
                    "fy": 606.6,
                    "cx": 320.8,
                    "cy": 256.1,
                }
            },
        },
        "extrinsics": {
            "depth_to_color": {
                "rotation_matrix_row_major": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "translation_m": [0.015, 0.0, 0.0],
            }
        },
    }
    return {
        "cameras": {
            "head_rgb": camera,
            "left_wrist_rgb": camera,
            "right_wrist_rgb": camera,
        }
    }


def test_build_pointcloud_builder_config_xyz() -> None:
    config = exporter.build_pointcloud_builder_config(
        _calibration(),
        camera="head",
        pointcloud_mode="xyz",
        num_points=1024,
    )
    assert config["camera"]["aligned_depth_to_color"] is False
    assert config["camera"]["depth_scale"] == 0.001
    assert config["pointcloud"]["use_rgb"] is False
    assert config["pointcloud"]["output_format"] == "xyz"
    assert config["sampling"]["num_points"] == 1024


def test_build_pointcloud_builder_config_xyzrgb() -> None:
    config = exporter.build_pointcloud_builder_config(
        _calibration(),
        camera="head",
        pointcloud_mode="xyzrgb",
        num_points=512,
    )
    assert config["pointcloud"]["use_rgb"] is True
    assert config["pointcloud"]["rgb_mapping"] == "project_depth_to_color"
    assert config["pointcloud"]["rgb_sampling"] == "nearest"
    assert config["camera"]["depth_to_color_extrinsics"]["translation"] == [0.015, 0.0, 0.0]
    assert config["sampling"]["num_points"] == 512


def test_compute_episode_ends_from_episode_rows_and_max_frames() -> None:
    rows = [
        {"episode_index": 0, "length": 4, "dataset_from_index": 0, "dataset_to_index": 4},
        {"episode_index": 1, "length": 5, "dataset_from_index": 4, "dataset_to_index": 9},
    ]
    assert exporter.compute_episode_ends(episode_rows=rows, total_frames=9).tolist() == [4, 9]
    assert exporter.compute_episode_ends(
        episode_rows=rows,
        total_frames=9,
        max_frames=6,
    ).tolist() == [4, 6]


def test_compute_episode_ends_from_indices() -> None:
    indices = np.asarray([0, 0, 0, 1, 1, 2])
    ends = exporter.compute_episode_ends(episode_indices=indices, total_frames=6)
    assert ends.tolist() == [3, 5, 6]


def test_default_output_zarr_path_uses_sanitized_repo_id(tmp_path) -> None:
    output = exporter.default_output_zarr_path(
        tmp_path / "dataset",
        {"repo_id": "owner/my dataset"},
        camera="head",
        pointcloud_mode="xyzrgb",
        output_root=tmp_path,
    )
    assert output == tmp_path / "owner_my_dataset_head_xyzrgb.zarr"


def test_default_output_zarr_path_falls_back_to_lerobot_cache_relative(tmp_path) -> None:
    lerobot_path = Path.home() / ".cache" / "huggingface" / "lerobot" / "org" / "dataset"
    output = exporter.default_output_zarr_path(
        lerobot_path,
        {},
        camera="left_wrist",
        pointcloud_mode="xyz",
        output_root=tmp_path,
    )
    assert output == tmp_path / "org_dataset_left_wrist_xyz.zarr"


def test_write_dp3_zarr_xyz_and_overwrite(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    assert zarr is not None
    output = tmp_path / "mock_xyz.zarr"
    state = np.zeros((2, 28), dtype=np.float32)
    action = np.zeros((2, 14), dtype=np.float32)
    point_cloud = np.zeros((2, 8, 3), dtype=np.float32)
    exporter.write_dp3_zarr(
        output,
        state=state,
        action=action,
        point_cloud=point_cloud,
        episode_ends=np.asarray([2], dtype=np.int64),
        attrs={"source_lerobot_path": "/tmp/mock", "pointcloud_mode": "xyz"},
        overwrite=False,
    )
    root = zarr.open(str(output), mode="r")
    assert root["data"]["point_cloud"].shape == (2, 8, 3)
    assert root.attrs["export_status"] == "complete"
    assert root.attrs["converted_frames"] == 2
    assert exporter.verify_dp3_zarr(output)["state"] == root.attrs["integrity"]["state"]
    with pytest.raises(FileExistsError):
        exporter.write_dp3_zarr(
            output,
            state=state,
            action=action,
            point_cloud=point_cloud,
            episode_ends=np.asarray([2], dtype=np.int64),
            attrs={},
            overwrite=False,
        )
    exporter.write_dp3_zarr(
        output,
        state=state,
        action=action,
        point_cloud=point_cloud,
        episode_ends=np.asarray([2], dtype=np.int64),
        attrs={},
        overwrite=True,
    )


def test_write_dp3_zarr_xyzrgb(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    assert zarr is not None
    output = tmp_path / "mock_xyzrgb.zarr"
    exporter.write_dp3_zarr(
        output,
        state=np.zeros((3, 28), dtype=np.float32),
        action=np.zeros((3, 14), dtype=np.float32),
        point_cloud=np.zeros((3, 4, 6), dtype=np.float32),
        episode_ends=np.asarray([1, 3], dtype=np.int64),
        attrs={"pointcloud_mode": "xyzrgb"},
        overwrite=True,
    )
    root = zarr.open(str(output), mode="r")
    assert root["data"]["point_cloud"].shape == (3, 4, 6)
    assert root["meta"]["episode_ends"][:].tolist() == [1, 3]
    assert exporter.verify_dp3_zarr(output) == root.attrs["integrity"]


def test_verify_rejects_incomplete_export(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    output = tmp_path / "incomplete.zarr"
    root = zarr.group(str(output))
    root.attrs["export_status"] = "in_progress"

    with pytest.raises(ValueError, match="export is not complete"):
        exporter.verify_dp3_zarr(output)


def test_verify_rejects_modified_array(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    output = tmp_path / "modified.zarr"
    exporter.write_dp3_zarr(
        output,
        state=np.zeros((2, 28), dtype=np.float32),
        action=np.zeros((2, 14), dtype=np.float32),
        point_cloud=np.zeros((2, 4, 3), dtype=np.float32),
        episode_ends=np.asarray([2], dtype=np.int64),
        attrs={},
    )
    root = zarr.open(str(output), mode="a")
    root["data"]["action"][1, 0] = 1.0

    with pytest.raises(ValueError, match="checksum mismatch"):
        exporter.verify_dp3_zarr(output)
