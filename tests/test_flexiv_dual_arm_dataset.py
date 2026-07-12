from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))

zarr = pytest.importorskip("zarr")

from diffusion_policy_3d.dataset.flexiv_dual_arm_dataset import FlexivDualArmDataset  # noqa: E402


def _write_zarr(path: Path, *, state_dim: int = 28, action_dim: int = 14, pc_dim: int = 3) -> None:
    root = zarr.group(str(path))
    data = root.create_group("data")
    meta = root.create_group("meta")
    total_steps = 5
    state = data.create_dataset(
        "state",
        data=np.zeros((total_steps, state_dim), dtype=np.float32),
        chunks=(total_steps, state_dim),
    )
    action = data.create_dataset(
        "action",
        data=np.zeros((total_steps, action_dim), dtype=np.float32),
        chunks=(total_steps, action_dim),
    )
    point_cloud = data.create_dataset(
        "point_cloud",
        data=np.zeros((total_steps, 8, pc_dim), dtype=np.float32),
        chunks=(total_steps, 8, pc_dim),
    )
    meta.create_dataset(
        "episode_ends",
        data=np.asarray([total_steps], dtype=np.int64),
        chunks=(1,),
    )
    root.attrs.update(
        {
            "export_status": "complete",
            "expected_total_frames": total_steps,
            "converted_frames": total_steps,
            "integrity": {
                "state": _sha256(state[:]),
                "action": _sha256(action[:]),
                "point_cloud": _sha256(point_cloud[:]),
            },
        }
    )


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def test_flexiv_dataset_reads_xyz_zarr(tmp_path: Path) -> None:
    zarr_path = tmp_path / "flexiv_xyz.zarr"
    _write_zarr(zarr_path, pc_dim=3)

    dataset = FlexivDualArmDataset(
        zarr_path=str(zarr_path),
        horizon=4,
        pad_before=1,
        pad_after=2,
        val_ratio=0.0,
        expected_num_points=8,
        expected_pointcloud_dim=3,
    )

    sample = dataset[0]
    assert sample["obs"]["agent_pos"].shape == (4, 28)
    assert sample["obs"]["point_cloud"].shape == (4, 8, 3)
    assert sample["action"].shape == (4, 14)


def test_flexiv_dataset_rejects_wrong_state_dim(tmp_path: Path) -> None:
    zarr_path = tmp_path / "bad_state.zarr"
    _write_zarr(zarr_path, state_dim=27)

    with pytest.raises(ValueError, match="data/state dim 27 != 28"):
        FlexivDualArmDataset(
            zarr_path=str(zarr_path),
            expected_num_points=8,
            expected_pointcloud_dim=3,
        )


def test_flexiv_dataset_rejects_incomplete_zarr(tmp_path: Path) -> None:
    zarr_path = tmp_path / "incomplete.zarr"
    _write_zarr(zarr_path)
    root = zarr.open(str(zarr_path), mode="a")
    root.attrs["export_status"] = "in_progress"

    with pytest.raises(ValueError, match="export is not complete"):
        FlexivDualArmDataset(
            zarr_path=str(zarr_path),
            expected_num_points=8,
            expected_pointcloud_dim=3,
        )


def test_flexiv_dataset_rejects_checksum_mismatch(tmp_path: Path) -> None:
    zarr_path = tmp_path / "corrupt.zarr"
    _write_zarr(zarr_path)
    root = zarr.open(str(zarr_path), mode="a")
    root["data"]["state"][0, 0] = 1.0

    with pytest.raises(ValueError, match="checksum mismatch"):
        FlexivDualArmDataset(
            zarr_path=str(zarr_path),
            expected_num_points=8,
            expected_pointcloud_dim=3,
        )
