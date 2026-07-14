from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))

zarr = pytest.importorskip("zarr")

from diffusion_policy_3d.dataset.flexiv_dual_arm_dataset import (  # noqa: E402
    FLEXIV_NORMALIZER_SCHEMA,
    FlexivDualArmDataset,
)


def _write_zarr(
    path: Path,
    *,
    state_dim: int = 28,
    action_dim: int = 14,
    pc_dim: int = 3,
    state_values: np.ndarray | None = None,
    action_values: np.ndarray | None = None,
) -> None:
    root = zarr.group(str(path))
    data = root.create_group("data")
    meta = root.create_group("meta")
    total_steps = 5 if state_values is None and action_values is None else int(
        state_values.shape[0] if state_values is not None else action_values.shape[0]
    )
    state_data = (
        np.zeros((total_steps, state_dim), dtype=np.float32)
        if state_values is None
        else np.asarray(state_values, dtype=np.float32)
    )
    action_data = (
        np.zeros((total_steps, action_dim), dtype=np.float32)
        if action_values is None
        else np.asarray(action_values, dtype=np.float32)
    )
    state = data.create_dataset(
        "state",
        data=state_data,
        chunks=(total_steps, state_dim),
    )
    action = data.create_dataset(
        "action",
        data=action_data,
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


def test_flexiv_dataset_clips_raw_teleop_action_to_execution_contract(tmp_path: Path) -> None:
    zarr_path = tmp_path / "action_clip.zarr"
    actions = np.zeros((5, 14), dtype=np.float32)
    actions[:, 12:14] = 1.0
    actions[0, 0:3] = [0.03, 0.04, 0.0]
    actions[0, 3:6] = [0.0, 0.0, 0.08]
    actions[0, 6:9] = [0.0, -0.03, 0.04]
    actions[0, 9:12] = [0.06, 0.0, 0.08]
    actions[0, 12:14] = [-0.5, 1.5]
    _write_zarr(zarr_path, action_values=actions)

    dataset = FlexivDualArmDataset(
        zarr_path=str(zarr_path),
        horizon=1,
        expected_num_points=8,
        expected_pointcloud_dim=3,
    )

    applied = dataset[0]["action"][0].numpy()
    assert applied[0:3] == pytest.approx([0.012, 0.016, 0.0])
    assert applied[3:6] == pytest.approx([0.0, 0.0, 0.04])
    assert applied[6:9] == pytest.approx([0.0, -0.012, 0.016])
    assert applied[9:12] == pytest.approx([0.024, 0.0, 0.032])
    assert applied[12:14] == pytest.approx([0.0, 1.0])


def test_flexiv_physical_normalizer_is_symmetric_and_range_floored(tmp_path: Path) -> None:
    zarr_path = tmp_path / "physical_normalizer.zarr"
    states = np.zeros((5, 28), dtype=np.float32)
    states[:, 13] = np.linspace(0.0, 1.0, 5)
    states[:, 27] = 1.0
    actions = np.zeros((5, 14), dtype=np.float32)
    actions[:, 12] = np.linspace(0.0, 1.0, 5)
    actions[:, 13] = 1.0
    _write_zarr(zarr_path, state_values=states, action_values=actions)
    dataset = FlexivDualArmDataset(
        zarr_path=str(zarr_path),
        expected_num_points=8,
        expected_pointcloud_dim=3,
        normalizer_schema=FLEXIV_NORMALIZER_SCHEMA,
    )

    normalizer = dataset.get_normalizer()
    action_scale = normalizer["action"].params_dict["scale"].detach().cpu().numpy()
    action_offset = normalizer["action"].params_dict["offset"].detach().cpu().numpy()
    state_scale = normalizer["agent_pos"].params_dict["scale"].detach().cpu().numpy()

    assert action_scale[0:3] == pytest.approx([50.0, 50.0, 50.0])
    assert action_scale[6:9] == pytest.approx([50.0, 50.0, 50.0])
    assert action_scale[3:6] == pytest.approx([25.0, 25.0, 25.0])
    assert action_scale[9:12] == pytest.approx([25.0, 25.0, 25.0])
    assert action_scale[12:14] == pytest.approx([2.0, 2.0])
    assert action_offset == pytest.approx([*([0.0] * 12), -1.0, -1.0])
    assert state_scale[[0, 14]] == pytest.approx([10.0, 10.0])
    assert state_scale[[7, 21]] == pytest.approx([20.0, 20.0])
    assert state_scale[[10, 24]] == pytest.approx([10.0, 10.0])
    assert state_scale[[13, 27]] == pytest.approx([2.0, 2.0])


def test_flexiv_physical_normalizer_roundtrips_applied_action(tmp_path: Path) -> None:
    zarr_path = tmp_path / "normalizer_roundtrip.zarr"
    actions = np.zeros((5, 14), dtype=np.float32)
    actions[:, 12:14] = 1.0
    actions[0, :3] = [0.03, 0.04, 0.0]
    _write_zarr(zarr_path, action_values=actions)
    dataset = FlexivDualArmDataset(
        zarr_path=str(zarr_path),
        expected_num_points=8,
        expected_pointcloud_dim=3,
    )

    applied = dataset[0]["action"].numpy()
    normalizer = dataset.get_normalizer()["action"]
    restored = normalizer.unnormalize(normalizer.normalize(applied)).numpy()

    assert restored == pytest.approx(applied, abs=1e-7)
