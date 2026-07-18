from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))

zarr = pytest.importorskip("zarr")

from diffusion_policy_3d.common.flexiv_state_contract import (  # noqa: E402
    FLEXIV_RAW_FORCE_DROPPED_STATE_NAMES,
    FLEXIV_RAW_FORCE_STATE_NAMES,
    FLEXIV_RAW_FORCE_STATE_SCHEMA,
    FLEXIV_RAW_FORCE_TO_V2_TRANSFORM,
    flexiv_action_names,
    flexiv_state_names,
)
from diffusion_policy_3d.dataset.flexiv_dual_arm_dataset import (  # noqa: E402
    FLEXIV_ACTION_DIM,
    FLEXIV_NORMALIZER_SCHEMA,
    FLEXIV_STATE_DIM,
    FlexivDualArmDataset,
)


def _write_zarr(
    path: Path,
    *,
    state_dim: int = FLEXIV_STATE_DIM,
    action_dim: int = FLEXIV_ACTION_DIM,
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
    state_data = np.zeros((total_steps, state_dim), dtype=np.float32)
    if state_values is not None:
        state_data = np.asarray(state_values, dtype=np.float32)
    if state_values is None and state_dim == FLEXIV_STATE_DIM:
        state_data[:, 10:16] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        state_data[:, 27:33] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        state_data[:, 16] = 0.5
        state_data[:, 33] = 0.5
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
                "raw_source_state": _sha256(state_data),
                "derived_state": _sha256(state_data),
            },
            "state_schema": "flexiv_abs_rot6d_v2",
            "state_dim": FLEXIV_STATE_DIM,
            "action_dim": FLEXIV_ACTION_DIM,
            "state_names": flexiv_state_names(),
            "action_names": flexiv_action_names(),
            "state_rotation_representation": "rotation_6d",
            "state_rotation_reference": "absolute_rdk_world_base",
            "rotation6d_convention": "matrix_columns_0_1",
            "rotation6d_order": ["c0x", "c0y", "c0z", "c1x", "c1y", "c1z"],
            "action_rotation_representation": "rotvec",
            "source_state_schema": "flexiv_abs_rot6d_v2",
            "source_state_names": flexiv_state_names(),
            "state_transform": "passthrough_v2",
            "source_fps": 30,
            "raw_source_state_sha256": _sha256(state_data),
            "derived_state_sha256": _sha256(state_data),
            "exported_state_sha256": _sha256(state_data),
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
    assert sample["obs"]["agent_pos"].shape == (4, FLEXIV_STATE_DIM)
    assert sample["obs"]["point_cloud"].shape == (4, 8, 3)
    assert sample["action"].shape == (4, 14)


def test_flexiv_dataset_consumes_v3_provenance_but_normalizer_stays_34d(tmp_path: Path) -> None:
    zarr_path = tmp_path / "flexiv_v3_source.zarr"
    _write_zarr(zarr_path, pc_dim=3)
    root = zarr.open(str(zarr_path), mode="a")
    target_state = np.asarray(root["data/state"][:], dtype=np.float32)
    raw_state = np.concatenate(
        [target_state, np.full((target_state.shape[0], 14), 1e6, dtype=np.float32)],
        axis=1,
    )
    raw_hash = _sha256(raw_state)
    root.attrs.update(
        {
            "source_state_schema": FLEXIV_RAW_FORCE_STATE_SCHEMA,
            "source_state_dim": 48,
            "source_state_names": list(FLEXIV_RAW_FORCE_STATE_NAMES),
            "state_transform": FLEXIV_RAW_FORCE_TO_V2_TRANSFORM,
            "dropped_state_names": list(FLEXIV_RAW_FORCE_DROPPED_STATE_NAMES),
            "raw_source_state_sha256": raw_hash,
            "integrity": {
                **root.attrs["integrity"],
                "raw_source_state": raw_hash,
            },
        }
    )
    dataset = FlexivDualArmDataset(
        zarr_path=str(zarr_path),
        horizon=1,
        expected_num_points=8,
        expected_pointcloud_dim=3,
    )
    normalizer = dataset.get_normalizer()
    assert tuple(normalizer["agent_pos"].params_dict["scale"].shape) == (34,)


def test_flexiv_dataset_rejects_wrong_state_dim(tmp_path: Path) -> None:
    zarr_path = tmp_path / "bad_state.zarr"
    _write_zarr(zarr_path, state_dim=27)

    with pytest.raises(ValueError, match="data/state dim 27 != 34"):
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


def test_flexiv_dataset_rejects_v1_zarr_contract(tmp_path: Path) -> None:
    zarr_path = tmp_path / "legacy.zarr"
    _write_zarr(zarr_path)
    root = zarr.open(str(zarr_path), mode="a")
    root.attrs["state_schema"] = "flexiv_physical_v1"
    with pytest.raises(ValueError, match="v2 contract mismatch for state_schema"):
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


def test_flexiv_rot6d_normalizer_is_symmetric_and_range_floored(tmp_path: Path) -> None:
    zarr_path = tmp_path / "physical_normalizer.zarr"
    states = np.zeros((5, FLEXIV_STATE_DIM), dtype=np.float32)
    states[:, 10:16] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    states[:, 27:33] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    states[:, 16] = np.linspace(0.0, 1.0, 5)
    states[:, 33] = 1.0
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
    assert state_scale[[0, 17]] == pytest.approx([10.0, 10.0])
    assert state_scale[[7, 24]] == pytest.approx([20.0, 20.0])
    assert state_scale[np.r_[10:16, 27:33]] == pytest.approx(np.ones(12))
    assert state_scale[[16, 33]] == pytest.approx([2.0, 2.0])


def test_flexiv_rot6d_normalizer_roundtrips_applied_action(tmp_path: Path) -> None:
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
