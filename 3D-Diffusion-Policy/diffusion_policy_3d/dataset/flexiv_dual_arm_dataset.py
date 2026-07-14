from typing import Any, Dict
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
import zarr

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


FLEXIV_NORMALIZER_SCHEMA = "flexiv_physical_v1"
FLEXIV_STATE_DIM = 28
FLEXIV_ACTION_DIM = 14
_LEFT_ACTION_XYZ = slice(0, 3)
_LEFT_ACTION_ROTATION = slice(3, 6)
_RIGHT_ACTION_XYZ = slice(6, 9)
_RIGHT_ACTION_ROTATION = slice(9, 12)
_ACTION_GRIPPERS = slice(12, 14)
_STATE_JOINT_INDICES = np.asarray([*range(0, 7), *range(14, 21)], dtype=np.int64)
_STATE_EE_POSITION_INDICES = np.asarray([*range(7, 10), *range(21, 24)], dtype=np.int64)
_STATE_EE_ROTATION_INDICES = np.asarray([*range(10, 13), *range(24, 27)], dtype=np.int64)
_STATE_GRIPPER_INDICES = np.asarray([13, 27], dtype=np.int64)


class FlexivDualArmDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
        expected_state_dim=28,
        expected_action_dim=14,
        expected_pointcloud_dim=None,
        expected_num_points=1024,
        normalizer_schema=FLEXIV_NORMALIZER_SCHEMA,
        clip_actions_to_execution_limits=True,
        action_xyz_limit=0.02,
        action_rotation_limit=0.04,
        state_joint_range_floor=0.20,
        state_ee_position_range_floor=0.10,
        state_ee_rotation_range_floor=0.20,
        normalizer_quantile_low=0.005,
        normalizer_quantile_high=0.995,
    ):
        super().__init__()
        self.task_name = task_name
        self.zarr_path = str(zarr_path)
        if normalizer_schema != FLEXIV_NORMALIZER_SCHEMA:
            raise ValueError(
                f"normalizer_schema must be {FLEXIV_NORMALIZER_SCHEMA!r}, "
                f"got {normalizer_schema!r}"
            )
        self.normalizer_schema = normalizer_schema
        self.clip_actions_to_execution_limits = _require_bool(
            clip_actions_to_execution_limits,
            name="clip_actions_to_execution_limits",
        )
        self.action_xyz_limit = _positive_float(action_xyz_limit, name="action_xyz_limit")
        self.action_rotation_limit = _positive_float(
            action_rotation_limit,
            name="action_rotation_limit",
        )
        self.state_joint_range_floor = _positive_float(
            state_joint_range_floor,
            name="state_joint_range_floor",
        )
        self.state_ee_position_range_floor = _positive_float(
            state_ee_position_range_floor,
            name="state_ee_position_range_floor",
        )
        self.state_ee_rotation_range_floor = _positive_float(
            state_ee_rotation_range_floor,
            name="state_ee_rotation_range_floor",
        )
        self.normalizer_quantile_low, self.normalizer_quantile_high = _quantile_bounds(
            normalizer_quantile_low,
            normalizer_quantile_high,
        )
        self.schema = _validate_zarr_schema(
            zarr_path=zarr_path,
            expected_state_dim=expected_state_dim,
            expected_action_dim=expected_action_dim,
            expected_pointcloud_dim=expected_pointcloud_dim,
            expected_num_points=expected_num_points,
        )
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=["state", "action", "point_cloud"]
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        if mode != "limits":
            raise ValueError(
                f"Flexiv physical normalizer only supports mode='limits', got {mode!r}"
            )
        action_raw = np.asarray(self.replay_buffer["action"], dtype=np.float32)
        action = self._preprocess_action(action_raw)
        agent_pos = np.asarray(self.replay_buffer["state"], dtype=np.float32)

        normalizer = LinearNormalizer()
        normalizer["action"] = _make_action_normalizer(
            action,
            xyz_limit=self.action_xyz_limit,
            rotation_limit=self.action_rotation_limit,
        )
        normalizer["agent_pos"] = _make_state_normalizer(
            agent_pos,
            joint_range_floor=self.state_joint_range_floor,
            ee_position_range_floor=self.state_ee_position_range_floor,
            ee_rotation_range_floor=self.state_ee_rotation_range_floor,
            quantile_low=self.normalizer_quantile_low,
            quantile_high=self.normalizer_quantile_high,
        )
        normalizer["point_cloud"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["point_cloud"],
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        _audit_normalizer(
            normalizer,
            action_raw=action_raw,
            action_applied=action,
            action_xyz_limit=self.action_xyz_limit,
            action_rotation_limit=self.action_rotation_limit,
            state_joint_range_floor=self.state_joint_range_floor,
            state_ee_position_range_floor=self.state_ee_position_range_floor,
            state_ee_rotation_range_floor=self.state_ee_rotation_range_floor,
        )
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        data = {
            "obs": {
                "point_cloud": sample["point_cloud"].astype(np.float32, copy=False),
                "agent_pos": sample["state"].astype(np.float32, copy=False),
            },
            "action": self._preprocess_action(sample["action"]),
        }
        return data

    def _preprocess_action(self, action: Any) -> np.ndarray:
        values = np.asarray(action, dtype=np.float32)
        if values.shape[-1] != FLEXIV_ACTION_DIM:
            raise ValueError(
                f"Flexiv action last dimension must be {FLEXIV_ACTION_DIM}, "
                f"got {values.shape}"
            )
        out = values.copy()
        if self.clip_actions_to_execution_limits:
            for action_slice, limit in (
                (_LEFT_ACTION_XYZ, self.action_xyz_limit),
                (_RIGHT_ACTION_XYZ, self.action_xyz_limit),
                (_LEFT_ACTION_ROTATION, self.action_rotation_limit),
                (_RIGHT_ACTION_ROTATION, self.action_rotation_limit),
            ):
                out[..., action_slice] = _clip_vector_norm(out[..., action_slice], limit)
        out[..., _ACTION_GRIPPERS] = np.clip(out[..., _ACTION_GRIPPERS], 0.0, 1.0)
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)


def _make_action_normalizer(
    action: np.ndarray,
    *,
    xyz_limit: float,
    rotation_limit: float,
) -> SingleFieldLinearNormalizer:
    scale = np.ones(FLEXIV_ACTION_DIM, dtype=np.float32)
    offset = np.zeros(FLEXIV_ACTION_DIM, dtype=np.float32)
    scale[[0, 1, 2, 6, 7, 8]] = 1.0 / float(xyz_limit)
    scale[[3, 4, 5, 9, 10, 11]] = 1.0 / float(rotation_limit)
    scale[12:14] = 2.0
    offset[12:14] = -1.0
    return _manual_normalizer(action, scale=scale, offset=offset)


def _make_state_normalizer(
    state: np.ndarray,
    *,
    joint_range_floor: float,
    ee_position_range_floor: float,
    ee_rotation_range_floor: float,
    quantile_low: float,
    quantile_high: float,
) -> SingleFieldLinearNormalizer:
    values = np.asarray(state, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != FLEXIV_STATE_DIM:
        raise ValueError(
            f"Flexiv state must have shape (T, {FLEXIV_STATE_DIM}), got {values.shape}"
        )
    lower = np.quantile(values, quantile_low, axis=0).astype(np.float32)
    upper = np.quantile(values, quantile_high, axis=0).astype(np.float32)
    center = (lower + upper) * 0.5
    effective_range = upper - lower
    effective_range[_STATE_JOINT_INDICES] = np.maximum(
        effective_range[_STATE_JOINT_INDICES],
        joint_range_floor,
    )
    effective_range[_STATE_EE_POSITION_INDICES] = np.maximum(
        effective_range[_STATE_EE_POSITION_INDICES],
        ee_position_range_floor,
    )
    effective_range[_STATE_EE_ROTATION_INDICES] = np.maximum(
        effective_range[_STATE_EE_ROTATION_INDICES],
        ee_rotation_range_floor,
    )
    center[_STATE_GRIPPER_INDICES] = 0.5
    effective_range[_STATE_GRIPPER_INDICES] = 1.0
    scale = 2.0 / effective_range
    offset = -center * scale
    return _manual_normalizer(
        values,
        scale=scale.astype(np.float32),
        offset=offset.astype(np.float32),
    )


def _manual_normalizer(
    data: np.ndarray,
    *,
    scale: np.ndarray,
    offset: np.ndarray,
) -> SingleFieldLinearNormalizer:
    values = np.asarray(data, dtype=np.float32).reshape(-1, scale.shape[0])
    input_stats = {
        "min": values.min(axis=0).astype(np.float32),
        "max": values.max(axis=0).astype(np.float32),
        "mean": values.mean(axis=0).astype(np.float32),
        "std": values.std(axis=0).astype(np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=np.asarray(scale, dtype=np.float32),
        offset=np.asarray(offset, dtype=np.float32),
        input_stats_dict=input_stats,
    )


def _clip_vector_norm(values: np.ndarray, limit: float) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    factors = np.minimum(1.0, float(limit) / np.maximum(norms, 1e-12))
    return values * factors


def _audit_normalizer(
    normalizer: LinearNormalizer,
    *,
    action_raw: np.ndarray,
    action_applied: np.ndarray,
    action_xyz_limit: float,
    action_rotation_limit: float,
    state_joint_range_floor: float,
    state_ee_position_range_floor: float,
    state_ee_rotation_range_floor: float,
) -> None:
    action_scale = normalizer["action"].params_dict["scale"].detach().cpu().numpy()
    state_scale = normalizer["agent_pos"].params_dict["scale"].detach().cpu().numpy()
    expected_action_scale = np.asarray(
        [
            *([1.0 / action_xyz_limit] * 3),
            *([1.0 / action_rotation_limit] * 3),
            *([1.0 / action_xyz_limit] * 3),
            *([1.0 / action_rotation_limit] * 3),
            2.0,
            2.0,
        ],
        dtype=np.float32,
    )
    if not np.allclose(action_scale, expected_action_scale, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Flexiv action normalizer scale does not match its physical limits")
    maximum_state_scales = np.full(FLEXIV_STATE_DIM, np.inf, dtype=np.float32)
    maximum_state_scales[_STATE_JOINT_INDICES] = 2.0 / state_joint_range_floor
    maximum_state_scales[_STATE_EE_POSITION_INDICES] = 2.0 / state_ee_position_range_floor
    maximum_state_scales[_STATE_EE_ROTATION_INDICES] = 2.0 / state_ee_rotation_range_floor
    maximum_state_scales[_STATE_GRIPPER_INDICES] = 2.0
    if np.any(state_scale > maximum_state_scales + 1e-5):
        indices = np.flatnonzero(state_scale > maximum_state_scales + 1e-5).tolist()
        raise RuntimeError(f"Flexiv state normalizer exceeded range-floor scale at indices {indices}")
    changed = np.any(~np.isclose(action_raw, action_applied, rtol=0.0, atol=1e-8), axis=1)
    print(
        "[FlexivNormalizer] "
        f"schema={FLEXIV_NORMALIZER_SCHEMA} "
        f"action_xyz_limit={action_xyz_limit:g}m "
        f"action_rotation_limit={action_rotation_limit:g}rad "
        f"execution_clipped_frames={int(np.count_nonzero(changed))}/{action_raw.shape[0]} "
        f"state_range_floors=(joint={state_joint_range_floor:g}rad,"
        f"xyz={state_ee_position_range_floor:g}m,"
        f"rotation={state_ee_rotation_range_floor:g}rad) "
        f"max_state_scale={float(np.max(state_scale)):.6g}"
    )


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive float")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite positive float") from None
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive float")
    return result


def _require_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _quantile_bounds(low: Any, high: Any) -> tuple[float, float]:
    try:
        low_f = float(low)
        high_f = float(high)
    except (TypeError, ValueError):
        raise ValueError("normalizer quantiles must be finite floats") from None
    if not np.isfinite(low_f) or not np.isfinite(high_f) or not 0.0 <= low_f < high_f <= 1.0:
        raise ValueError(
            "normalizer quantiles must satisfy 0 <= low < high <= 1, "
            f"got low={low!r}, high={high!r}"
        )
    return low_f, high_f

def _validate_zarr_schema(
    zarr_path: str | Path,
    *,
    expected_state_dim: int | None,
    expected_action_dim: int | None,
    expected_pointcloud_dim: int | None,
    expected_num_points: int | None,
) -> dict[str, Any]:
    path = Path(zarr_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    root = zarr.open(str(path), mode="r")
    export_status = root.attrs.get("export_status")
    if export_status != "complete":
        raise ValueError(
            f"Zarr export is not complete: export_status={export_status!r}; "
            "re-export the dataset before training"
        )
    if "data" not in root:
        raise KeyError("Missing zarr group: data")
    if "meta" not in root:
        raise KeyError("Missing zarr group: meta")

    data = root["data"]
    meta = root["meta"]
    for key in ("state", "action", "point_cloud"):
        if key not in data:
            raise KeyError(f"Missing zarr array: data/{key}")
    if "episode_ends" not in meta:
        raise KeyError("Missing zarr array: meta/episode_ends")

    state = data["state"]
    action = data["action"]
    point_cloud = data["point_cloud"]
    episode_ends = meta["episode_ends"][:]

    if state.ndim != 2:
        raise ValueError(f"data/state must be T x D, got {state.shape}")
    if action.ndim != 2:
        raise ValueError(f"data/action must be T x D, got {action.shape}")
    if point_cloud.ndim != 3:
        raise ValueError(f"data/point_cloud must be T x N x C, got {point_cloud.shape}")

    total_steps = int(state.shape[0])
    expected_total_frames = int(root.attrs.get("expected_total_frames", -1))
    converted_frames = int(root.attrs.get("converted_frames", -1))
    if expected_total_frames != total_steps or converted_frames != total_steps:
        raise ValueError(
            "Zarr completion metadata does not match data length: "
            f"T={total_steps}, expected_total_frames={expected_total_frames}, "
            f"converted_frames={converted_frames}"
        )
    if int(action.shape[0]) != total_steps:
        raise ValueError(
            f"data/action length {action.shape[0]} does not match state length {total_steps}"
        )
    if int(point_cloud.shape[0]) != total_steps:
        raise ValueError(
            "data/point_cloud length "
            f"{point_cloud.shape[0]} does not match state length {total_steps}"
        )

    if episode_ends.ndim != 1 or episode_ends.shape[0] == 0:
        raise ValueError(f"meta/episode_ends must be a non-empty vector, got {episode_ends.shape}")
    if not np.all(np.diff(episode_ends) > 0):
        raise ValueError("meta/episode_ends must be strictly increasing")
    if int(episode_ends[-1]) != total_steps:
        raise ValueError(
            f"episode_ends[-1] ({episode_ends[-1]}) does not equal T ({total_steps})"
        )

    if expected_state_dim is not None and int(state.shape[1]) != int(expected_state_dim):
        raise ValueError(f"data/state dim {state.shape[1]} != {expected_state_dim}")
    if expected_action_dim is not None and int(action.shape[1]) != int(expected_action_dim):
        raise ValueError(f"data/action dim {action.shape[1]} != {expected_action_dim}")
    pointcloud_dim = int(point_cloud.shape[2])
    if expected_pointcloud_dim is not None:
        if pointcloud_dim != int(expected_pointcloud_dim):
            raise ValueError(
                f"data/point_cloud dim {pointcloud_dim} != {expected_pointcloud_dim}"
            )
    elif pointcloud_dim not in (3, 6):
        raise ValueError(f"data/point_cloud dim must be 3 or 6, got {pointcloud_dim}")
    if expected_num_points is not None and int(point_cloud.shape[1]) != int(expected_num_points):
        raise ValueError(f"data/point_cloud points {point_cloud.shape[1]} != {expected_num_points}")

    integrity = root.attrs.get("integrity")
    if not isinstance(integrity, dict):
        raise ValueError("Zarr export is missing integrity checksums")
    for key, array in (
        ("state", state),
        ("action", action),
        ("point_cloud", point_cloud),
    ):
        expected_hash = integrity.get(key)
        if not isinstance(expected_hash, str):
            raise ValueError(f"Zarr export is missing data/{key} checksum")
        actual_hash = _zarr_array_sha256(array)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Zarr data/{key} checksum mismatch: "
                f"actual={actual_hash}, expected={expected_hash}"
            )

    return {
        "state_shape": tuple(state.shape),
        "action_shape": tuple(action.shape),
        "point_cloud_shape": tuple(point_cloud.shape),
        "episode_ends": episode_ends.tolist(),
    }


def _zarr_array_sha256(array: Any) -> str:
    hasher = hashlib.sha256()
    rows_per_chunk = int(array.chunks[0]) if array.chunks else 128
    for start in range(0, int(array.shape[0]), rows_per_chunk):
        chunk = np.ascontiguousarray(array[start : start + rows_per_chunk])
        if np.issubdtype(chunk.dtype, np.floating) and not np.isfinite(chunk).all():
            raise ValueError(f"Zarr array contains NaN or Inf near frame {start}")
        hasher.update(chunk.tobytes())
    return hasher.hexdigest()
