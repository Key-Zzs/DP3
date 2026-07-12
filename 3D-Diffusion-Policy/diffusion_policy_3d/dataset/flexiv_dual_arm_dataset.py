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
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer


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
    ):
        super().__init__()
        self.task_name = task_name
        self.zarr_path = str(zarr_path)
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
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][..., :],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        data = {
            "obs": {
                "point_cloud": sample["point_cloud"].astype(np.float32, copy=False),
                "agent_pos": sample["state"].astype(np.float32, copy=False),
            },
            "action": sample["action"].astype(np.float32, copy=False),
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)


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
