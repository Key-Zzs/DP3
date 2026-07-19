#!/usr/bin/env python3
"""Offline DP3 policy-output stability analysis.

This tool deliberately stops at ``policy.predict_action``.  It never imports
Flexiv RDK, connects a camera, or sends an action.  Checkpoint construction,
the inference scheduler, the action-step slice, and the Flexiv normalizer
contract are shared with the formal inference helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DP3_ROOT = REPO_ROOT / "3D-Diffusion-Policy"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DP3_ROOT) not in sys.path:
    sys.path.insert(0, str(DP3_ROOT))

import torch  # noqa: E402
import zarr  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from diffusion_policy_3d.common.flexiv_state_contract import (  # noqa: E402
    FLEXIV_ACTION_DIM,
    FLEXIV_STATE_DIM,
    STATE_EE_POSITION_INDICES,
    STATE_EE_ROTATION_6D_INDICES,
    STATE_JOINT_INDICES,
    STATE_GRIPPER_INDICES,
    validate_flexiv_state_rotation6d,
)
from diffusion_policy_3d.real_world.flexiv_dual_arm_dp3 import (  # noqa: E402
    configure_policy_action_steps,
    configure_policy_inference_scheduler,
    load_dp3_policy_from_checkpoint,
    policy_contract_from_cfg,
    validate_flexiv_normalizer_contract,
    validate_policy_contract,
)


GROUPS = ("A", "B", "C", "D")
GROUP_DESCRIPTIONS = {
    "A": "fixed observation + random noise",
    "B": "fixed observation + fixed noise",
    "C": "varying static observations + fixed noise",
    "D": "varying static observations + random noise",
}
ACTION_CHANNELS = {
    "left_xyz": (0, 1, 2),
    "left_rotvec": (3, 4, 5),
    "right_xyz": (6, 7, 8),
    "right_rotvec": (9, 10, 11),
    "left_gripper": (12,),
    "right_gripper": (13,),
}
ACTION_NAMES = tuple(
    [
        "left_x",
        "left_y",
        "left_z",
        "left_rx",
        "left_ry",
        "left_rz",
        "right_x",
        "right_y",
        "right_z",
        "right_rx",
        "right_ry",
        "right_rz",
        "left_gripper",
        "right_gripper",
    ]
)


@dataclass(frozen=True)
class ZarrDataset:
    path: Path
    point_cloud: np.ndarray
    state: np.ndarray
    action: np.ndarray
    episode_ends: np.ndarray
    attrs: dict[str, Any]


@dataclass(frozen=True)
class MotionMetrics:
    action_xyz: np.ndarray
    action_rotvec: np.ndarray
    gripper_change: np.ndarray
    tcp_position_change: np.ndarray
    tcp_rotation_change: np.ndarray
    joint_change: np.ndarray


@dataclass(frozen=True)
class StaticWindow:
    episode_index: int
    start: int
    end: int
    eligible_count: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class MotionWindow:
    episode_index: int
    start: int
    end: int
    anchor_frame: int
    active_count: int
    active_fraction: float
    normalized_motion_score: float
    selection_reason: str

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ExperimentCase:
    group: str
    sample_index: int
    obs_frame_indices: tuple[int, int]
    episode_index: int
    seed: int


def _jsonable(value: Any) -> Any:
    """Convert Zarr/OmegaConf/Numpy values into JSON-safe primitives."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_jsonable(v) for v in value]
    return str(value)


def _cfg_select(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    value = cfg
    for key in dotted_key.split("."):
        if isinstance(value, Mapping):
            if key not in value:
                return default
            value = value[key]
        else:
            try:
                value = getattr(value, key)
            except AttributeError:
                return default
    return value


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, tuple):
        return list(value)
    return _jsonable(value)


def _sha256_file(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _zarr_metadata_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.name in {".zarray", ".zattrs", ".zgroup"}
    )
    for candidate in files:
        digest.update(str(candidate.relative_to(path)).encode("utf-8"))
        digest.update(candidate.read_bytes())
    return digest.hexdigest()


def load_zarr_dataset(path: str | Path) -> ZarrDataset:
    """Load and validate the exact DP3 replay-buffer contract."""

    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Zarr path does not exist: {dataset_path}")
    root = zarr.open(str(dataset_path), mode="r")
    required = ("data/point_cloud", "data/state", "data/action", "meta/episode_ends")
    missing = [name for name in required if name not in root]
    if missing:
        raise ValueError(f"Zarr is missing required arrays: {missing}")

    point_cloud = np.asarray(root["data/point_cloud"][:])
    state = np.asarray(root["data/state"][:])
    action = np.asarray(root["data/action"][:])
    episode_ends = np.asarray(root["meta/episode_ends"][:])
    attrs = _jsonable(dict(root.attrs))

    if point_cloud.ndim != 3 or point_cloud.shape[1:] != (2048, 3):
        raise ValueError(f"point_cloud must have shape [T, 2048, 3], got {point_cloud.shape}")
    if state.ndim != 2 or state.shape[1:] != (FLEXIV_STATE_DIM,):
        raise ValueError(f"state must have shape [T, 34], got {state.shape}")
    if action.ndim != 2 or action.shape[1:] != (FLEXIV_ACTION_DIM,):
        raise ValueError(f"action must have shape [T, 14], got {action.shape}")
    if not (point_cloud.shape[0] == state.shape[0] == action.shape[0]):
        raise ValueError("point_cloud/state/action must have the same frame count")
    if episode_ends.ndim != 1 or episode_ends.size == 0:
        raise ValueError("meta/episode_ends must be a non-empty 1D array")
    if episode_ends[-1] != state.shape[0] or np.any(np.diff(episode_ends) <= 0):
        raise ValueError("meta/episode_ends must be increasing and end at the frame count")
    for name, values in (
        ("point_cloud", point_cloud),
        ("state", state),
        ("action", action),
        ("episode_ends", episode_ends),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or Inf")

    for attr_name, expected in (
        ("num_points", 2048),
        ("pointcloud_dim", 3),
        ("state_dim", FLEXIV_STATE_DIM),
        ("action_dim", FLEXIV_ACTION_DIM),
    ):
        if attr_name in attrs and int(attrs[attr_name]) != expected:
            raise ValueError(f"Zarr attr {attr_name}={attrs[attr_name]!r}, expected {expected}")
    validate_flexiv_state_rotation6d(state, context="Zarr state")
    return ZarrDataset(
        path=dataset_path,
        point_cloud=point_cloud,
        state=state,
        action=action,
        episode_ends=episode_ends.astype(np.int64, copy=False),
        attrs=attrs,
    )


def _rot6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rotation_6d, dtype=np.float64)
    c0 = values[..., :3]
    c1 = values[..., 3:]
    c0 = c0 / np.linalg.norm(c0, axis=-1, keepdims=True)
    c1 = c1 - np.sum(c1 * c0, axis=-1, keepdims=True) * c0
    c1 = c1 / np.linalg.norm(c1, axis=-1, keepdims=True)
    c2 = np.cross(c0, c1)
    return np.stack((c0, c1, c2), axis=-1)


def _transition_values(values: np.ndarray, norm_axis: int | None = None) -> np.ndarray:
    result = np.zeros((values.shape[0],) + values.shape[1:-1], dtype=np.float64)
    delta = np.diff(values, axis=0)
    if norm_axis is None:
        result[1:] = np.max(np.abs(delta), axis=-1)
    else:
        result[1:] = np.linalg.norm(delta, axis=norm_axis)
    return result


def compute_motion_metrics(state: np.ndarray, action: np.ndarray) -> MotionMetrics:
    """Compute physical, per-frame static-window metrics for both arms."""

    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if state.shape[1:] != (34,) or action.shape[1:] != (14,):
        raise ValueError(f"unexpected state/action shapes: {state.shape}, {action.shape}")

    action_pose = action[:, :12].reshape(-1, 2, 6)
    action_xyz = np.linalg.norm(action_pose[:, :, :3], axis=-1)
    action_rotvec = np.linalg.norm(action_pose[:, :, 3:], axis=-1)
    gripper_change = np.zeros((len(action), 2), dtype=np.float64)
    gripper_change[1:] = np.abs(np.diff(action[:, 12:14], axis=0))

    tcp_position = np.stack((state[:, 7:10], state[:, 24:27]), axis=1)
    tcp_position_change = np.zeros((len(state), 2), dtype=np.float64)
    tcp_position_change[1:] = np.linalg.norm(np.diff(tcp_position, axis=0), axis=-1)

    rotations = np.stack(
        (_rot6d_to_matrix(state[:, 10:16]), _rot6d_to_matrix(state[:, 27:33])),
        axis=1,
    )
    relative = np.einsum("tski,tskj->tsij", rotations[:-1], rotations[1:])
    tcp_rotation_change = np.zeros((len(state), 2), dtype=np.float64)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    tcp_rotation_change[1:] = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))

    joint_change = np.zeros((len(state), 2), dtype=np.float64)
    joint_change[1:, 0] = np.max(np.abs(np.diff(state[:, :7], axis=0)), axis=-1)
    joint_change[1:, 1] = np.max(np.abs(np.diff(state[:, 17:24], axis=0)), axis=-1)
    return MotionMetrics(
        action_xyz=action_xyz,
        action_rotvec=action_rotvec,
        gripper_change=gripper_change,
        tcp_position_change=tcp_position_change,
        tcp_rotation_change=tcp_rotation_change,
        joint_change=joint_change,
    )


def _longest_true_run(flags: np.ndarray) -> tuple[int, int]:
    best_start = best_end = 0
    current_start: int | None = None
    for index, flag in enumerate(np.asarray(flags, dtype=bool)):
        if flag and current_start is None:
            current_start = index
        if (not flag or index == len(flags) - 1) and current_start is not None:
            current_end = index if not flag else index + 1
            if current_end - current_start > best_end - best_start:
                best_start, best_end = current_start, current_end
            current_start = None
    return best_start, best_end


def derive_joint_threshold(metrics: MotionMetrics) -> tuple[float, str]:
    """Derive a transparent joint threshold from the low-motion data scale."""

    low_quartile = float(np.percentile(np.max(metrics.joint_change, axis=1), 25.0))
    threshold = float(np.clip(low_quartile * 5.0, 0.001, 0.01))
    rationale = (
        "joint threshold was not supplied; it is 5x the lower quartile of the "
        "frame-wise max joint delta, clipped to [0.001, 0.01] rad to retain "
        "sensor-level motion while rejecting clear joint movement"
    )
    return threshold, rationale


def find_static_windows(
    dataset: ZarrDataset,
    metrics: MotionMetrics,
    *,
    action_xyz_threshold: float = 0.0002,
    action_rotation_threshold: float = 0.001745,
    gripper_change_threshold: float = 0.01,
    tcp_position_threshold: float = 0.0002,
    tcp_rotation_threshold: float = 0.001745,
    joint_change_threshold: float | None = None,
) -> tuple[list[StaticWindow], dict[str, Any]]:
    if joint_change_threshold is None:
        joint_change_threshold, joint_rationale = derive_joint_threshold(metrics)
    else:
        joint_change_threshold = float(joint_change_threshold)
        joint_rationale = "joint threshold supplied explicitly by CLI"
    thresholds = {
        "action_xyz_m": float(action_xyz_threshold),
        "action_rotvec_rad": float(action_rotation_threshold),
        "gripper_change": float(gripper_change_threshold),
        "tcp_position_change_m": float(tcp_position_threshold),
        "tcp_rotation_change_rad": float(tcp_rotation_threshold),
        "joint_change_rad": float(joint_change_threshold),
        "joint_threshold_rationale": joint_rationale,
        "reference": "CLI defaults are the objective's physical starting values; all final values are reported",
    }
    eligible = (
        (np.max(metrics.action_xyz, axis=1) <= action_xyz_threshold)
        & (np.max(metrics.action_rotvec, axis=1) <= action_rotation_threshold)
        & (np.max(metrics.gripper_change, axis=1) <= gripper_change_threshold)
        & (np.max(metrics.tcp_position_change, axis=1) <= tcp_position_threshold)
        & (np.max(metrics.tcp_rotation_change, axis=1) <= tcp_rotation_threshold)
        & (np.max(metrics.joint_change, axis=1) <= joint_change_threshold)
    )
    windows: list[StaticWindow] = []
    start = 0
    for episode_index, end_value in enumerate(dataset.episode_ends):
        end = int(end_value)
        local_start, local_end = _longest_true_run(eligible[start:end])
        # The first frame cannot have a same-episode history.  Make this
        # explicit even if a future metric implementation changes frame zero.
        local_start = max(local_start, 1)
        if local_end < local_start:
            local_end = local_start
        windows.append(
            StaticWindow(
                episode_index=episode_index,
                start=start + local_start,
                end=start + local_end,
                eligible_count=int(np.count_nonzero(eligible[start:end])),
            )
        )
        start = end
    return windows, thresholds


def select_static_window(
    windows: Sequence[StaticWindow],
    *,
    requested_samples: int,
    minimum_samples: int = 20,
) -> tuple[StaticWindow, int]:
    """Select one physical interval; never concatenate episodes."""

    candidates_100 = [window for window in windows if window.length >= requested_samples]
    candidates_min = [window for window in windows if window.length >= minimum_samples]
    candidates = candidates_100 or candidates_min
    if not candidates:
        longest = max(windows, key=lambda item: item.length, default=None)
        longest_text = "none" if longest is None else str(longest.length)
        raise RuntimeError(
            "No same-episode static observation interval has at least "
            f"{minimum_samples} target frames (longest={longest_text}); refusing to stitch episodes"
        )
    selected = max(candidates, key=lambda item: (item.length, -item.episode_index))
    count = min(int(requested_samples), selected.length)
    return selected, count


def find_grasp_motion_windows(
    dataset: ZarrDataset,
    metrics: MotionMetrics,
    *,
    requested_samples: int,
    action_xyz_threshold: float = 0.0002,
    action_rotation_threshold: float = 0.001745,
    gripper_change_threshold: float = 0.01,
) -> tuple[list[MotionWindow], dict[str, Any]]:
    """Find one same-episode motion window around each right-gripper close.

    The task's demonstrations start with an open right gripper (near 1) and
    close it toward 0 during grasping.  Anchoring on the largest negative
    command transition keeps this analysis on the grasp phase instead of
    selecting an unrelated high-speed return-to-start segment.
    """

    if requested_samples <= 0:
        raise ValueError("requested_samples must be positive")
    windows: list[MotionWindow] = []
    episode_start = 0
    for episode_index, end_value in enumerate(dataset.episode_ends):
        episode_end = int(end_value)
        first_target = episode_start + 1
        available = episode_end - first_target
        if available <= 0:
            episode_start = episode_end
            continue
        length = min(int(requested_samples), available)
        right_gripper = np.asarray(
            dataset.action[episode_start:episode_end, 13], dtype=np.float64
        )
        change = np.diff(right_gripper, prepend=right_gripper[0])
        local_anchor = int(np.argmin(change))
        anchor_frame = episode_start + local_anchor
        # Put 40% of the selected interval before the closing transition so
        # the window contains approach, close, and post-grasp motion.
        start = anchor_frame - int(round(0.4 * length))
        start = max(first_target, min(start, episode_end - length))
        end = start + length

        action_xyz = np.max(metrics.action_xyz[start:end], axis=1)
        action_rotation = np.max(metrics.action_rotvec[start:end], axis=1)
        gripper_change = np.max(metrics.gripper_change[start:end], axis=1)
        active = (
            (action_xyz > action_xyz_threshold)
            | (action_rotation > action_rotation_threshold)
            | (gripper_change > gripper_change_threshold)
        )
        normalized_energy = (
            action_xyz / action_xyz_threshold
            + action_rotation / action_rotation_threshold
            + gripper_change / gripper_change_threshold
        )
        windows.append(
            MotionWindow(
                episode_index=episode_index,
                start=start,
                end=end,
                anchor_frame=anchor_frame,
                active_count=int(np.count_nonzero(active)),
                active_fraction=float(np.mean(active)),
                normalized_motion_score=float(np.mean(normalized_energy)),
                selection_reason="largest negative right-gripper action transition",
            )
        )
        episode_start = episode_end
    metadata = {
        "mode": "grasp_motion",
        "anchor": "largest negative right-gripper action transition in each episode",
        "layout": "40% before anchor and 60% from anchor onward, clipped within one episode",
        "active_thresholds": {
            "action_xyz_m": float(action_xyz_threshold),
            "action_rotvec_rad": float(action_rotation_threshold),
            "gripper_change": float(gripper_change_threshold),
        },
    }
    return windows, metadata


def select_grasp_motion_window(
    windows: Sequence[MotionWindow],
    *,
    minimum_samples: int = 20,
) -> tuple[MotionWindow, int]:
    """Choose the grasp window with the most sustained physical activity."""

    candidates = [window for window in windows if window.length >= minimum_samples]
    if not candidates:
        longest = max(windows, key=lambda item: item.length, default=None)
        longest_text = "none" if longest is None else str(longest.length)
        raise RuntimeError(
            "No same-episode grasp-motion interval has at least "
            f"{minimum_samples} target frames (longest={longest_text})"
        )
    selected = max(
        candidates,
        key=lambda item: (
            item.active_fraction,
            item.normalized_motion_score,
            -item.episode_index,
        ),
    )
    return selected, selected.length


def build_experiment_plan(
    window: StaticWindow | MotionWindow,
    *,
    sample_count: int,
    seed_base: int,
    n_obs_steps: int = 2,
) -> list[ExperimentCase]:
    if n_obs_steps != 2:
        raise ValueError("This stability experiment requires exactly two observation history frames")
    targets = list(range(window.start, window.start + sample_count))
    if not targets:
        raise ValueError("sample_count must be positive")
    cases: list[ExperimentCase] = []
    for group in GROUPS:
        for sample_index, target in enumerate(targets):
            if group in {"A", "B"}:
                target_for_obs = targets[0]
            else:
                target_for_obs = target
            seed = seed_base + sample_index if group in {"A", "D"} else seed_base
            cases.append(
                ExperimentCase(
                    group=group,
                    sample_index=sample_index,
                    obs_frame_indices=(target_for_obs - 1, target_for_obs),
                    episode_index=window.episode_index,
                    seed=int(seed),
                )
            )
    return cases


def _episode_for_frame(episode_ends: np.ndarray, frame_index: int) -> int:
    return int(np.searchsorted(episode_ends, frame_index, side="right"))


def validate_observation_indices(
    obs_frame_indices: Sequence[int],
    *,
    episode_ends: np.ndarray,
    n_obs_steps: int = 2,
) -> int:
    if len(obs_frame_indices) != n_obs_steps:
        raise ValueError(f"observation history must have {n_obs_steps} frames")
    frames = [int(item) for item in obs_frame_indices]
    if any(frame < 0 for frame in frames) or any(b != a + 1 for a, b in zip(frames, frames[1:])):
        raise ValueError(f"observation history must be consecutive, got {frames}")
    episodes = {_episode_for_frame(episode_ends, frame) for frame in frames}
    if len(episodes) != 1:
        raise ValueError(f"observation history crosses an episode boundary: {frames}")
    return episodes.pop()


def _summary_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    return np.concatenate(
        (
            np.mean(values, axis=0),
            np.std(values, axis=0),
            np.min(values, axis=0),
            np.max(values, axis=0),
        )
    ).astype(np.float32)


def summarize_observation(dataset: ZarrDataset, frames: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    pointcloud_summary = np.stack(
        [_summary_vector(dataset.point_cloud[int(frame)]) for frame in frames]
    )
    state_summary = np.stack(
        [_summary_vector(dataset.state[int(frame)]) for frame in frames]
    )
    return pointcloud_summary, state_summary


def selected_tcp_pose_range(
    dataset: ZarrDataset,
    window: StaticWindow,
    sample_count: int,
    *,
    tcp_position_threshold: float,
    tcp_rotation_threshold: float,
) -> dict[str, Any]:
    """Check absolute TCP pose drift over every frame used by the histories."""

    frames = np.arange(window.start - 1, window.start + sample_count, dtype=np.int64)
    positions = np.stack(
        (dataset.state[frames, 7:10], dataset.state[frames, 24:27]), axis=1
    )
    position_range = np.max(positions, axis=0) - np.min(positions, axis=0)
    rotations = np.stack(
        (
            _rot6d_to_matrix(dataset.state[frames, 10:16]),
            _rot6d_to_matrix(dataset.state[frames, 27:33]),
        ),
        axis=1,
    )
    relative_rotation_range = np.zeros((2,), dtype=np.float64)
    for side in range(2):
        relative = np.einsum("ij,njk->nik", rotations[0, side].T, rotations[:, side])
        angle = np.arccos(
            np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
        )
        relative_rotation_range[side] = np.max(angle)
    max_position_range = float(np.max(position_range))
    max_rotation_range = float(np.max(relative_rotation_range))
    # A static interval can accumulate small same-direction transition noise;
    # two transition thresholds is a transparent, conservative range check.
    return {
        "frame_range_including_history": [int(frames[0]), int(frames[-1] + 1)],
        "position_range_m_by_side_xyz": position_range.tolist(),
        "relative_rotation_range_rad_by_side": relative_rotation_range.tolist(),
        "max_position_range_m": max_position_range,
        "max_relative_rotation_range_rad": max_rotation_range,
        "range_tolerance_multiple": 2.0,
        "position_threshold_m": float(tcp_position_threshold),
        "rotation_threshold_rad": float(tcp_rotation_threshold),
        "near_static": bool(
            max_position_range <= 2.0 * tcp_position_threshold
            and max_rotation_range <= 2.0 * tcp_rotation_threshold
        ),
    }


def _rng_context(device: torch.device, seed: int):
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    return torch.random.fork_rng(devices=devices, enabled=True), seed


def validate_policy_result(result: Mapping[str, Any], *, action_steps: int, action_dim: int, horizon: int) -> None:
    if "action" not in result or "action_pred" not in result:
        raise ValueError("policy.predict_action result must contain action and action_pred")
    action = result["action"]
    action_pred = result["action_pred"]
    if tuple(action.shape) != (1, action_steps, action_dim):
        raise ValueError(f"result['action'] shape {tuple(action.shape)} != (1, {action_steps}, {action_dim})")
    if tuple(action_pred.shape) != (1, horizon, action_dim):
        raise ValueError(f"result['action_pred'] shape {tuple(action_pred.shape)} != (1, {horizon}, {action_dim})")
    if not torch.isfinite(action).all() or not torch.isfinite(action_pred).all():
        raise ValueError("policy output contains NaN or Inf")


def run_policy_once(
    policy: Any,
    observation: Mapping[str, torch.Tensor],
    *,
    seed: int,
    device: torch.device,
    action_steps: int,
    action_dim: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run one isolated, seeded prediction without changing global RNG state."""

    started = time.perf_counter()
    with torch.random.fork_rng(
        devices=[device.index] if device.type == "cuda" and device.index is not None else [],
        enabled=True,
    ):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        with torch.inference_mode():
            result = policy.predict_action(observation)
        validate_policy_result(
            result,
            action_steps=action_steps,
            action_dim=action_dim,
            horizon=horizon,
        )
        action = result["action"][0].detach().cpu().numpy().astype(np.float32, copy=True)
        action_pred = result["action_pred"][0].detach().cpu().numpy().astype(np.float32, copy=True)
    return action, action_pred, time.perf_counter() - started


def _stats(values: np.ndarray) -> dict[str, float | None]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if flattened.size == 0:
        return {"mean": None, "std": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(flattened)),
        "std": float(np.std(flattened)),
        "p50": float(np.percentile(flattened, 50)),
        "p95": float(np.percentile(flattened, 95)),
        "max": float(np.max(flattened)),
    }


def _stats_by_channel(values: np.ndarray) -> dict[str, dict[str, float | None]]:
    array = np.asarray(values)
    return {
        name: _stats(array[..., indices])
        for name, indices in ACTION_CHANNELS.items()
    } | {
        name: _stats(array[..., index])
        for index, name in enumerate(ACTION_NAMES)
    }


def action_scale_stats(action: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(np.asarray(action, dtype=np.float64))
    return {
        "p50_abs": np.percentile(absolute, 50, axis=0).tolist(),
        "p95_abs": np.percentile(absolute, 95, axis=0).tolist(),
        "p99_abs": np.percentile(absolute, 99, axis=0).tolist(),
        "names": list(ACTION_NAMES),
    }


def _latency_stats(latencies: np.ndarray) -> dict[str, float]:
    values = np.asarray(latencies, dtype=np.float64) * 1000.0
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
    }


def _relative_l2(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        np.asarray(values, dtype=np.float64) - np.asarray(baseline, dtype=np.float64),
        axis=(-2, -1),
    )


def summarize_groups(
    records: list[dict[str, Any]],
    *,
    action_scale: dict[str, Any],
    observation_regime: str = "static",
) -> dict[str, Any]:
    by_group = {
        group: [record for record in records if record["group"] == group]
        for group in GROUPS
    }
    baseline = np.mean(
        np.stack([record["action_pred"] for record in by_group["B"]]).astype(np.float64),
        axis=0,
    )
    p95_scale = np.asarray(action_scale["p95_abs"], dtype=np.float64)
    valid_scale = p95_scale > 0.0
    output: dict[str, Any] = {}
    for group, group_records in by_group.items():
        actions = np.stack([record["action"] for record in group_records])
        predictions = np.stack([record["action_pred"] for record in group_records])
        latencies = np.asarray([record["policy_latency_sec"] for record in group_records])
        action_variance = np.var(actions.astype(np.float64), axis=0)
        horizon_variance = np.var(predictions.astype(np.float64), axis=0)
        normalized_std = np.full_like(horizon_variance, np.nan, dtype=np.float64)
        normalized_std[:, valid_scale] = (
            np.sqrt(horizon_variance[:, valid_scale]) / p95_scale[valid_scale][None, :]
        )
        l2 = _relative_l2(predictions, baseline)
        normalized_mean = (
            float(np.nanmean(normalized_std)) if np.any(np.isfinite(normalized_std)) else None
        )
        normalized_json = [
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in normalized_std.tolist()
        ]
        output[group] = {
            "description": (
                GROUP_DESCRIPTIONS[group]
                if observation_regime == "static"
                else GROUP_DESCRIPTIONS[group].replace(
                    "varying static observations", "varying grasp-motion observations"
                )
            ),
            "sample_count": len(group_records),
            "seeds": sorted({int(record["seed"]) for record in group_records}),
            "channel_stats_action": _stats_by_channel(actions),
            "channel_stats_action_pred": _stats_by_channel(predictions),
            "action_step_variance": action_variance.tolist(),
            "complete_horizon_variance": horizon_variance.tolist(),
            "normalized_horizon_std_by_dim": normalized_json,
            "normalized_horizon_std_mean": normalized_mean,
            "normalized_scale_valid_dimensions": [
                ACTION_NAMES[index] for index, is_valid in enumerate(valid_scale) if is_valid
            ],
            "normalized_scale_zero_dimensions": [
                ACTION_NAMES[index] for index, is_valid in enumerate(valid_scale) if not is_valid
            ],
            "raw_horizon_std_mean": float(np.mean(np.sqrt(horizon_variance))),
            "relative_fixed_baseline_l2": _stats(l2),
            "latency": _latency_stats(latencies),
        }
    output["baseline_group"] = "B mean action_pred"
    return output


def _jump_metrics(chunks: np.ndarray, *, boundary_stride: int = 1) -> dict[str, np.ndarray]:
    chunks = np.asarray(chunks)
    if boundary_stride <= 0:
        raise ValueError("boundary_stride must be positive")
    if chunks.ndim != 3 or chunks.shape[-1] != FLEXIV_ACTION_DIM:
        raise ValueError(
            f"chunks must have shape [N, K, {FLEXIV_ACTION_DIM}], got {chunks.shape}"
        )
    internal = chunks[:, 1:, :] - chunks[:, :-1, :]
    boundary = (
        chunks[boundary_stride:, 0, :] - chunks[:-boundary_stride, -1, :]
        if chunks.shape[0] > boundary_stride
        else np.empty((0, FLEXIV_ACTION_DIM), dtype=chunks.dtype)
    )

    def metrics(values: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "left_xyz": np.linalg.norm(values[:, :3], axis=-1),
            "left_rotvec": np.linalg.norm(values[:, 3:6], axis=-1),
            "right_xyz": np.linalg.norm(values[:, 6:9], axis=-1),
            "right_rotvec": np.linalg.norm(values[:, 9:12], axis=-1),
            "left_gripper": np.abs(values[:, 12]),
            "right_gripper": np.abs(values[:, 13]),
        }

    return {"internal": metrics(internal.reshape(-1, 14)), "boundary": metrics(boundary)}


def summarize_chunk_seams(
    records: list[dict[str, Any]],
    *,
    action_steps: int,
) -> dict[str, Any]:
    """Summarize command jumps using the deployment's synchronous chunk stride.

    Groups C/D advance their observation by one recorded frame per sample.  A
    synchronous K-action rollout therefore pairs sample ``i`` with ``i + K``;
    pairing adjacent samples would instead describe receding one-step control.
    Groups A/B deliberately keep one observation fixed, so adjacent independent
    predictions remain the appropriate isolation experiment for sampling noise.
    """

    if action_steps <= 0:
        raise ValueError("action_steps must be positive")
    output: dict[str, Any] = {}
    for group in GROUPS:
        group_records = sorted(
            [record for record in records if record["group"] == group],
            key=lambda item: item["sample_index"],
        )
        chunks = np.stack([record["action"] for record in group_records])
        boundary_stride = 1 if group in {"A", "B"} else action_steps
        raw = _jump_metrics(chunks, boundary_stride=boundary_stride)
        group_output: dict[str, Any] = {}
        for name in raw["internal"]:
            internal_stats = _stats(raw["internal"][name])
            boundary_stats = _stats(raw["boundary"][name])
            internal_p95 = internal_stats["p95"] or 0.0
            ratio = None if internal_p95 == 0.0 else float((boundary_stats["p95"] or 0.0) / internal_p95)
            group_output[name] = {
                "internal": internal_stats,
                "boundary": boundary_stats,
                "boundary_internal_p95_ratio": ratio,
            }
        group_output["pairing"] = {
            "boundary_stride_samples": boundary_stride,
            "boundary_pair_count": max(0, len(group_records) - boundary_stride),
            "observation_semantics": (
                "fixed observation; adjacent independent predictions isolate sampling noise"
                if group in {"A", "B"}
                else f"varying observations; sample i is paired with i+{action_steps} after executing the full chunk"
            ),
        }
        output[group] = group_output
    return output


def _temporal_metric(values: np.ndarray) -> dict[str, dict[str, float | None]]:
    return {
        name: _stats(
            np.linalg.norm(values[..., indices], axis=-1)
            if len(indices) > 1
            else np.abs(values[..., indices[0]])
        )
        for name, indices in ACTION_CHANNELS.items()
    }


def _chunk_overlap_contract(
    *,
    n_obs_steps: int,
    action_steps: int,
    horizon: int,
) -> dict[str, int]:
    if n_obs_steps <= 0 or action_steps <= 0 or horizon <= 0:
        raise ValueError("n_obs_steps, action_steps, and horizon must be positive")
    action_start = n_obs_steps - 1
    old_tail_start = action_start + action_steps
    overlap_steps = min(action_steps, max(0, horizon - old_tail_start))
    return {
        "chunk_stride_frames": action_steps,
        "action_start_index": action_start,
        "old_tail_start_index": old_tail_start,
        "new_head_start_index": action_start,
        "overlap_steps": overlap_steps,
    }


def summarize_temporal_alignment(
    records: list[dict[str, Any]],
    *,
    n_obs_steps: int,
    action_steps: int,
    horizon: int,
) -> dict[str, Any]:
    contract = _chunk_overlap_contract(
        n_obs_steps=n_obs_steps,
        action_steps=action_steps,
        horizon=horizon,
    )
    overlap = contract["overlap_steps"]
    output: dict[str, Any] = {}
    for group in ("C", "D"):
        group_records = sorted(
            [record for record in records if record["group"] == group],
            key=lambda item: item["sample_index"],
        )
        predictions = np.stack([record["action_pred"] for record in group_records])
        stride = contract["chunk_stride_frames"]
        if overlap == 0 or len(predictions) <= stride:
            aligned = np.empty((0, FLEXIV_ACTION_DIM), dtype=predictions.dtype)
            unaligned = np.empty((0, FLEXIV_ACTION_DIM), dtype=predictions.dtype)
        else:
            old_tail = predictions[:-stride, contract["old_tail_start_index"] : contract["old_tail_start_index"] + overlap, :]
            new_head = predictions[stride:, contract["new_head_start_index"] : contract["new_head_start_index"] + overlap, :]
            old_same_relative = predictions[:-stride, contract["new_head_start_index"] : contract["new_head_start_index"] + overlap, :]
            aligned = new_head - old_tail
            unaligned = new_head - old_same_relative
        aligned_metrics = _temporal_metric(aligned.reshape(-1, 14))
        unaligned_metrics = _temporal_metric(unaligned.reshape(-1, 14))
        ratios = {}
        for name in aligned_metrics:
            denominator = unaligned_metrics[name]["p95"] or 0.0
            ratios[name] = None if denominator == 0 else (aligned_metrics[name]["p95"] or 0.0) / denominator
        output[group] = {
            "contract": contract,
            "num_chunk_pairs": max(0, len(group_records) - stride),
            "aligned_future_delta_disagreement": aligned_metrics,
            "unaligned_same_relative_step_disagreement": unaligned_metrics,
            "aligned_to_unaligned_p95_ratio": ratios,
            "interpretation": (
                "Old action_pred tail and new executable head are aligned at the same future control times "
                "after one complete synchronous chunk. Values are delta-command disagreements, not absolute-pose errors."
            ),
        }
    return output


def summarize_temporal_ensemble(
    records: list[dict[str, Any]],
    *,
    dataset_action: np.ndarray,
    episode_ends: np.ndarray,
    n_obs_steps: int,
    action_steps: int,
    horizon: int,
    new_prediction_weights: Sequence[float] = (0.25, 0.5, 0.75),
) -> dict[str, Any]:
    """Simulate overlap blending without changing or executing the policy.

    The old prediction's unused horizon tail and the next prediction's
    executable head refer to the same future frames.  Candidate ensembles blend
    those aligned deltas with either fixed or deployment-style linearly ramped
    weights; the non-overlapping final action remains the new prediction.
    Ground-truth errors use the Zarr actions for the new chunk.
    """

    contract = _chunk_overlap_contract(
        n_obs_steps=n_obs_steps,
        action_steps=action_steps,
        horizon=horizon,
    )
    overlap = contract["overlap_steps"]
    stride = contract["chunk_stride_frames"]
    action_start = contract["action_start_index"]
    old_tail_start = contract["old_tail_start_index"]
    weights = tuple(float(value) for value in new_prediction_weights)
    if any(not 0.0 <= value <= 1.0 for value in weights):
        raise ValueError("new_prediction_weights must be within [0, 1]")

    output: dict[str, Any] = {}
    for group in ("C", "D"):
        group_records = sorted(
            [record for record in records if record["group"] == group],
            key=lambda item: item["sample_index"],
        )
        pairs: list[tuple[dict[str, Any], dict[str, Any], np.ndarray]] = []
        by_sample = {int(record["sample_index"]): record for record in group_records}
        for old_record in group_records:
            new_record = by_sample.get(int(old_record["sample_index"]) + stride)
            if new_record is None:
                continue
            target = int(new_record["obs_frame_indices"][-1])
            end = target + action_steps
            episode = _episode_for_frame(episode_ends, target)
            if end > int(episode_ends[episode]):
                continue
            pairs.append((old_record, new_record, np.asarray(dataset_action[target:end])))

        if not pairs:
            output[group] = {
                "contract": contract,
                "pair_count": 0,
                "baseline_new_prediction": None,
                "candidates": {},
            }
            continue

        old_predictions = np.stack([item[0]["action_pred"] for item in pairs])
        new_chunks = np.stack([item[1]["action"] for item in pairs])
        ground_truth = np.stack([item[2] for item in pairs])
        old_last = np.stack([item[0]["action"][-1] for item in pairs])

        def summarize_candidate(chunks: np.ndarray) -> dict[str, Any]:
            seam = chunks[:, 0, :] - old_last
            error = chunks - ground_truth
            return {
                "boundary_jump": _temporal_metric(seam),
                "zarr_action_error": _temporal_metric(error.reshape(-1, FLEXIV_ACTION_DIM)),
            }

        candidates: dict[str, Any] = {}
        for new_weight in weights:
            if overlap > 1:
                ramp_weights = np.linspace(new_weight, 1.0, overlap)
            elif overlap == 1:
                ramp_weights = np.asarray([new_weight])
            else:
                ramp_weights = np.empty((0,), dtype=np.float64)
            blended = new_chunks.copy()
            if overlap > 0:
                old_tail = old_predictions[:, old_tail_start : old_tail_start + overlap, :]
                blended[:, :overlap, :] = (
                    (1.0 - new_weight) * old_tail
                    + new_weight * blended[:, :overlap, :]
                )
            candidates[f"new_weight_{new_weight:g}"] = {
                "new_prediction_weight": new_weight,
                "old_prediction_weight": 1.0 - new_weight,
                "blend_scope": "all_action_channels",
                **summarize_candidate(blended),
            }
            pose_only = new_chunks.copy()
            if overlap > 0:
                pose_only[:, :overlap, :12] = (
                    (1.0 - new_weight) * old_tail[:, :, :12]
                    + new_weight * pose_only[:, :overlap, :12]
                )
            candidates[f"pose_only_new_weight_{new_weight:g}"] = {
                "new_prediction_weight": new_weight,
                "old_prediction_weight": 1.0 - new_weight,
                "blend_scope": "cartesian_pose_only_grippers_use_new_prediction",
                **summarize_candidate(pose_only),
            }
            ramp_pose_only = new_chunks.copy()
            if overlap > 0:
                ramp_pose_only[:, :overlap, :12] = (
                    (1.0 - ramp_weights[None, :, None]) * old_tail[:, :, :12]
                    + ramp_weights[None, :, None]
                    * ramp_pose_only[:, :overlap, :12]
                )
            candidates[f"pose_only_ramp_new_weight_{new_weight:g}"] = {
                "initial_new_prediction_weight": new_weight,
                "new_prediction_weights": ramp_weights.tolist(),
                "old_prediction_weights": (1.0 - ramp_weights).tolist(),
                "weight_mode": "linear_ramp",
                "blend_scope": "cartesian_pose_only_grippers_use_new_prediction",
                **summarize_candidate(ramp_pose_only),
            }

        output[group] = {
            "contract": contract,
            "pair_count": len(pairs),
            "baseline_new_prediction": summarize_candidate(new_chunks),
            "candidates": candidates,
            "interpretation": (
                "This is a deployment-aligned offline simulation. It does not modify the checkpoint or claim real-robot improvement."
            ),
        }
    return output


def _file_provenance(config_path: Path, checkpoint_path: Path, dataset: ZarrDataset) -> dict[str, Any]:
    return {
        "config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
            "size_bytes": config_path.stat().st_size,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "zarr": {
            "path": str(dataset.path),
            "metadata_sha256": _zarr_metadata_sha256(dataset.path),
            "shapes": {
                "point_cloud": list(dataset.point_cloud.shape),
                "state": list(dataset.state.shape),
                "action": list(dataset.action.shape),
                "episode_ends": list(dataset.episode_ends.shape),
            },
            "episode_ends": dataset.episode_ends.tolist(),
            "attrs": dataset.attrs,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "numpy": np.__version__,
        },
    }


def _load_and_configure_policy(config_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    inference_cfg = OmegaConf.load(str(config_path))
    use_ema = bool(_cfg_select(inference_cfg, "use_ema", True))
    policy, checkpoint_cfg, _workspace = load_dp3_policy_from_checkpoint(
        checkpoint_path,
        device,
        use_ema=use_ema,
    )
    contract = policy_contract_from_cfg(checkpoint_cfg)
    validate_policy_contract(contract)

    expected_target = _cfg_select(inference_cfg, "policy._target_")
    actual_target = _cfg_select(checkpoint_cfg, "policy._target_")
    if _plain(expected_target) != _plain(actual_target):
        raise ValueError(f"checkpoint policy target {actual_target!r} != inference config {expected_target!r}")
    for checkpoint_key, inference_key in (
        ("horizon", "policy.horizon"),
        ("n_obs_steps", "policy.n_obs_steps"),
        ("shape_meta.obs.point_cloud.shape", "shape_meta.obs.point_cloud.shape"),
        ("shape_meta.obs.agent_pos.shape", "shape_meta.obs.agent_pos.shape"),
        ("shape_meta.action.shape", "shape_meta.action.shape"),
    ):
        actual = _plain(_cfg_select(checkpoint_cfg, checkpoint_key))
        expected = _plain(_cfg_select(inference_cfg, inference_key))
        if actual != expected:
            raise ValueError(f"checkpoint {checkpoint_key}={actual!r} != inference config {expected!r}")

    n_action_steps = int(_cfg_select(inference_cfg, "algorithm_profiles.simple_dp3.n_action_steps", 4))
    max_action_steps = configure_policy_action_steps(
        policy,
        horizon=_cfg_select(checkpoint_cfg, "horizon"),
        n_obs_steps=_cfg_select(checkpoint_cfg, "n_obs_steps"),
        n_action_steps=n_action_steps,
    )
    if n_action_steps != 4:
        raise ValueError(f"objective requires n_action_steps=4, got {n_action_steps}")
    scheduler_name = str(_cfg_select(inference_cfg, "inference.scheduler", "ddim"))
    scheduler_class = configure_policy_inference_scheduler(
        policy,
        scheduler_name,
        clip_sample=bool(_cfg_select(inference_cfg, "policy.noise_scheduler.clip_sample", True)),
    )
    num_inference_steps = int(_cfg_select(inference_cfg, "inference.num_inference_steps", 10))
    if num_inference_steps != 10:
        raise ValueError(f"objective requires num_inference_steps=10, got {num_inference_steps}")
    policy.num_inference_steps = num_inference_steps

    normalizer_audit = validate_flexiv_normalizer_contract(
        policy,
        normalizer_schema=_cfg_select(checkpoint_cfg, "task.dataset.normalizer_schema"),
        state_schema=_cfg_select(checkpoint_cfg, "task.dataset.state_schema"),
        rotation6d_convention=_cfg_select(checkpoint_cfg, "task.dataset.rotation6d_convention"),
        action_rotation_representation=_cfg_select(checkpoint_cfg, "task.dataset.action_rotation_representation"),
        clip_actions_to_execution_limits=_cfg_select(checkpoint_cfg, "task.dataset.clip_actions_to_execution_limits"),
        action_xyz_limit=_cfg_select(checkpoint_cfg, "task.dataset.action_xyz_limit"),
        action_rotation_limit=_cfg_select(checkpoint_cfg, "task.dataset.action_rotation_limit"),
        state_joint_range_floor=_cfg_select(checkpoint_cfg, "task.dataset.state_joint_range_floor"),
        state_ee_position_range_floor=_cfg_select(checkpoint_cfg, "task.dataset.state_ee_position_range_floor"),
    )
    runtime_contract = {
        "use_ema": use_ema,
        "scheduler": scheduler_class,
        "scheduler_requested": scheduler_name,
        "num_inference_steps": num_inference_steps,
        "n_obs_steps": int(_cfg_select(checkpoint_cfg, "n_obs_steps")),
        "n_action_steps": n_action_steps,
        "horizon": int(_cfg_select(checkpoint_cfg, "horizon")),
        "max_action_steps": max_action_steps,
        "policy_class": type(policy).__name__,
        "policy_contract": _jsonable(contract.__dict__),
        "normalizer_audit": normalizer_audit,
    }
    return policy, checkpoint_cfg, runtime_contract


def _render_bar(value: float, maximum: float) -> str:
    width = 0 if maximum <= 0 else min(100, max(0, 100 * value / maximum))
    return f'<span class="bar"><span style="width:{width:.1f}%"></span></span>'


def write_plot_html(path: Path, summary: dict[str, Any]) -> None:
    sensitivities = {
        group: float(summary["groups"][group]["normalized_horizon_std_mean"])
        for group in GROUPS
    }
    maximum = max(sensitivities.values(), default=0.0)
    rows = "".join(
        f"<tr><td>Group {group}</td><td>{sensitivities[group]:.8g}</td><td>{_render_bar(sensitivities[group], maximum)}</td></tr>"
        for group in GROUPS
    )
    seam_rows = []
    for group in GROUPS:
        ratio = summary["chunk_seams"][group]["left_xyz"]["boundary_internal_p95_ratio"]
        pairing = summary["chunk_seams"][group]["pairing"]
        seam_rows.append(
            f"<tr><td>Group {group}</td><td>{pairing['boundary_stride_samples']}</td>"
            f"<td>{'n/a' if ratio is None else f'{ratio:.6g}'}</td></tr>"
        )
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DP3 policy stability</title>
<style>body{{font:14px sans-serif;margin:2rem;color:#222}}table{{border-collapse:collapse;margin:1rem 0}}th,td{{border:1px solid #ccc;padding:.4rem .7rem;text-align:left}}.bar{{display:inline-block;width:360px;background:#eee}}.bar span{{display:block;height:1rem;background:#4472c4}}</style>
</head><body><h1>Offline DP3 policy stability</h1>
<p>Normalized horizon standard deviation uses each Zarr action dimension's P95 absolute scale.</p>
<h2>Normalized sensitivity</h2><table><tr><th>Group</th><th>Mean normalized std</th><th></th></tr>{rows}</table>
<h2>Chunk boundary/internal P95 ratio (left xyz)</h2><table><tr><th>Group</th><th>Pair stride</th><th>Ratio</th></tr>{''.join(seam_rows)}</table>
<p>Full numerical arrays are in <code>samples.npz</code>; the Markdown report and JSON summary contain the complete channel-level statistics.</p>
</body></html>"""
    path.write_text(html_text, encoding="utf-8")


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    window_mode = summary.get("window_mode", "static")
    selected = summary.get("window_selection", summary.get("static_selection"))
    sensitivity = summary["sensitivity"]
    main_group = max(("A", "C"), key=lambda group: sensitivity[group]["normalized_horizon_std_mean"])
    a_value = sensitivity["A"]["normalized_horizon_std_mean"]
    c_value = sensitivity["C"]["normalized_horizon_std_mean"]
    if window_mode != "static":
        conclusion = (
            "This grasp-motion run evaluates cross-chunk continuity and action fidelity. "
            "Group C contains intentional task motion, so A-versus-C is not interpreted as "
            "sampling noise versus observation noise."
        )
    elif a_value > 2.0 * max(c_value, 1e-12) and a_value > 1e-6:
        conclusion = "Group A is substantially larger than Group C: diffusion sampling randomness is the primary measured source."
    elif c_value > 2.0 * max(a_value, 1e-12) and c_value > 1e-6:
        conclusion = "Group C is substantially larger than Group A: observation sensitivity is the primary measured source."
    elif a_value > 1e-6 and c_value > 1e-6:
        conclusion = "Groups A and C are both material: diffusion sampling and observation sensitivity both contribute."
    else:
        conclusion = "Groups A and C are both small in this offline replay; any remaining real-robot jitter is outside this experiment's coverage."
    deterministic = summary["determinism"]
    lines = [
        "# Offline DP3 policy stability report",
        "",
        "## Verdict",
        "",
        conclusion,
        (
            f"The largest normalized sensitivity among A/C is Group {main_group} ({_fmt(sensitivity[main_group]['normalized_horizon_std_mean'])})."
            if window_mode == "static"
            else "Use the deployment-aligned seam and Zarr-action-error tables below for the motion trade-off."
        ),
        f"Group B deterministic baseline: **{'PASS' if deterministic['pass'] else 'FAIL'}**; max absolute action_pred difference={_fmt(deterministic['max_abs_diff'])} (threshold 1e-6).",
        "",
        "## Safety and provenance",
        "",
        "This was fully offline. The tool did not import Flexiv RDK, connect a camera, connect a robot, or send actions. It used the shared checkpoint loader, action-step configurator, DDIM configurator, policy-contract validator, and Flexiv normalizer validator.",
        f"- Config: `{summary['provenance']['config']['path']}`",
        f"- Checkpoint: `{summary['provenance']['checkpoint']['path']}`",
        f"- Zarr: `{summary['provenance']['zarr']['path']}`",
        f"- Runtime contract: EMA={summary['runtime_contract']['use_ema']}, scheduler={summary['runtime_contract']['scheduler']}, steps={summary['runtime_contract']['num_inference_steps']}, n_obs_steps={summary['runtime_contract']['n_obs_steps']}, n_action_steps={summary['runtime_contract']['n_action_steps']}",
        "",
        f"## Zarr and {window_mode} window validation",
        "",
        f"Zarr arrays: point_cloud={summary['provenance']['zarr']['shapes']['point_cloud']}, state={summary['provenance']['zarr']['shapes']['state']}, action={summary['provenance']['zarr']['shapes']['action']}; episode_ends={summary['provenance']['zarr']['episode_ends']}.",
        f"Final physical thresholds: `{json.dumps(summary['thresholds'], ensure_ascii=False)}`.",
        "The objective's physical defaults were used for action/TCP/gripper/rotation criteria; the joint threshold was derived and its rationale is recorded above rather than hidden.",
        f"Selected absolute TCP pose range (including history): position max range={_fmt(summary['selected_tcp_pose_range']['max_position_range_m'])} m, relative rotation max range={_fmt(summary['selected_tcp_pose_range']['max_relative_rotation_range_rad'])} rad; near-static diagnostic={summary['selected_tcp_pose_range']['near_static']}.",
        "",
    ]
    if window_mode == "static":
        lines.extend(
            [
                "| Episode | Longest static target interval | Length | Eligible frames |",
                "|---:|---:|---:|---:|",
            ]
        )
        for item in summary["window_candidates"]:
            lines.append(
                f"| {item['episode_index']} | [{item['start']}, {item['end']}) | "
                f"{item['length']} | {item['eligible_count']} |"
            )
    else:
        lines.extend(
            [
                "The interval is anchored on the right gripper's strongest closing transition, then ranked by sustained motion. It is not selected from the static detector.",
                "",
                "| Episode | Grasp-motion interval | Anchor | Length | Active fraction | Motion score |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in summary["window_candidates"]:
            lines.append(
                f"| {item['episode_index']} | [{item['start']}, {item['end']}) | "
                f"{item['anchor_frame']} | {item['length']} | {_fmt(item['active_fraction'])} | "
                f"{_fmt(item['normalized_motion_score'])} |"
            )
    lines.extend(
        [
            "",
            f"Selected interval: episode {selected['episode_index']}, target frames [{selected['start']}, {selected['end']}) with {selected['sample_count']} samples. Every history is `[t-1, t]` within that episode; no episode was concatenated.",
            "",
            "## Four groups",
            "",
            "| Group | Samples | Normalized horizon std | Raw horizon std | Latency P50/P95 ms |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group in GROUPS:
        item = summary["groups"][group]
        lines.append(
            f"| {group} | {item['sample_count']} | {_fmt(item['normalized_horizon_std_mean'])} | {_fmt(item['raw_horizon_std_mean'])} | {_fmt(item['latency']['p50_ms'])}/{_fmt(item['latency']['p95_ms'])} |"
        )
    lines.extend(
        [
            "",
            (
                "Group A measures random initial diffusion noise at a fixed history; Group C measures replayed static-observation variation at fixed noise; Group D is the combined condition. These are normalized comparisons, not an assumption that variances add linearly."
                if window_mode == "static"
                else "Group A still isolates random diffusion initialization at one fixed grasp-phase history. Groups C/D follow the real moving trajectory; their variation includes intended task progress and must not be labeled observation noise."
            ),
            f"Normalized means exclude dimensions with zero Zarr P95 action scale: {', '.join(summary['groups']['A']['normalized_scale_zero_dimensions']) or 'none'}. Those dimensions remain present in the raw per-channel and horizon variance outputs.",
            "",
            "## Chunk seams",
            "",
            "Boundary/internal ratios below use P95 boundary jump divided by P95 internal jump; all jumps are differences between Cartesian delta commands. Groups C/D use the synchronous deployment stride `n_action_steps`; Groups A/B keep adjacent fixed-observation predictions to isolate sampling randomness.",
            "",
            "| Group | Pair stride | Pairs | Left xyz | Left rotvec | Right xyz | Right rotvec | Left gripper | Right gripper |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in GROUPS:
        seam = summary["chunk_seams"][group]
        values = [_fmt(seam[name]["boundary_internal_p95_ratio"]) for name in ACTION_CHANNELS]
        pairing = seam["pairing"]
        lines.append(
            f"| {group} | {pairing['boundary_stride_samples']} | {pairing['boundary_pair_count']} | "
            + " | ".join(values)
            + " |"
        )
    lines.extend(["", "## Temporal alignment", ""])
    for group in ("C", "D"):
        temporal = summary["temporal_alignment"][group]
        ratios = ", ".join(
            f"{name}={_fmt(value)}" for name, value in temporal["aligned_to_unaligned_p95_ratio"].items()
        )
        contract = temporal["contract"]
        lines.append(
            f"- Group {group}: {temporal['num_chunk_pairs']} deployment-aligned pairs at stride={contract['chunk_stride_frames']}; "
            f"old-tail/new-head overlap={contract['overlap_steps']} steps; aligned/unaligned P95 ratios: {ratios}."
        )
    lines.extend(
        [
            "",
            "A ratio below 1 means predictions agree more when the old unused horizon tail is aligned with the next executable head. Delta commands at different time bases must not be interpreted as absolute poses.",
            "",
            "## Deployment-aligned temporal ensemble simulation",
            "",
            "Each candidate blends only the overlapping old horizon tail and new chunk head. The final non-overlapping action remains the new prediction. Values below are P95; lower is better, but seam reduction must not be accepted if Zarr action error grows materially.",
            "",
            "| Group | Scope | New weight | Right xyz seam | Right rotvec seam | Right xyz action error | Right rotvec action error | Right gripper error |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in ("C", "D"):
        ensemble = summary["temporal_ensemble"][group]
        baseline = ensemble["baseline_new_prediction"]
        if baseline is None:
            continue
        lines.append(
            f"| {group} | baseline | 1 | {_fmt(baseline['boundary_jump']['right_xyz']['p95'])} | "
            f"{_fmt(baseline['boundary_jump']['right_rotvec']['p95'])} | "
            f"{_fmt(baseline['zarr_action_error']['right_xyz']['p95'])} | "
            f"{_fmt(baseline['zarr_action_error']['right_rotvec']['p95'])} | "
            f"{_fmt(baseline['zarr_action_error']['right_gripper']['p95'])} |"
        )
        for candidate in ensemble["candidates"].values():
            lines.append(
                f"| {group} | {candidate.get('blend_scope', 'all_action_channels')} | "
                f"{_fmt(candidate['new_prediction_weight'])} | "
                f"{_fmt(candidate['boundary_jump']['right_xyz']['p95'])} | "
                f"{_fmt(candidate['boundary_jump']['right_rotvec']['p95'])} | "
                f"{_fmt(candidate['zarr_action_error']['right_xyz']['p95'])} | "
                f"{_fmt(candidate['zarr_action_error']['right_rotvec']['p95'])} | "
                f"{_fmt(candidate['zarr_action_error']['right_gripper']['p95'])} |"
            )
    lines.extend(
        [
            "",
            "This simulation is evidence for or against an online ensemble implementation; it does not modify the checkpoint or execute a robot.",
            "",
            "## Scope limitation",
            "",
            "The experiment uses point clouds already generated in the Zarr. It cannot cover new online noise or ownership behavior from RealSense, Fast-FoundationStereo, owned buffers, or FPS sampling. If A and C are both small while historical hardware runs still jitter, inspect execution-chain timing, delta accumulation, coordinate mapping, chunk scheduling, and the controller.",
            "",
            "## Reproduction",
            "",
            f"```bash\ncd {REPO_ROOT}\nPYTHONNOUSERSITE=1 /home/deepcybo/miniconda3/bin/conda run -n dp3 python tools/analyze_dp3_policy_stability.py \\\n  --config {summary['provenance']['config']['path']} \\\n  --checkpoint {summary['provenance']['checkpoint']['path']} \\\n  --zarr {summary['provenance']['zarr']['path']} \\\n  --window-mode {window_mode} \\\n  --samples {summary['requested_samples']} \\\n  --seed-base {summary['seed_base']} \\\n  --output-dir {summary['output_dir']}\n```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "samples.npz",
        group=np.asarray([record["group"] for record in records]),
        sample_index=np.asarray([record["sample_index"] for record in records], dtype=np.int64),
        obs_frame_indices=np.asarray([record["obs_frame_indices"] for record in records], dtype=np.int64),
        episode_index=np.asarray([record["episode_index"] for record in records], dtype=np.int64),
        seed=np.asarray([record["seed"] for record in records], dtype=np.int64),
        policy_latency_sec=np.asarray([record["policy_latency_sec"] for record in records], dtype=np.float64),
        action=np.stack([record["action"] for record in records]),
        action_pred=np.stack([record["action_pred"] for record in records]),
        pointcloud_summary=np.stack([record["pointcloud_summary"] for record in records]),
        state_summary=np.stack([record["state_summary"] for record in records]),
    )
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row_index, record in enumerate(records):
            handle.write(
                json.dumps(
                    {
                        "npz_row": row_index,
                        "group": record["group"],
                        "sample_index": record["sample_index"],
                        "obs_frame_indices": record["obs_frame_indices"],
                        "episode_index": record["episode_index"],
                        "seed": record["seed"],
                        "policy_latency_sec": record["policy_latency_sec"],
                        "action_shape": list(record["action"].shape),
                        "action_pred_shape": list(record["action_pred"].shape),
                        "pointcloud_summary": record["pointcloud_summary"].tolist(),
                        "state_summary": record["state_summary"].tolist(),
                        "provenance": provenance,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if "groups" in summary and "chunk_seams" in summary:
        write_report(output_dir / "report.md", summary)
        write_plot_html(output_dir / "plots.html", summary)
    else:
        (output_dir / "report.md").write_text("# Test output\n", encoding="utf-8")
        (output_dir / "plots.html").write_text("<html><body>test</body></html>\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--window-mode",
        choices=("static", "grasp-motion"),
        default="static",
        help="analyze the longest static interval or a grasp-centered moving interval",
    )
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--action-xyz-threshold", type=float, default=0.0002)
    parser.add_argument("--action-rotation-threshold", type=float, default=0.001745)
    parser.add_argument("--gripper-change-threshold", type=float, default=0.01)
    parser.add_argument("--tcp-position-threshold", type=float, default=0.0002)
    parser.add_argument("--tcp-rotation-threshold", type=float, default=0.001745)
    parser.add_argument("--joint-change-threshold", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if "flexivrdk" in sys.modules:
        raise RuntimeError("Safety violation: Flexiv RDK is already imported")
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"config does not exist: {config_path}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint_path}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else REPO_ROOT / "logs" / f"dp3_policy_stability_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory is non-empty; refusing overwrite: {output_dir}")

    dataset = load_zarr_dataset(args.zarr)
    metrics = compute_motion_metrics(dataset.state, dataset.action)
    windows, thresholds = find_static_windows(
        dataset,
        metrics,
        action_xyz_threshold=args.action_xyz_threshold,
        action_rotation_threshold=args.action_rotation_threshold,
        gripper_change_threshold=args.gripper_change_threshold,
        tcp_position_threshold=args.tcp_position_threshold,
        tcp_rotation_threshold=args.tcp_rotation_threshold,
        joint_change_threshold=args.joint_change_threshold,
    )
    motion_windows: list[MotionWindow] = []
    motion_selection_metadata: dict[str, Any] | None = None
    if args.window_mode == "static":
        selected, sample_count = select_static_window(
            windows,
            requested_samples=args.samples,
            minimum_samples=20,
        )
        if selected.length < args.samples:
            thresholds["selection_note"] = (
                "No same-episode interval had the requested number of valid target frames; "
                "the longest interval meeting the 20-frame minimum was used."
            )
    else:
        motion_windows, motion_selection_metadata = find_grasp_motion_windows(
            dataset,
            metrics,
            requested_samples=args.samples,
            action_xyz_threshold=args.action_xyz_threshold,
            action_rotation_threshold=args.action_rotation_threshold,
            gripper_change_threshold=args.gripper_change_threshold,
        )
        selected, sample_count = select_grasp_motion_window(
            motion_windows,
            minimum_samples=20,
        )
    targets = list(range(selected.start, selected.start + sample_count))
    for target in targets:
        validate_observation_indices((target - 1, target), episode_ends=dataset.episode_ends)
    selected_pose_range = selected_tcp_pose_range(
        dataset,
        selected,
        sample_count,
        tcp_position_threshold=args.tcp_position_threshold,
        tcp_rotation_threshold=args.tcp_rotation_threshold,
    )
    if args.window_mode == "static" and not selected_pose_range["near_static"]:
        raise RuntimeError(
            "Selected static interval failed the absolute TCP pose range check: "
            f"{selected_pose_range}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"requested {device}, but CUDA is unavailable")
    policy, checkpoint_cfg, runtime_contract = _load_and_configure_policy(
        config_path, checkpoint_path, device
    )
    provenance = _file_provenance(config_path, checkpoint_path, dataset)
    provenance["runtime_contract"] = runtime_contract
    scale_stats = action_scale_stats(dataset.action)
    plan = build_experiment_plan(
        selected,
        sample_count=sample_count,
        seed_base=args.seed_base,
        n_obs_steps=runtime_contract["n_obs_steps"],
    )

    obs_cache: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    summary_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for case in plan:
        key = case.obs_frame_indices
        if key not in obs_cache:
            frames = list(key)
            obs_cache[key] = {
                "point_cloud": torch.from_numpy(dataset.point_cloud[frames]).unsqueeze(0).to(device),
                "agent_pos": torch.from_numpy(dataset.state[frames]).unsqueeze(0).to(device),
            }
            summary_cache[key] = summarize_observation(dataset, frames)

    records: list[dict[str, Any]] = []
    total = len(plan)
    for ordinal, case in enumerate(plan, start=1):
        action, action_pred, latency = run_policy_once(
            policy,
            obs_cache[case.obs_frame_indices],
            seed=case.seed,
            device=device,
            action_steps=runtime_contract["n_action_steps"],
            action_dim=FLEXIV_ACTION_DIM,
            horizon=runtime_contract["horizon"],
        )
        pointcloud_summary, state_summary = summary_cache[case.obs_frame_indices]
        records.append(
            {
                "group": case.group,
                "sample_index": case.sample_index,
                "obs_frame_indices": list(case.obs_frame_indices),
                "episode_index": case.episode_index,
                "seed": case.seed,
                "policy_latency_sec": latency,
                "action": action,
                "action_pred": action_pred,
                "pointcloud_summary": pointcloud_summary,
                "state_summary": state_summary,
            }
        )
        if ordinal == 1 or ordinal == total or ordinal % max(1, sample_count) == 0:
            print(f"[{ordinal}/{total}] completed group {case.group}", flush=True)

    groups_summary = summarize_groups(
        records,
        action_scale=scale_stats,
        observation_regime="static" if args.window_mode == "static" else "grasp-motion",
    )
    group_b_predictions = np.stack([record["action_pred"] for record in records if record["group"] == "B"])
    b_max_diff = float(np.max(np.abs(group_b_predictions - group_b_predictions[0])))
    determinism = {
        "max_abs_diff": b_max_diff,
        "threshold": 1e-6,
        "pass": bool(b_max_diff <= 1e-6),
        "method": "torch.random.fork_rng with torch.manual_seed and torch.cuda.manual_seed_all before every prediction",
    }
    sensitivity = {
        group: {
            "normalized_horizon_std_mean": groups_summary[group]["normalized_horizon_std_mean"],
            "raw_horizon_std_mean": groups_summary[group]["raw_horizon_std_mean"],
        }
        for group in ("A", "C", "D")
    }
    summary: dict[str, Any] = {
        "output_dir": str(output_dir),
        "requested_samples": int(args.samples),
        "actual_samples_per_group": int(sample_count),
        "window_mode": args.window_mode,
        "seed_base": int(args.seed_base),
        "runtime_contract": runtime_contract,
        "provenance": provenance,
        "thresholds": thresholds,
        "static_windows": [window.__dict__ | {"length": window.length} for window in windows],
        "window_candidates": [
            window.__dict__ | {"length": window.length}
            for window in (windows if args.window_mode == "static" else motion_windows)
        ],
        "window_selection": selected.__dict__ | {"length": selected.length, "sample_count": sample_count},
        "window_selection_metadata": motion_selection_metadata,
        "selected_tcp_pose_range": selected_pose_range,
        "action_scale_stats": scale_stats,
        "groups": groups_summary,
        "sensitivity": sensitivity,
        "determinism": determinism,
        "chunk_seams": summarize_chunk_seams(
            records,
            action_steps=runtime_contract["n_action_steps"],
        ),
        "temporal_alignment": summarize_temporal_alignment(
            records,
            n_obs_steps=runtime_contract["n_obs_steps"],
            action_steps=runtime_contract["n_action_steps"],
            horizon=runtime_contract["horizon"],
        ),
        "temporal_ensemble": summarize_temporal_ensemble(
            records,
            dataset_action=dataset.action,
            episode_ends=dataset.episode_ends,
            n_obs_steps=runtime_contract["n_obs_steps"],
            action_steps=runtime_contract["n_action_steps"],
            horizon=runtime_contract["horizon"],
        ),
        "safety": {
            "offline_only": True,
            "flexivrdk_imported": "flexivrdk" in sys.modules,
            "robot_connect_called": False,
            "camera_connect_called": False,
            "send_action_called": False,
        },
    }
    if args.window_mode == "static":
        summary["static_selection"] = summary["window_selection"]
    else:
        summary["motion_windows"] = summary["window_candidates"]
        summary["motion_selection"] = summary["window_selection"]
    save_outputs(output_dir, records, summary, provenance)
    print(json.dumps({"output_dir": str(output_dir), "summary": str(output_dir / 'summary.json'), "report": str(output_dir / 'report.md')}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
