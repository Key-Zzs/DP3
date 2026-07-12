"""Flexiv dual-arm DP3 inference utilities.

This module keeps the field-order contract and safety filtering independent
from the live robot loop so they can be tested without hardware.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

AXES = ("x", "y", "z", "rx", "ry", "rz")
SIDES = ("left", "right")

STATE_FIELD_NAMES = tuple(
    [f"left_joint_{idx}.pos" for idx in range(1, 8)]
    + [f"left_ee_pose.{axis}" for axis in AXES]
    + ["left_gripper_state_norm"]
    + [f"right_joint_{idx}.pos" for idx in range(1, 8)]
    + [f"right_ee_pose.{axis}" for axis in AXES]
    + ["right_gripper_state_norm"]
)

ACTION_FIELD_NAMES = tuple(
    [f"left_delta_ee_pose.{axis}" for axis in AXES]
    + [f"right_delta_ee_pose.{axis}" for axis in AXES]
    + ["left_gripper_cmd", "right_gripper_cmd"]
)


@dataclass(frozen=True)
class PolicyContract:
    n_obs_steps: int
    state_dim: int
    action_dim: int
    pointcloud_points: int
    pointcloud_dim: int


@dataclass(frozen=True)
class SafetyLimits:
    max_cartesian_delta: float | None = 0.01
    max_rotation_delta: float | None = 0.02
    low_speed_scale: float = 0.1
    min_gripper: float = 0.0
    max_gripper: float = 1.0


def configure_policy_inference_scheduler(
    policy: Any,
    scheduler_name: str,
    *,
    clip_sample: bool | None = None,
) -> str:
    """Select an inference-only scheduler without changing the checkpoint contract."""

    name = str(scheduler_name).strip().lower()
    if name == "checkpoint":
        if clip_sample is not None and bool(
            getattr(policy.noise_scheduler.config, "clip_sample", False)
        ) != bool(clip_sample):
            policy.noise_scheduler = type(policy.noise_scheduler).from_config(
                policy.noise_scheduler.config,
                clip_sample=bool(clip_sample),
            )
            if hasattr(policy, "noise_scheduler_pc"):
                policy.noise_scheduler_pc = copy.deepcopy(policy.noise_scheduler)
        return type(policy.noise_scheduler).__name__
    if name != "ddim":
        raise ValueError(
            f"Unsupported inference scheduler {scheduler_name!r}; "
            "expected 'checkpoint' or 'ddim'"
        )

    from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # noqa: PLC0415

    scheduler_overrides = {}
    if clip_sample is not None:
        scheduler_overrides["clip_sample"] = bool(clip_sample)
    scheduler = DDIMScheduler.from_config(
        policy.noise_scheduler.config,
        **scheduler_overrides,
    )
    policy.noise_scheduler = scheduler
    if hasattr(policy, "noise_scheduler_pc"):
        policy.noise_scheduler_pc = copy.deepcopy(scheduler)
    return type(scheduler).__name__


def configure_policy_action_steps(
    policy: Any,
    *,
    horizon: Any,
    n_obs_steps: Any,
    n_action_steps: Any,
) -> int:
    """Apply the inference rollout length allowed by the DP3 action slice.

    DP3 trains the complete ``horizon`` trajectory. ``n_action_steps`` is only
    used by ``predict_action()`` to slice executable actions starting at
    ``n_obs_steps - 1``, so it may differ from the training value within this
    bound.

    Returns the maximum legal rollout length.
    """

    resolved_horizon = _positive_int(horizon, label="horizon")
    resolved_n_obs_steps = _positive_int(n_obs_steps, label="n_obs_steps")
    resolved_n_action_steps = _positive_int(
        n_action_steps,
        label="n_action_steps",
    )
    max_action_steps = resolved_horizon - resolved_n_obs_steps + 1
    if max_action_steps <= 0:
        raise ValueError(
            "n_obs_steps must be no greater than horizon; "
            f"got n_obs_steps={resolved_n_obs_steps}, horizon={resolved_horizon}"
        )
    if resolved_n_action_steps > max_action_steps:
        raise ValueError(
            "n_action_steps must satisfy 1 <= n_action_steps <= "
            "horizon - n_obs_steps + 1; "
            f"got {resolved_n_action_steps} > {max_action_steps} "
            f"for horizon={resolved_horizon}, n_obs_steps={resolved_n_obs_steps}"
        )
    policy.n_action_steps = resolved_n_action_steps
    return max_action_steps


def policy_contract_from_cfg(cfg: Any) -> PolicyContract:
    pointcloud_shape = _cfg_get(cfg, "shape_meta.obs.point_cloud.shape")
    state_shape = _cfg_get(cfg, "shape_meta.obs.agent_pos.shape")
    action_shape = _cfg_get(cfg, "shape_meta.action.shape")
    return PolicyContract(
        n_obs_steps=_positive_int(_cfg_get(cfg, "n_obs_steps"), label="n_obs_steps"),
        state_dim=_positive_int(state_shape[0], label="state_dim"),
        action_dim=_positive_int(action_shape[0], label="action_dim"),
        pointcloud_points=_positive_int(pointcloud_shape[0], label="pointcloud_points"),
        pointcloud_dim=_positive_int(pointcloud_shape[1], label="pointcloud_dim"),
    )


def validate_policy_contract(contract: PolicyContract) -> None:
    """Fail early if a checkpoint is not compatible with the Flexiv DP3 runtime."""

    n_obs_steps = _positive_int(contract.n_obs_steps, label="n_obs_steps")
    state_dim = _positive_int(contract.state_dim, label="state_dim")
    action_dim = _positive_int(contract.action_dim, label="action_dim")
    pointcloud_points = _positive_int(contract.pointcloud_points, label="pointcloud_points")
    pointcloud_dim = _positive_int(contract.pointcloud_dim, label="pointcloud_dim")

    if n_obs_steps <= 0:
        raise ValueError(f"n_obs_steps must be positive, got {contract.n_obs_steps}")
    if state_dim != len(STATE_FIELD_NAMES):
        raise ValueError(
            f"Flexiv checkpoint state_dim must be {len(STATE_FIELD_NAMES)}, "
            f"got {contract.state_dim}"
        )
    if action_dim != len(ACTION_FIELD_NAMES):
        raise ValueError(
            f"Flexiv checkpoint action_dim must be {len(ACTION_FIELD_NAMES)}, "
            f"got {contract.action_dim}"
        )
    if pointcloud_points <= 0:
        raise ValueError(
            f"pointcloud_points must be positive, got {contract.pointcloud_points}"
        )
    if pointcloud_dim not in (3, 6):
        raise ValueError(
            f"Flexiv checkpoint pointcloud_dim must be 3 or 6, got {contract.pointcloud_dim}"
        )


def load_dp3_policy_from_checkpoint(
    ckpt_path: str | Path,
    device: str | torch.device,
    *,
    use_ema: bool | None = None,
):
    """Load a trained DP3 workspace checkpoint and return the inference policy.

    The checkpoint is first mapped to CPU so a smoke test can run on machines
    without the original CUDA device. The selected policy is then moved to
    ``device``.
    """

    import dill  # noqa: PLC0415
    from train import TrainDP3Workspace  # noqa: PLC0415

    ckpt_path = Path(ckpt_path).expanduser()
    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
    workspace = TrainDP3Workspace(payload["cfg"])
    workspace.load_payload(payload, exclude_keys=("optimizer",))
    if use_ema is None:
        use_ema = bool(_cfg_get(workspace.cfg, "training.use_ema", default=False))
    if use_ema and workspace.ema_model is None:
        raise ValueError(
            "use_ema=true, but this checkpoint does not contain an EMA model; "
            "set use_ema=false or select a checkpoint trained with EMA enabled"
        )
    policy = workspace.ema_model if use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    return policy, workspace.cfg, workspace


def build_agent_pos(
    observation: Mapping[str, Any],
    *,
    default_gripper_state: float | None = None,
) -> np.ndarray:
    """Build the 28D Flexiv state vector used by training."""

    values = []
    for key in STATE_FIELD_NAMES:
        if key in observation:
            values.append(_as_float(observation[key], key=key))
            continue
        if key.endswith("_gripper_state_norm") and default_gripper_state is not None:
            values.append(float(default_gripper_state))
            continue
        raise KeyError(f"Missing observation field required by DP3 state: {key}")
    return np.asarray(values, dtype=np.float32)


def validate_agent_pos(
    agent_pos: Any,
    *,
    expected_dim: int = len(STATE_FIELD_NAMES),
) -> np.ndarray:
    """Validate the live low-dimensional observation before policy inference."""

    values = np.asarray(agent_pos, dtype=np.float32).reshape(-1)
    if values.shape != (int(expected_dim),):
        raise ValueError(f"agent_pos shape {values.shape} != ({int(expected_dim)},)")
    if not np.isfinite(values).all():
        raise ValueError("agent_pos contains NaN or Inf")
    for index, key in ((13, "left_gripper_state_norm"), (27, "right_gripper_state_norm")):
        if index < values.shape[0] and not 0.0 <= float(values[index]) <= 1.0:
            raise ValueError(f"{key}={float(values[index]):.6g} outside [0, 1]")
    return values


def build_pointcloud_frame_from_observation(
    observation: Mapping[str, Any],
    *,
    camera_name: str = "head_rgb",
    rgb_key: str | None = None,
    depth_key: str | None = None,
) -> dict[str, Any]:
    """Extract a PointCloudBuilder live frame from a Flexiv observation dict."""

    base_name = _camera_base_name(camera_name)
    resolved_rgb_key = rgb_key or _first_present_key(
        observation,
        (camera_name, f"{base_name}_rgb", f"{base_name}_image", base_name),
    )
    resolved_depth_key = depth_key or _first_present_key(
        observation,
        (f"sidecar.{base_name}_depth", f"sidecar.{camera_name}_depth"),
    )
    if resolved_depth_key is None:
        raise KeyError(
            "Missing depth frame. Enable `save_depth_sidecar` on the Flexiv robot "
            f"config and expected one of sidecar.{base_name}_depth or sidecar.{camera_name}_depth."
        )

    frame: dict[str, Any] = {"depth": observation[resolved_depth_key]}
    if resolved_rgb_key is not None:
        frame["rgb"] = observation[resolved_rgb_key]
    timestamp_key = f"{base_name}_rgbd_timestamp"
    if timestamp_key in observation:
        frame["timestamp"] = _as_float(observation[timestamp_key], key=timestamp_key)
    wall_time_key = f"{base_name}_rgbd_wall_time"
    if wall_time_key in observation:
        frame["wall_time"] = _as_float(observation[wall_time_key], key=wall_time_key)
    if "global_frame_index" in observation:
        frame["global_frame_index"] = _as_non_negative_int(
            observation["global_frame_index"],
            key="global_frame_index",
        )
    return frame


def prepare_point_cloud(
    point_cloud: Any,
    *,
    expected_num_points: int,
    expected_dim: int,
) -> np.ndarray:
    """Convert a builder point cloud to the checkpoint's expected shape."""

    if isinstance(point_cloud, torch.Tensor):
        pc = point_cloud.detach().cpu().numpy()
    else:
        pc = np.asarray(point_cloud)
    if pc.ndim != 2:
        raise ValueError(f"point_cloud must be N x C, got shape {pc.shape}")
    if int(pc.shape[0]) != int(expected_num_points):
        raise ValueError(
            f"point_cloud has {pc.shape[0]} points, expected {expected_num_points}"
        )
    if not np.isfinite(pc).all():
        raise ValueError("point_cloud contains NaN or Inf")
    channels = int(pc.shape[1])
    if channels == int(expected_dim):
        return pc.astype(np.float32, copy=False)
    if channels == 6 and int(expected_dim) == 3:
        return pc[:, :3].astype(np.float32, copy=False)
    raise ValueError(f"point_cloud has {channels} channels, expected {expected_dim}")


def history_to_policy_obs(
    agent_pos_history: Sequence[np.ndarray],
    point_cloud_history: Sequence[np.ndarray],
    *,
    n_obs_steps: int,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """Pad recent observations and build a DP3 policy input dict."""

    if not agent_pos_history or not point_cloud_history:
        raise ValueError("Observation history is empty")
    if len(agent_pos_history) != len(point_cloud_history):
        raise ValueError(
            f"agent_pos history length {len(agent_pos_history)} does not match "
            f"point_cloud history length {len(point_cloud_history)}"
        )

    agents = _tail_with_left_padding(agent_pos_history, n_obs_steps)
    pcs = _tail_with_left_padding(point_cloud_history, n_obs_steps)
    agent_batch = np.stack(agents, axis=0).astype(np.float32, copy=False)
    pc_batch = np.stack(pcs, axis=0).astype(np.float32, copy=False)
    target_device = torch.device(device)
    return {
        "agent_pos": torch.from_numpy(agent_batch).unsqueeze(0).to(target_device),
        "point_cloud": torch.from_numpy(pc_batch).unsqueeze(0).to(target_device),
    }


def filter_action_vector(
    action: Any,
    limits: SafetyLimits,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply finite checks, low-speed scaling, delta clipping, and gripper clipping."""

    action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_vec.shape != (14,):
        raise ValueError(f"DP3 Flexiv action must have shape (14,), got {action_vec.shape}")
    if not np.isfinite(action_vec).all():
        raise ValueError("DP3 action contains NaN or Inf")
    low_speed_scale = _finite_float(limits.low_speed_scale, label="low_speed_scale")
    max_cartesian_delta = _optional_positive_float(
        limits.max_cartesian_delta,
        label="max_cartesian_delta",
    )
    max_rotation_delta = _optional_positive_float(
        limits.max_rotation_delta,
        label="max_rotation_delta",
    )
    min_gripper = _finite_float(limits.min_gripper, label="min_gripper")
    max_gripper = _finite_float(limits.max_gripper, label="max_gripper")
    if not 0.0 <= low_speed_scale <= 1.0:
        raise ValueError("low_speed_scale must be in [0, 1]")
    if not 0.0 <= min_gripper <= 1.0:
        raise ValueError("min_gripper must be in [0, 1]")
    if not 0.0 <= max_gripper <= 1.0:
        raise ValueError("max_gripper must be in [0, 1]")
    if max_gripper < min_gripper:
        raise ValueError("max_gripper must be >= min_gripper")
    safe = action_vec.copy()
    safe[:12] *= low_speed_scale
    diagnostics: dict[str, Any] = {
        "low_speed_scale": low_speed_scale,
        "max_cartesian_delta": max_cartesian_delta,
        "max_rotation_delta": max_rotation_delta,
        "min_gripper": min_gripper,
        "max_gripper": max_gripper,
    }

    for side, xyz_slice, rot_slice in (
        ("left", slice(0, 3), slice(3, 6)),
        ("right", slice(6, 9), slice(9, 12)),
    ):
        safe[xyz_slice], xyz_diag = _clip_vector_norm(safe[xyz_slice], max_cartesian_delta)
        safe[rot_slice], rot_diag = _clip_vector_norm(safe[rot_slice], max_rotation_delta)
        diagnostics[f"{side}_xyz_norm"] = xyz_diag["input_norm"]
        diagnostics[f"{side}_xyz_clipped"] = xyz_diag["clipped"]
        diagnostics[f"{side}_rot_norm"] = rot_diag["input_norm"]
        diagnostics[f"{side}_rot_clipped"] = rot_diag["clipped"]

    safe[12] = np.clip(safe[12], min_gripper, max_gripper)
    safe[13] = np.clip(safe[13], min_gripper, max_gripper)
    diagnostics["left_gripper_clipped"] = bool(not np.isclose(safe[12], action_vec[12]))
    diagnostics["right_gripper_clipped"] = bool(not np.isclose(safe[13], action_vec[13]))
    return safe, diagnostics


def action_vector_to_flexiv_dict(action: Any) -> dict[str, float]:
    action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_vec.shape != (14,):
        raise ValueError(f"DP3 Flexiv action must have shape (14,), got {action_vec.shape}")
    if not np.isfinite(action_vec).all():
        raise ValueError("DP3 action contains NaN or Inf")
    return {key: float(value) for key, value in zip(ACTION_FIELD_NAMES, action_vec)}


def summarize_action(action: Any) -> dict[str, float]:
    action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_vec.shape != (14,):
        raise ValueError(f"DP3 Flexiv action must have shape (14,), got {action_vec.shape}")
    if not np.isfinite(action_vec).all():
        raise ValueError("DP3 action contains NaN or Inf")
    return {
        "left_xyz_norm": float(np.linalg.norm(action_vec[:3])),
        "left_rot_norm": float(np.linalg.norm(action_vec[3:6])),
        "right_xyz_norm": float(np.linalg.norm(action_vec[6:9])),
        "right_rot_norm": float(np.linalg.norm(action_vec[9:12])),
        "left_gripper_cmd": float(action_vec[12]),
        "right_gripper_cmd": float(action_vec[13]),
    }


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    current = cfg
    for part in path.split("."):
        try:
            if isinstance(current, Mapping):
                current = current[part]
            else:
                current = getattr(current, part)
        except (AttributeError, KeyError):
            if default is not None:
                return default
            raise
    return current


def _as_float(value: Any, *, key: str) -> float:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.shape == ():
        scalar = array.item()
    else:
        flat = array.reshape(-1)
        if flat.size != 1:
            raise ValueError(f"Observation field {key} must be scalar, got shape {array.shape}")
        scalar = flat[0].item() if hasattr(flat[0], "item") else flat[0]
    if isinstance(scalar, (bool, np.bool_, str, bytes)):
        raise ValueError(f"Observation field {key} must be a finite float")
    try:
        value_f = float(scalar)
    except (TypeError, ValueError):
        raise ValueError(f"Observation field {key} must be a finite float") from None
    if not np.isfinite(value_f):
        raise ValueError(f"Observation field {key} must be a finite float")
    return value_f


def _as_non_negative_int(value: Any, *, key: str) -> int:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.shape == ():
        scalar = array.item()
    else:
        flat = array.reshape(-1)
        if flat.size != 1:
            raise ValueError(f"Observation field {key} must be scalar, got shape {array.shape}")
        scalar = flat[0].item() if hasattr(flat[0], "item") else flat[0]
    if isinstance(scalar, (bool, np.bool_, str, bytes)):
        raise ValueError(f"Observation field {key} must be a non-negative integer")
    try:
        value_f = float(scalar)
    except (TypeError, ValueError):
        raise ValueError(f"Observation field {key} must be a non-negative integer") from None
    if not np.isfinite(value_f) or value_f < 0 or int(value_f) != value_f:
        raise ValueError(f"Observation field {key} must be a non-negative integer")
    return int(value_f)


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite positive integer")
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a finite positive integer") from None
    if not np.isfinite(value_f) or value_f <= 0 or int(value_f) != value_f:
        raise ValueError(f"{label} must be a finite positive integer")
    return int(value_f)


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(f"{label} must be finite")
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be finite") from None
    if not np.isfinite(value_f):
        raise ValueError(f"{label} must be finite")
    return value_f


def _optional_positive_float(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    value_f = _finite_float(value, label=label)
    if value_f <= 0.0:
        raise ValueError(f"{label} must be positive")
    return value_f


def _camera_base_name(camera_name: str) -> str:
    if camera_name.endswith("_rgb"):
        return camera_name.removesuffix("_rgb")
    if camera_name.endswith("_image"):
        return camera_name.removesuffix("_image")
    return camera_name


def _first_present_key(mapping: Mapping[str, Any], candidates: Sequence[str]) -> str | None:
    for key in candidates:
        if key in mapping:
            return key
    return None


def _tail_with_left_padding(history: Sequence[np.ndarray], length: int) -> list[np.ndarray]:
    if length <= 0:
        raise ValueError("n_obs_steps must be positive")
    items = list(history)[-length:]
    while len(items) < length:
        items.insert(0, items[0])
    return items


def _clip_vector_norm(vector: np.ndarray, limit: float | None) -> tuple[np.ndarray, dict[str, Any]]:
    norm = float(np.linalg.norm(vector))
    if limit is not None:
        limit = _optional_positive_float(limit, label="limit")
    if limit is None or norm <= float(limit) or norm < 1e-12:
        return vector, {"input_norm": norm, "clipped": False}
    return vector * (float(limit) / norm), {"input_norm": norm, "clipped": True}
