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

from diffusion_policy_3d.common.flexiv_state_contract import (
    FLEXIV_ACTION_DIM,
    FLEXIV_STATE_DIM,
    FLEXIV_STATE_ROTATION_REFERENCE,
    FLEXIV_STATE_ROTATION_REPRESENTATION,
    FLEXIV_STATE_SCHEMA,
    FLEXIV_ROTATION6D_CONVENTION,
    STATE_EE_POSITION_INDICES,
    STATE_EE_ROTATION_6D_INDICES,
    STATE_GRIPPER_INDICES,
    STATE_JOINT_INDICES,
    flexiv_action_names,
    flexiv_state_names,
    validate_flexiv_state_rotation6d,
)

AXES = ("x", "y", "z", "rx", "ry", "rz")
SIDES = ("left", "right")
FLEXIV_NORMALIZER_SCHEMA = FLEXIV_STATE_SCHEMA
STATE_FIELD_NAMES = tuple(flexiv_state_names())
ACTION_FIELD_NAMES = tuple(flexiv_action_names())
FLEXIV_STATE_INDEX_GROUPS = {
    "joints": STATE_JOINT_INDICES.copy(),
    "ee_position": STATE_EE_POSITION_INDICES.copy(),
    "ee_rotation_6d": STATE_EE_ROTATION_6D_INDICES.copy(),
    "grippers": STATE_GRIPPER_INDICES.copy(),
}

FFS_BACKEND_NAMES = (
    "pytorch",
    "tensorrt_single",
    "tensorrt_two_stage",
    "tensorrt_plugin",
)


@dataclass(frozen=True)
class PolicyContract:
    n_obs_steps: int
    state_dim: int
    action_dim: int
    pointcloud_points: int
    pointcloud_dim: int
    state_schema: str | None = None
    state_rotation_representation: str | None = None
    state_rotation_reference: str | None = None
    rotation6d_convention: str | None = None
    action_rotation_representation: str | None = None


@dataclass(frozen=True)
class PointCloudRuntimeContract:
    """The live observation contract derived from one PointCloudBuilder."""

    depth_source: str
    output_format: str
    use_rgb: bool
    num_points: int
    camera_name: str
    ffs_backend: str | None = None
    ffs_left_key: str | None = None
    ffs_right_key: str | None = None
    ffs_width: int | None = None
    ffs_height: int | None = None
    ffs_artifact_id: str | None = None
    ffs_config: Any | None = None

    @property
    def pointcloud_dim(self) -> int:
        return 6 if self.output_format == "xyzrgb" else 3


def pointcloud_runtime_contract_from_builder(builder: Any) -> PointCloudRuntimeContract:
    """Read all live camera/depth/point-cloud choices from a parsed builder."""

    config = getattr(builder, "config", None)
    if config is None:
        raise ValueError("PointCloudBuilder must expose its parsed config")
    camera = getattr(config, "camera", None)
    pointcloud = getattr(config, "pointcloud", None)
    sampling = getattr(config, "sampling", None)
    depth_source = getattr(config, "depth_source", None)
    if camera is None or pointcloud is None or sampling is None or depth_source is None:
        raise ValueError(
            "PointCloudBuilder config must expose camera, pointcloud, sampling, and depth_source"
        )

    output_format = str(getattr(pointcloud, "output_format", "")).strip().lower()
    if output_format not in {"xyz", "xyzrgb"}:
        raise ValueError(
            "PointCloudBuilder pointcloud.output_format must be 'xyz' or 'xyzrgb', "
            f"got {output_format!r}"
        )
    use_rgb = bool(getattr(pointcloud, "use_rgb", False))
    if use_rgb != (output_format == "xyzrgb"):
        raise ValueError(
            "PointCloudBuilder pointcloud.use_rgb must agree with "
            f"pointcloud.output_format={output_format!r}"
        )
    if not bool(getattr(sampling, "enabled", False)):
        raise ValueError(
            "Formal Flexiv inference requires PointCloudBuilder sampling.enabled=true"
        )
    num_points = _positive_int(
        getattr(sampling, "num_points", None),
        label="PointCloudBuilder sampling.num_points",
    )

    mode = str(getattr(depth_source, "mode", "frame")).strip().lower()
    camera_name = str(getattr(camera, "name", "camera")).strip() or "camera"
    if mode == "frame":
        return PointCloudRuntimeContract(
            depth_source="native_depth",
            output_format=output_format,
            use_rgb=use_rgb,
            num_points=num_points,
            camera_name=camera_name,
        )
    if mode != "ffs_stereo":
        raise ValueError(
            "PointCloudBuilder depth_source.mode must be 'frame' or 'ffs_stereo', "
            f"got {mode!r}"
        )

    ffs = getattr(depth_source, "ffs", None)
    if ffs is None:
        raise ValueError("PointCloudBuilder depth_source.ffs is required for ffs_stereo")
    backend = str(getattr(ffs, "backend", "")).strip().lower()
    if backend not in FFS_BACKEND_NAMES:
        raise ValueError(
            "PointCloudBuilder depth_source.ffs.backend must be one of "
            f"{FFS_BACKEND_NAMES}, got {backend!r}"
        )
    left_key = str(getattr(ffs, "left_key", "")).strip()
    right_key = str(getattr(ffs, "right_key", "")).strip()
    if not left_key or not right_key:
        raise ValueError("PointCloudBuilder FFS left_key and right_key must be non-empty")
    ffs_width = _positive_int(getattr(ffs, "width", None), label="FFS input width")
    ffs_height = _positive_int(getattr(ffs, "height", None), label="FFS input height")
    if (ffs_height, ffs_width) != (480, 640):
        raise ValueError(
            "Formal FFS inference requires input height=480,width=640; "
            f"got height={ffs_height}, width={ffs_width}"
        )
    artifact_id = getattr(ffs, "artifact_id", None)
    artifact_id = None if artifact_id is None else str(artifact_id).strip() or None
    return PointCloudRuntimeContract(
        depth_source="ffs_stereo",
        output_format=output_format,
        use_rgb=use_rgb,
        num_points=num_points,
        camera_name=camera_name,
        ffs_backend=backend,
        ffs_left_key=left_key,
        ffs_right_key=right_key,
        ffs_width=ffs_width,
        ffs_height=ffs_height,
        ffs_artifact_id=artifact_id,
        ffs_config=ffs,
    )


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
        state_schema=_optional_cfg_value(cfg, "task.dataset.state_schema"),
        state_rotation_representation=_optional_cfg_value(
            cfg,
            "task.dataset.state_rotation_representation",
        ),
        state_rotation_reference=_optional_cfg_value(
            cfg,
            "task.dataset.state_rotation_reference",
        ),
        rotation6d_convention=_optional_cfg_value(
            cfg,
            "task.dataset.rotation6d_convention",
        ),
        action_rotation_representation=_optional_cfg_value(
            cfg,
            "task.dataset.action_rotation_representation",
        ),
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
    if state_dim != FLEXIV_STATE_DIM:
        raise ValueError(
            f"Flexiv checkpoint state_dim must be {FLEXIV_STATE_DIM}, "
            f"got {contract.state_dim}"
        )
    if action_dim != FLEXIV_ACTION_DIM:
        raise ValueError(
            f"Flexiv checkpoint action_dim must be {FLEXIV_ACTION_DIM}, "
            f"got {contract.action_dim}"
        )
    if contract.state_schema != FLEXIV_STATE_SCHEMA:
        raise ValueError(
            "Flexiv checkpoint state_schema must be "
            f"{FLEXIV_STATE_SCHEMA!r}, got {contract.state_schema!r}"
        )
    if contract.state_rotation_representation != FLEXIV_STATE_ROTATION_REPRESENTATION:
        raise ValueError(
            "Flexiv checkpoint state rotation representation must be "
            f"{FLEXIV_STATE_ROTATION_REPRESENTATION!r}, "
            f"got {contract.state_rotation_representation!r}"
        )
    if contract.state_rotation_reference != FLEXIV_STATE_ROTATION_REFERENCE:
        raise ValueError(
            "Flexiv checkpoint state rotation reference must be "
            f"{FLEXIV_STATE_ROTATION_REFERENCE!r}, "
            f"got {contract.state_rotation_reference!r}"
        )
    if contract.rotation6d_convention != FLEXIV_ROTATION6D_CONVENTION:
        raise ValueError(
            "Flexiv checkpoint rotation6d convention must be "
            f"{FLEXIV_ROTATION6D_CONVENTION!r}, got {contract.rotation6d_convention!r}"
        )
    if contract.action_rotation_representation != "rotvec":
        raise ValueError(
            "Flexiv checkpoint action rotation representation must be 'rotvec', "
            f"got {contract.action_rotation_representation!r}"
        )
    if pointcloud_points <= 0:
        raise ValueError(
            f"pointcloud_points must be positive, got {contract.pointcloud_points}"
        )
    if pointcloud_dim not in (3, 6):
        raise ValueError(
            f"Flexiv checkpoint pointcloud_dim must be 3 or 6, got {contract.pointcloud_dim}"
        )


def validate_flexiv_normalizer_contract(
    policy: Any,
    *,
    normalizer_schema: Any,
    state_schema: Any = None,
    rotation6d_convention: Any = None,
    action_rotation_representation: Any = None,
    clip_actions_to_execution_limits: Any,
    action_xyz_limit: Any,
    action_rotation_limit: Any,
    state_joint_range_floor: Any,
    state_ee_position_range_floor: Any,
) -> dict[str, Any]:
    """Reject checkpoints that do not carry the v2 physical Flexiv contract."""

    if normalizer_schema != FLEXIV_NORMALIZER_SCHEMA:
        raise ValueError(
            "Checkpoint does not declare the Flexiv v2 normalizer "
            f"({FLEXIV_NORMALIZER_SCHEMA}); retrain it with the current DP3 train config"
        )
    if state_schema != FLEXIV_STATE_SCHEMA:
        raise ValueError(
            f"Checkpoint state_schema must be {FLEXIV_STATE_SCHEMA!r}, got {state_schema!r}"
        )
    if rotation6d_convention != FLEXIV_ROTATION6D_CONVENTION:
        raise ValueError(
            "Checkpoint rotation6d convention must be "
            f"{FLEXIV_ROTATION6D_CONVENTION!r}, got {rotation6d_convention!r}"
        )
    if action_rotation_representation != "rotvec":
        raise ValueError(
            "Checkpoint action rotation representation must be 'rotvec', "
            f"got {action_rotation_representation!r}"
        )
    if clip_actions_to_execution_limits is not True:
        raise ValueError(
            "Checkpoint did not train on collection-time execution-clipped actions"
        )
    xyz_limit = _positive_float(action_xyz_limit, label="action_xyz_limit")
    rotation_limit = _positive_float(
        action_rotation_limit,
        label="action_rotation_limit",
    )
    joint_floor = _positive_float(
        state_joint_range_floor,
        label="state_joint_range_floor",
    )
    position_floor = _positive_float(
        state_ee_position_range_floor,
        label="state_ee_position_range_floor",
    )
    normalizer = getattr(policy, "normalizer", None)
    if normalizer is None:
        raise ValueError("Checkpoint policy is missing its normalizer")
    try:
        action_params = normalizer["action"].params_dict
        state_params = normalizer["agent_pos"].params_dict
        action_scale = action_params["scale"].detach().cpu().numpy()
        action_offset = action_params["offset"].detach().cpu().numpy()
        state_scale = state_params["scale"].detach().cpu().numpy()
        state_offset = state_params["offset"].detach().cpu().numpy()
    except (AttributeError, KeyError) as exc:
        raise ValueError("Checkpoint normalizer is missing Flexiv action/agent_pos parameters") from exc
    expected_action_scale = np.asarray(
        [
            *([1.0 / xyz_limit] * 3),
            *([1.0 / rotation_limit] * 3),
            *([1.0 / xyz_limit] * 3),
            *([1.0 / rotation_limit] * 3),
            2.0,
            2.0,
        ],
        dtype=np.float32,
    )
    expected_action_offset = np.asarray([*([0.0] * 12), -1.0, -1.0], dtype=np.float32)
    if action_scale.shape != (FLEXIV_ACTION_DIM,) or not np.allclose(
        action_scale,
        expected_action_scale,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(
            "Checkpoint action normalizer does not match the configured physical limits"
        )
    if action_offset.shape != (FLEXIV_ACTION_DIM,) or not np.allclose(
        action_offset,
        expected_action_offset,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Checkpoint action normalizer does not use zero-centered deltas and [0,1] grippers"
        )
    if (
        state_scale.shape != (FLEXIV_STATE_DIM,)
        or state_offset.shape != (FLEXIV_STATE_DIM,)
        or not np.isfinite(state_scale).all()
        or not np.isfinite(state_offset).all()
    ):
        raise ValueError("Checkpoint agent_pos normalizer scale is invalid")
    if not np.allclose(
        state_scale[STATE_EE_ROTATION_6D_INDICES],
        1.0,
        rtol=0.0,
        atol=1e-6,
    ) or not np.allclose(
        state_offset[STATE_EE_ROTATION_6D_INDICES],
        0.0,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Checkpoint rotation-6D normalizer must use fixed scale=1 and offset=0"
        )
    if not np.allclose(
        state_scale[STATE_GRIPPER_INDICES],
        2.0,
        rtol=0.0,
        atol=1e-6,
    ) or not np.allclose(
        state_offset[STATE_GRIPPER_INDICES],
        -1.0,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Checkpoint state gripper normalizer must map [0, 1] to [-1, 1]"
        )
    maximum_state_scale = np.full(FLEXIV_STATE_DIM, np.inf, dtype=np.float32)
    joint_indices = STATE_JOINT_INDICES
    position_indices = STATE_EE_POSITION_INDICES
    rotation_indices = STATE_EE_ROTATION_6D_INDICES
    gripper_indices = STATE_GRIPPER_INDICES
    maximum_state_scale[joint_indices] = 2.0 / joint_floor
    maximum_state_scale[position_indices] = 2.0 / position_floor
    maximum_state_scale[rotation_indices] = 1.0
    maximum_state_scale[gripper_indices] = 2.0
    if np.any(state_scale > maximum_state_scale + 1e-5):
        indices = np.flatnonzero(state_scale > maximum_state_scale + 1e-5).tolist()
        raise ValueError(
            "Checkpoint agent_pos normalizer violates the configured range floors "
            f"at indices {indices}"
        )
    return {
        "schema": FLEXIV_NORMALIZER_SCHEMA,
        "action_xyz_limit": xyz_limit,
        "action_rotation_limit": rotation_limit,
        "max_agent_pos_scale": float(np.max(state_scale)),
        "rotation6d_scale": 1.0,
        "rotation6d_offset": 0.0,
    }


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
    """Build the strict 34D absolute rotation-6D Flexiv state vector."""

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
    if int(expected_dim) != FLEXIV_STATE_DIM:
        raise ValueError(
            f"Flexiv v2 agent_pos expected_dim must be {FLEXIV_STATE_DIM}, got {expected_dim}"
        )
    validate_flexiv_state_rotation6d(values, context="live agent_pos")
    for index, key in ((16, "left_gripper_state_norm"), (33, "right_gripper_state_norm")):
        if index < values.shape[0] and not 0.0 <= float(values[index]) <= 1.0:
            raise ValueError(f"{key}={float(values[index]):.6g} outside [0, 1]")
    return values


def build_pointcloud_frame_from_observation(
    observation: Mapping[str, Any],
    *,
    camera_name: str = "head_rgb",
    rgb_key: str | None = None,
    depth_key: str | None = None,
    builder: Any | None = None,
    runtime_contract: PointCloudRuntimeContract | None = None,
) -> dict[str, Any]:
    """Extract one native-depth or FFS frame from a Flexiv observation dict.

    The FFS branch intentionally never reads or inserts native ``depth``.  The
    adapter publishes stable sidecar names, while the Builder's configured
    ``left_key``/``right_key`` decide the keys passed to its estimator.
    """

    contract = runtime_contract
    if contract is None and builder is not None:
        contract = pointcloud_runtime_contract_from_builder(builder)
    if contract is not None and contract.depth_source == "ffs_stereo":
        return _build_ffs_pointcloud_frame_from_observation(
            observation,
            camera_name=camera_name,
            rgb_key=rgb_key,
            contract=contract,
        )

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
            f"Missing depth frame for camera {camera_name!r} with depth_source='native_depth'. "
            "Enable `save_depth_sidecar` on the Flexiv robot "
            f"config and expected one of sidecar.{base_name}_depth or sidecar.{camera_name}_depth."
        )

    frame: dict[str, Any] = {"depth": observation[resolved_depth_key]}
    if resolved_rgb_key is not None:
        frame["rgb"] = observation[resolved_rgb_key]
    if contract is not None and contract.output_format == "xyzrgb" and resolved_rgb_key is None:
        raise KeyError(
            f"Missing RGB field for camera {camera_name!r} with depth_source='native_depth' "
            "and pointcloud output_format='xyzrgb'"
        )
    _copy_camera_frame_metadata(frame, observation, base_name=base_name, required=False)
    frame["depth_source"] = "native_depth"
    return frame


def _build_ffs_pointcloud_frame_from_observation(
    observation: Mapping[str, Any],
    *,
    camera_name: str,
    rgb_key: str | None,
    contract: PointCloudRuntimeContract,
) -> dict[str, Any]:
    """Build the exact stereo-IR frame required by PointCloudBuilder."""

    base_name = _camera_base_name(camera_name)
    left_key = str(contract.ffs_left_key)
    right_key = str(contract.ffs_right_key)
    left_observation_key = _resolve_ffs_observation_key(
        observation,
        configured_key=left_key,
        camera_name=camera_name,
        base_name=base_name,
        side="left",
    )
    right_observation_key = _resolve_ffs_observation_key(
        observation,
        configured_key=right_key,
        camera_name=camera_name,
        base_name=base_name,
        side="right",
    )
    expected_shape = (int(contract.ffs_height), int(contract.ffs_width))
    left = _validate_live_ir(
        observation[left_observation_key],
        field=left_observation_key,
        camera_name=camera_name,
        expected_shape=expected_shape,
    )
    right = _validate_live_ir(
        observation[right_observation_key],
        field=right_observation_key,
        camera_name=camera_name,
        expected_shape=expected_shape,
    )
    if left.shape != right.shape:
        raise ValueError(
            f"FFS stereo IR shape mismatch for camera {camera_name!r} with "
            f"depth_source='ffs_stereo': left={left.shape}, right={right.shape}"
        )

    timestamp_key = f"{base_name}_rgbd_timestamp"
    wall_time_key = f"{base_name}_rgbd_wall_time"
    if timestamp_key not in observation:
        raise KeyError(
            f"Missing {timestamp_key!r} for camera {camera_name!r} with "
            "depth_source='ffs_stereo'; stereo freshness cannot be checked"
        )
    if wall_time_key not in observation:
        raise KeyError(
            f"Missing {wall_time_key!r} for camera {camera_name!r} with "
            "depth_source='ffs_stereo'; stereo frame age cannot be checked"
        )
    timestamp = _as_float(observation[timestamp_key], key=timestamp_key)
    wall_time = _as_float(observation[wall_time_key], key=wall_time_key)
    if "global_frame_index" not in observation:
        raise KeyError(
            f"Missing 'global_frame_index' for camera {camera_name!r} with "
            "depth_source='ffs_stereo'; stereo frame reuse cannot be checked"
        )
    global_frame_index = _as_non_negative_int(
        observation["global_frame_index"],
        key="global_frame_index",
    )
    frame: dict[str, Any] = {
        left_key: left,
        right_key: right,
        "timestamp": timestamp,
        "wall_time": wall_time,
        "global_frame_index": global_frame_index,
        "depth_source": "ffs_stereo",
        "ffs_backend": contract.ffs_backend,
        "ffs_left_key": left_key,
        "ffs_right_key": right_key,
    }
    _copy_camera_frame_metadata(frame, observation, base_name=base_name, required=False)
    _validate_ffs_pair_metadata(
        frame,
        observation,
        base_name=base_name,
        camera_name=camera_name,
        common_timestamp=timestamp,
    )

    resolved_rgb_key = rgb_key or _first_present_key(
        observation,
        (camera_name, f"{base_name}_rgb", f"{base_name}_image", base_name),
    )
    if resolved_rgb_key is not None:
        frame["rgb"] = observation[resolved_rgb_key]
    if contract.output_format == "xyzrgb" and resolved_rgb_key is None:
        raise KeyError(
            f"Missing RGB field for camera {camera_name!r} with depth_source='ffs_stereo' "
            "and pointcloud output_format='xyzrgb'; FFS does not fall back to native depth"
        )
    return frame


def _resolve_ffs_observation_key(
    observation: Mapping[str, Any],
    *,
    configured_key: str,
    camera_name: str,
    base_name: str,
    side: str,
) -> str:
    """Resolve a Builder key while retaining the adapter's sidecar names."""

    candidates: list[str] = [configured_key]
    candidates.extend(
        (
            f"sidecar.{base_name}_{configured_key}",
            f"{base_name}_{configured_key}",
            f"{camera_name}_{configured_key}",
            f"sidecar.{base_name}_{side}_ir",
            f"{base_name}_{side}_ir",
            f"{camera_name}_{side}_ir",
            f"{side}_ir",
        )
    )
    for candidate in dict.fromkeys(candidates):
        if candidate in observation:
            return candidate
    raise KeyError(
        f"Missing {side} IR field for camera {camera_name!r} with "
        "depth_source='ffs_stereo'; "
        f"Builder {side}_key={configured_key!r}; checked {list(dict.fromkeys(candidates))!r}. "
        "Native depth is not an FFS fallback"
    )


def _validate_live_ir(
    value: Any,
    *,
    field: str,
    camera_name: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != expected_shape:
        raise ValueError(
            f"FFS stereo IR {field} shape for camera {camera_name!r} with "
            "depth_source='ffs_stereo' "
            f"is {array.shape}, expected {expected_shape}"
        )
    if array.dtype.kind not in {"b", "u", "i", "f"}:
        raise ValueError(f"FFS {field} for camera {camera_name!r} must be numeric")
    numeric = np.asarray(array, dtype=np.float32)
    if not np.isfinite(numeric).all() or np.any(numeric < 0.0) or np.any(numeric > 255.0):
        raise ValueError(
            f"FFS {field} for camera {camera_name!r} must contain finite 0..255 pixels"
        )
    return array


def _copy_camera_frame_metadata(
    frame: dict[str, Any],
    observation: Mapping[str, Any],
    *,
    base_name: str,
    required: bool,
) -> None:
    timestamp_key = f"{base_name}_rgbd_timestamp"
    wall_time_key = f"{base_name}_rgbd_wall_time"
    frame_index_key = f"{base_name}_rgbd_frame_index"
    if timestamp_key in observation:
        frame["timestamp"] = _as_float(observation[timestamp_key], key=timestamp_key)
    elif required:
        raise KeyError(f"Missing {timestamp_key!r}")
    if wall_time_key in observation:
        frame["wall_time"] = _as_float(observation[wall_time_key], key=wall_time_key)
    elif required:
        raise KeyError(f"Missing {wall_time_key!r}")
    if frame_index_key in observation:
        frame["frame_index"] = _as_non_negative_int(
            observation[frame_index_key],
            key=frame_index_key,
        )
    if "global_frame_index" in observation:
        frame["global_frame_index"] = _as_non_negative_int(
            observation["global_frame_index"],
            key="global_frame_index",
        )
    for suffix, field_name in (
        ("left_ir_timestamp", "left_ir_timestamp"),
        ("right_ir_timestamp", "right_ir_timestamp"),
        ("left_ir_frame_index", "left_ir_frame_index"),
        ("right_ir_frame_index", "right_ir_frame_index"),
    ):
        key = f"{base_name}_{suffix}"
        if key not in observation:
            continue
        if suffix.endswith("frame_index"):
            frame[field_name] = _as_non_negative_int(observation[key], key=key)
        else:
            frame[field_name] = _as_float(observation[key], key=key)


def _validate_ffs_pair_metadata(
    frame: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    base_name: str,
    camera_name: str,
    common_timestamp: float,
) -> None:
    del observation
    left_timestamp = frame.get("left_ir_timestamp")
    right_timestamp = frame.get("right_ir_timestamp")
    if left_timestamp is None and right_timestamp is None:
        raise ValueError(
            f"FFS stereo IR timestamp metadata is required for camera {camera_name!r}: "
            f"{base_name}_left_ir_timestamp and {base_name}_right_ir_timestamp must "
            "identify the same frameset"
        )
    if (left_timestamp is None) != (right_timestamp is None):
        raise ValueError(
            f"FFS stereo IR timestamp metadata is incomplete for camera {camera_name!r}: "
            f"{base_name}_left_ir_timestamp and {base_name}_right_ir_timestamp are a pair"
        )
    if left_timestamp is not None:
        if not np.isclose(float(left_timestamp), float(right_timestamp), rtol=0.0, atol=1e-6):
            raise ValueError(
                f"FFS stereo IR timestamp mismatch for camera {camera_name!r}: "
                f"left={left_timestamp!r}, right={right_timestamp!r}"
            )
        if not np.isclose(float(left_timestamp), float(common_timestamp), rtol=0.0, atol=1e-6):
            raise ValueError(
                f"FFS stereo IR timestamp does not match the RGB frame for camera {camera_name!r}: "
                f"ir={left_timestamp!r}, rgb={common_timestamp!r}"
            )
    left_index = frame.get("left_ir_frame_index")
    right_index = frame.get("right_ir_frame_index")
    if left_index is None and right_index is None:
        raise ValueError(
            f"FFS stereo IR frame-index metadata is required for camera {camera_name!r}: "
            f"{base_name}_left_ir_frame_index and {base_name}_right_ir_frame_index "
            "must identify the same frameset"
        )
    if (left_index is None) != (right_index is None):
        raise ValueError(
            f"FFS stereo IR frame-index metadata is incomplete for camera {camera_name!r}: "
            f"{base_name}_left_ir_frame_index and {base_name}_right_ir_frame_index are a pair"
        )
    if left_index is not None and int(left_index) != int(right_index):
        raise ValueError(
            f"FFS stereo IR frame-index mismatch for camera {camera_name!r}: "
            f"left={left_index!r}, right={right_index!r}"
        )
    camera_index = frame.get("frame_index")
    if left_index is not None and camera_index is not None and int(left_index) != int(camera_index):
        raise ValueError(
            f"FFS stereo IR frame index does not match the RGB frame for camera {camera_name!r}: "
            f"ir={left_index!r}, rgb={camera_index!r}"
        )


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


def _optional_cfg_value(cfg: Any, path: str) -> Any:
    try:
        return _cfg_get(cfg, path)
    except (AttributeError, KeyError, TypeError):
        return None


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


def _positive_float(value: Any, *, label: str) -> float:
    value_f = _finite_float(value, label=label)
    if value_f <= 0.0:
        raise ValueError(f"{label} must be positive")
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
