#!/usr/bin/env python3
"""Run Flexiv dual-arm DP3 inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import logging
import math
import os
import site
import sys
import time
import types
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

USER_SITE = site.getusersitepackages()
sys.path = [path for path in sys.path if Path(path).resolve() != Path(USER_SITE).resolve()]

import numpy as np
import yaml
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
DP3_ROOT = REPO_ROOT / "3D-Diffusion-Policy"
POINTCLOUD_BUILDER_ROOT = REPO_ROOT / "PointCloudBuilder"
DEFAULT_INFERENCE_CONFIG = (
    DP3_ROOT / "diffusion_policy_3d/config/dp3_inference_config.yaml"
)


def _has_flexiv_interface_files(path: Path) -> bool:
    return (path / "config_flexiv.py").is_file() and (path / "flexiv_dual_arm.py").is_file()


def _resolve_flexiv_interface_dir(repo_root: Path = REPO_ROOT) -> Path:
    current_layout = repo_root / "third_party" / "real" / "dual_flexiv_rizon4s" / "interface"
    legacy_layout = repo_root / "third_party" / "real" / "flexiv-GN01" / "interface"
    for candidate in (current_layout, legacy_layout):
        if _has_flexiv_interface_files(candidate):
            return candidate
    return current_layout


FLEXIV_INTERFACE_DIR = _resolve_flexiv_interface_dir()
DEFAULT_LEROBOT_SRC = Path(os.environ.get("LEROBOT_SRC", Path.home() / "flexiv_ws" / "Le-nero" / "src"))

for path in (REPO_ROOT, DP3_ROOT, POINTCLOUD_BUILDER_ROOT, DEFAULT_LEROBOT_SRC):
    if path.exists():
        sys.path.insert(0, str(path))

from diffusion_policy_3d.real_world.flexiv_dual_arm_dp3 import (  # noqa: E402
    ACTION_FIELD_NAMES,
    STATE_FIELD_NAMES,
    SafetyLimits,
    action_vector_to_flexiv_dict,
    build_agent_pos,
    build_pointcloud_frame_from_observation,
    filter_action_vector,
    history_to_policy_obs,
    load_dp3_policy_from_checkpoint,
    policy_contract_from_cfg,
    prepare_point_cloud,
    summarize_action,
    validate_agent_pos,
    validate_policy_contract,
)
from pointcloud_builder import PointCloudBuilder  # noqa: E402
from pointcloud_builder.config import load_config as load_pointcloud_config  # noqa: E402
from tools.flexiv_dp3_live_viewer import (  # noqa: E402
    LiveVisualizationPublisher,
    ViewerConfig,
)
LOGGER = logging.getLogger("flexiv_dp3_inference")
FUTURE_WALL_TIME_TOLERANCE_S = 0.25
MAX_OPEN3D_VISUALIZATION_RATE_HZ = 2.0
MAX_OPEN3D_DISPLAY_POINTS = 50_000

CLI_DESCRIPTION = """Run Flexiv dual-arm DP3 inference.

All deployment parameters are loaded from dp3_inference_config.yaml by default.
Inference connects to the configured hardware and sends policy actions directly.
Use --check-config to validate files, checkpoint compatibility, PointCloudBuilder,
and the Flexiv adapter without calling robot.connect().
"""


@dataclass(frozen=True)
class _QueuedAction:
    vector: Any
    predicted_at: float
    chunk_index: int
    chunk_size: int
    camera_frame_age_ms_at_prediction: float | None
    camera_frame_timestamp: float | None
    camera_frame_wall_time: float | None
    camera_frame_index: int | None
    point_cloud_padded: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=CLI_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_INFERENCE_CONFIG,
        help="Inference YAML. Defaults to diffusion_policy_3d/config/dp3_inference_config.yaml.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration and checkpoint compatibility without hardware access.",
    )
    cli_args = parser.parse_args()
    return _args_from_inference_config(cli_args.config, check_config=cli_args.check_config)


def _args_from_inference_config(config_path: Path, *, check_config: bool) -> argparse.Namespace:
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"Inference config does not exist: {config_path}")
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Inference config must contain a mapping: {config_path}")

    checkpoint = _required_mapping(cfg, "checkpoint", config_path)
    robot = _required_mapping(cfg, "robot", config_path)
    pointcloud = _required_mapping(cfg, "pointcloud", config_path)
    inference = _required_mapping(cfg, "inference", config_path)
    visualization = _required_mapping(cfg, "visualization", config_path)

    rate_hz = float(inference["rate_hz"])
    duration_value = inference.get("duration_seconds")
    duration_seconds = None if duration_value is None else float(duration_value)
    num_steps = (
        None
        if duration_seconds is None
        else max(1, int(round(rate_hz * duration_seconds)))
    )
    point_dim = int(cfg["shape_meta"]["obs"]["point_cloud"]["shape"][1])
    point_mode = "xyz" if point_dim == 3 else "xyzrgb"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = _resolve_repo_path(inference["log_dir"])
    run_extent = "until_stopped" if num_steps is None else f"{num_steps}step"
    summary_jsonl = log_dir / f"flexiv_dp3_inference_{point_mode}_{stamp}_{run_extent}.jsonl"

    return argparse.Namespace(
        config_path=config_path,
        inference_config=cfg,
        ckpt=_resolve_repo_path(checkpoint["path"]),
        pointcloud_config=_resolve_repo_path(pointcloud["config"]),
        robot_config=_resolve_repo_path(robot["config"]),
        lerobot_src=_resolve_repo_path(robot.get("lerobot_src", DEFAULT_LEROBOT_SRC)),
        mode="inference",
        gpu_id=int(inference["gpu_id"]),
        device=str(inference["device"]),
        pointcloud_device=str(pointcloud["device"]),
        duration_seconds=duration_seconds,
        num_steps=num_steps,
        rate_hz=rate_hz,
        action_mode=str(inference["action_mode"]),
        check_config=bool(check_config),
        camera_name=str(robot.get("camera_name", "head_rgb")),
        rgb_key=None,
        depth_key=None,
        head_camera_serial=None,
        camera_width=None,
        camera_height=None,
        camera_fps=int(robot.get("camera_fps", 30)),
        default_gripper_state=None,
        read_gripper_state=False,
        low_speed_scale=float(inference["low_speed_scale"]),
        max_cartesian_delta=float(inference["max_cartesian_delta"]),
        max_rotation_delta=float(inference["max_rotation_delta"]),
        stop_file=_resolve_repo_path(inference["stop_file"]),
        summary_jsonl=None if check_config else summary_jsonl,
        overwrite_summary_jsonl=False,
        sampled_pointcloud_dir=None,
        sampled_pointcloud_every=1,
        visualize_live=bool(visualization["enabled"]) and not check_config,
        visualization_rate_hz=float(visualization["rate_hz"]),
        visualization_max_raw_points=int(visualization["max_raw_points"]),
        visualization_max_cropped_points=int(visualization["max_cropped_points"]),
        visualization_point_size=float(visualization["point_size"]),
        allow_reused_rgbd=False,
        max_policy_latency_ms=float(inference["max_policy_latency_ms"]),
        max_camera_frame_age_ms=float(inference["max_camera_frame_age_ms"]),
        max_action_age_ms=float(inference["max_action_age_ms"]),
        max_send_duration_ms=float(inference["max_send_duration_ms"]),
        max_loop_overrun_ms=float(inference["max_loop_overrun_ms"]),
        i_understand_this_moves_robot=True,
        allow_connect_motion=bool(robot.get("allow_connect_motion", False)),
        switch_cartesian_mode_on_connect=bool(
            robot.get("switch_cartesian_mode_on_connect", True)
        ),
        robot_debug=False,
        num_inference_steps=int(inference["num_inference_steps"]),
        log_level=str(inference.get("log_level", "INFO")).upper(),
    )


def _required_mapping(cfg: Mapping[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"`{key}` must be a mapping in inference config: {path}")
    return value


def _resolve_repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


_CONSISTENCY_FIELDS = (
    ("policy._target_", "policy._target_"),
    ("horizon", "policy.horizon"),
    ("n_obs_steps", "policy.n_obs_steps"),
    ("n_action_steps", "policy.n_action_steps"),
    ("obs_as_global_cond", "policy.obs_as_global_cond"),
    ("policy.use_point_crop", "policy.use_point_crop"),
    ("policy.condition_type", "policy.condition_type"),
    ("policy.use_down_condition", "policy.use_down_condition"),
    ("policy.use_mid_condition", "policy.use_mid_condition"),
    ("policy.use_up_condition", "policy.use_up_condition"),
    ("policy.diffusion_step_embed_dim", "policy.diffusion_step_embed_dim"),
    ("policy.down_dims", "policy.down_dims"),
    ("policy.crop_shape", "policy.crop_shape"),
    ("policy.encoder_output_dim", "policy.encoder_output_dim"),
    ("policy.kernel_size", "policy.kernel_size"),
    ("policy.n_groups", "policy.n_groups"),
    ("policy.noise_scheduler._target_", "policy.noise_scheduler._target_"),
    ("policy.noise_scheduler.num_train_timesteps", "policy.noise_scheduler.num_train_timesteps"),
    ("policy.noise_scheduler.beta_start", "policy.noise_scheduler.beta_start"),
    ("policy.noise_scheduler.beta_end", "policy.noise_scheduler.beta_end"),
    ("policy.noise_scheduler.beta_schedule", "policy.noise_scheduler.beta_schedule"),
    ("policy.noise_scheduler.clip_sample", "policy.noise_scheduler.clip_sample"),
    ("policy.noise_scheduler.set_alpha_to_one", "policy.noise_scheduler.set_alpha_to_one"),
    ("policy.noise_scheduler.steps_offset", "policy.noise_scheduler.steps_offset"),
    ("policy.noise_scheduler.prediction_type", "policy.noise_scheduler.prediction_type"),
    ("policy.use_pc_color", "policy.use_pc_color"),
    ("policy.pointnet_type", "policy.pointnet_type"),
    ("policy.pointcloud_encoder_cfg.in_channels", "policy.pointcloud_encoder_cfg.in_channels"),
    ("policy.pointcloud_encoder_cfg.out_channels", "policy.pointcloud_encoder_cfg.out_channels"),
    ("policy.pointcloud_encoder_cfg.use_layernorm", "policy.pointcloud_encoder_cfg.use_layernorm"),
    ("policy.pointcloud_encoder_cfg.final_norm", "policy.pointcloud_encoder_cfg.final_norm"),
    ("policy.pointcloud_encoder_cfg.normal_channel", "policy.pointcloud_encoder_cfg.normal_channel"),
    ("shape_meta.obs.point_cloud.shape", "shape_meta.obs.point_cloud.shape"),
    ("shape_meta.obs.agent_pos.shape", "shape_meta.obs.agent_pos.shape"),
    ("shape_meta.action.shape", "shape_meta.action.shape"),
    ("training.use_ema", "use_ema"),
)


def _validate_checkpoint_inference_config(checkpoint_cfg: Any, inference_cfg: Mapping[str, Any]) -> None:
    mismatches = []
    for checkpoint_key, inference_key in _CONSISTENCY_FIELDS:
        actual = _plain_config_value(OmegaConf.select(checkpoint_cfg, checkpoint_key))
        expected = _plain_config_value(_mapping_select(inference_cfg, inference_key))
        if actual != expected:
            mismatches.append(
                f"{checkpoint_key}: checkpoint={actual!r}, inference_config={expected!r}"
            )
    if mismatches:
        details = "\n  - ".join(mismatches)
        raise SystemExit(
            "dp3_inference_config.yaml does not match the checkpoint training contract:\n"
            f"  - {details}\n"
            "Use the matching checkpoint/config pair. Only inference-only fields may differ."
        )


def _mapping_select(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _plain_config_value(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, tuple):
        return list(value)
    return value


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    _validate_args(args)
    _validate_input_files(args)

    device = _resolve_device(args.device)
    policy, cfg, _workspace = load_dp3_policy_from_checkpoint(args.ckpt, device)
    _validate_checkpoint_inference_config(cfg, args.inference_config)
    if args.num_inference_steps is not None:
        policy.num_inference_steps = int(args.num_inference_steps)
    contract = policy_contract_from_cfg(cfg)
    validate_policy_contract(contract)
    LOGGER.info(
        "Loaded checkpoint=%s n_obs_steps=%d point_cloud=(%d,%d) state_dim=%d action_dim=%d device=%s",
        args.ckpt,
        contract.n_obs_steps,
        contract.pointcloud_points,
        contract.pointcloud_dim,
        contract.state_dim,
        contract.action_dim,
        device,
    )

    builder = _load_pointcloud_builder(args.pointcloud_config, args.pointcloud_device)
    _validate_builder_contract(builder, contract)
    LOGGER.info("Loaded PointCloudBuilder config=%s device=%s", args.pointcloud_config, builder.device)
    artifacts = _artifact_audit(args)

    _add_import_path(args.lerobot_src)
    if args.check_config:
        FlexivDualArmConfig, FlexivDualArm = _load_flexiv_interface(FLEXIV_INTERFACE_DIR)
        robot_config = _load_flexiv_config(FlexivDualArmConfig, args)
        _validate_robot_camera_contract(robot_config, builder, args.camera_name)
        robot_probe = FlexivDualArm(robot_config)
        _validate_adapter_feature_contract(
            robot_probe,
            args.camera_name,
            default_gripper_state=args.default_gripper_state,
        )
        print(
            json.dumps(
                _config_check_summary(args, cfg, contract, builder, robot_config, artifacts),
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 0

    print(f"[inference] checkpoint: {_display_path(args.ckpt)}")
    print(f"[inference] log: {_display_path(args.summary_jsonl)}")
    duration_label = (
        "until Ctrl+C or stop file"
        if args.duration_seconds is None
        else f"{args.duration_seconds:g}s ({args.num_steps} steps max)"
    )
    print(
        "[inference] "
        f"duration={duration_label} rate_hz={args.rate_hz:g} "
        f"action_mode={args.action_mode} diffusion_steps={policy.num_inference_steps}"
    )
    robot = _make_flexiv_robot(args, builder)
    limits = SafetyLimits(
        max_cartesian_delta=args.max_cartesian_delta,
        max_rotation_delta=args.max_rotation_delta,
        low_speed_scale=args.low_speed_scale,
    )
    try:
        return _run_inference_loop(
            args=args,
            robot=robot,
            policy=policy,
            builder=builder,
            contract=contract,
            device=device,
            limits=limits,
            connect_robot=True,
            artifacts=artifacts,
        )
    except KeyboardInterrupt:
        print("\n[inference] stopped by user; robot cleanup complete")
        return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.mode != "inference":
        raise SystemExit(f"Unsupported runtime mode: {args.mode!r}")
    if args.robot_debug is not False:
        raise SystemExit("Inference requires robot debug=false so send_action is active")
    if args.allow_reused_rgbd:
        raise SystemExit("Inference never sends actions from reused RGB-D frames")
    if args.default_gripper_state is not None:
        raise SystemExit("Inference requires live hardware gripper state")
    if args.overwrite_summary_jsonl:
        raise SystemExit("Inference logs must use a fresh path")
    if args.camera_name != "head_rgb":
        raise SystemExit("Inference requires robot.camera_name=head_rgb")
    if args.rgb_key is not None or args.depth_key is not None:
        raise SystemExit("Inference requires the default head RGB-D observation keys")
    if args.action_mode not in {"receding", "chunk"}:
        raise SystemExit("inference.action_mode must be receding or chunk")
    if args.gpu_id < 0:
        raise SystemExit("inference.gpu_id must be a non-negative integer")
    if args.device.startswith("cuda") and args.device != "cuda:0":
        raise SystemExit(
            "inference.device must be cuda:0 after gpu_id masks the selected physical GPU"
        )
    if args.pointcloud_device.startswith("cuda") and args.pointcloud_device != "cuda:0":
        raise SystemExit(
            "pointcloud.device must be cuda:0 after gpu_id masks the selected physical GPU"
        )
    if args.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise SystemExit("inference.log_level must be DEBUG, INFO, WARNING, or ERROR")
    for name in (
        "max_policy_latency_ms",
        "max_camera_frame_age_ms",
        "max_action_age_ms",
        "max_send_duration_ms",
        "max_loop_overrun_ms",
    ):
        if getattr(args, name) is None:
            raise SystemExit(f"Missing inference.{name}")
    if not args.check_config and args.summary_jsonl is None:
        raise SystemExit("Inference requires an audit JSONL path")
    if (
        not args.check_config
        and args.summary_jsonl is not None
        and Path(args.summary_jsonl).expanduser().exists()
        and not args.overwrite_summary_jsonl
    ):
        raise SystemExit(
            f"--summary-jsonl already exists; refusing to overwrite audit log: {args.summary_jsonl}"
        )
    if (
        not args.check_config
        and args.summary_jsonl is not None
        and args.stop_file is not None
        and _same_resolved_path(args.summary_jsonl, args.stop_file)
    ):
        raise SystemExit(
            "--summary-jsonl and --stop-file must be different paths so the "
            "operator stop signal cannot collide with the audit log"
        )
    switch_cartesian_mode = bool(getattr(args, "switch_cartesian_mode_on_connect", False))
    if args.allow_connect_motion and switch_cartesian_mode:
        raise SystemExit(
            "robot.allow_connect_motion and robot.switch_cartesian_mode_on_connect are mutually exclusive"
        )
    if args.robot_config is None:
        raise SystemExit("robot.config is required for Flexiv inference")
    if (
        not args.check_config
        and args.stop_file is not None
        and Path(args.stop_file).expanduser().exists()
    ):
        raise SystemExit(f"--stop-file already exists; refusing to start inference: {args.stop_file}")
    if args.num_steps is not None:
        args.num_steps = _positive_int(args.num_steps, label="inference step limit")
    if args.duration_seconds is not None:
        args.duration_seconds = _positive_float(
            args.duration_seconds,
            label="inference.duration_seconds",
        )
    args.rate_hz = _positive_float(args.rate_hz, label="--rate-hz")
    args.sampled_pointcloud_every = _positive_int(
        args.sampled_pointcloud_every,
        label="--sampled-pointcloud-every",
    )
    args.visualization_rate_hz = _positive_float(
        args.visualization_rate_hz,
        label="--visualization-rate-hz",
    )
    if args.visualization_rate_hz > MAX_OPEN3D_VISUALIZATION_RATE_HZ:
        raise SystemExit(
            "--visualization-rate-hz must be <= "
            f"{MAX_OPEN3D_VISUALIZATION_RATE_HZ:g} to protect the inference loop"
        )
    for name in ("visualization_max_raw_points", "visualization_max_cropped_points"):
        value = _positive_int(
            getattr(args, name),
            label=f"--{name.replace('_', '-')}",
        )
        if value > MAX_OPEN3D_DISPLAY_POINTS:
            raise SystemExit(
                f"--{name.replace('_', '-')} must be <= {MAX_OPEN3D_DISPLAY_POINTS}"
            )
        setattr(args, name, value)
    args.visualization_point_size = _positive_float(
        args.visualization_point_size,
        label="--visualization-point-size",
    )
    if args.num_inference_steps is not None:
        args.num_inference_steps = _positive_int(
            args.num_inference_steps,
            label="--num-inference-steps",
        )
    args.low_speed_scale = _bounded_float(
        args.low_speed_scale,
        label="--low-speed-scale",
        minimum=0.0,
        maximum=1.0,
    )
    if args.max_cartesian_delta is not None:
        args.max_cartesian_delta = _positive_float(
            args.max_cartesian_delta,
            label="--max-cartesian-delta",
        )
    if args.max_rotation_delta is not None:
        args.max_rotation_delta = _positive_float(
            args.max_rotation_delta,
            label="--max-rotation-delta",
        )
    if args.default_gripper_state is not None:
        args.default_gripper_state = _bounded_float(
            args.default_gripper_state,
            label="--default-gripper-state",
            minimum=0.0,
            maximum=1.0,
        )
    for name in ("camera_width", "camera_height", "camera_fps"):
        value = getattr(args, name)
        if value is None:
            continue
        setattr(args, name, _positive_int(value, label=f"--{name.replace('_', '-')}"))
    if args.pointcloud_device is not None:
        args.pointcloud_device = _validate_pointcloud_device_arg(args.pointcloud_device)
    for name in (
        "max_policy_latency_ms",
        "max_camera_frame_age_ms",
        "max_action_age_ms",
        "max_send_duration_ms",
        "max_loop_overrun_ms",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, _positive_float(value, label=f"--{name.replace('_', '-')}"))


def _validate_input_files(args: argparse.Namespace) -> None:
    _require_input_file(args.ckpt, "--ckpt")
    _require_input_file(args.pointcloud_config, "--pointcloud-config")
    _require_input_file(args.robot_config, "--robot-config")


def _require_input_file(path: str | Path, label: str) -> Path:
    expanded = Path(path).expanduser()
    if not expanded.is_file():
        raise SystemExit(f"{label} does not exist or is not a file: {path}")
    return expanded


def _connect_motion_allowed(args: argparse.Namespace) -> bool:
    return args.mode == "inference" and bool(args.allow_connect_motion)


def _switch_cartesian_mode_allowed(args: argparse.Namespace) -> bool:
    return args.mode == "inference" and bool(
        getattr(args, "switch_cartesian_mode_on_connect", False)
    )


def _run_inference_loop(
    *,
    args: argparse.Namespace,
    robot: Any,
    policy: Any,
    builder: PointCloudBuilder,
    contract: Any,
    device: torch.device,
    limits: SafetyLimits,
    connect_robot: bool,
    artifacts: dict[str, Any] | None = None,
) -> int:
    _enforce_runtime_inference_gates(args, robot, limits, connect_robot=connect_robot)
    agent_history: deque = deque(maxlen=contract.n_obs_steps)
    point_cloud_history: deque = deque(maxlen=contract.n_obs_steps)
    action_queue: deque = deque()
    last_camera_frame_identity: tuple[str, float] | None = None
    connect_started = False
    stop_file = getattr(args, "stop_file", None)
    stop_path = Path(stop_file).expanduser() if stop_file is not None else None
    observation_source = "flexiv_live"
    if stop_path is not None and stop_path.exists():
        if connect_robot:
            raise RuntimeError(f"Stop file exists before hardware connection: {stop_file}")
        raise RuntimeError(f"Stop file exists before inference start: {stop_file}")
    summary_file = _open_summary_jsonl(
        getattr(args, "summary_jsonl", None),
        overwrite=getattr(args, "overwrite_summary_jsonl", False),
    )
    visualization = _start_live_visualization(args, builder)

    try:
        if stop_path is not None and stop_path.exists():
            if connect_robot:
                raise RuntimeError(f"Stop file exists before hardware connection: {stop_file}")
            raise RuntimeError(f"Stop file exists before inference start: {stop_file}")
        if connect_robot:
            connect_started = True
            robot.connect()
            LOGGER.info("Flexiv robot connected; mode=%s robot_debug=%s", args.mode, robot.config.debug)
        else:
            LOGGER.info("Mock live source active; no hardware connection")
        period_s = 1.0 / float(args.rate_hz)
        step_indices = itertools.count() if args.num_steps is None else range(args.num_steps)
        for step_idx in step_indices:
            step_start = time.monotonic()
            if stop_path is not None and stop_path.exists():
                summary = {
                    "step": step_idx,
                    "event": "stop_file_before_inference",
                    "event_reason": f"Stop file exists before inference step: {stop_path}; stop loop",
                    "mode": args.mode,
                    "action_mode": args.action_mode,
                    "observation_source": observation_source,
                }
                robot_summary = _runtime_robot_summary(args, robot)
                if robot_summary is not None:
                    summary["robot"] = robot_summary
                if artifacts is not None:
                    summary["artifacts"] = artifacts
                _add_cycle_timing_summary(summary, step_start, period_s)
                _write_summary(summary, summary_file)
                LOGGER.warning("Stop file exists before step %d: %s", step_idx, stop_path)
                break
            point_cloud_summary = None
            agent_pos_summary = None
            camera_frame_summary = None
            predicted_chunk = False
            visualization_stages = None

            if args.action_mode == "receding" or not action_queue:
                observation = robot.get_observation()
                rgbd_reused_by_adapter = _observation_rgbd_reused(observation, args.camera_name)
                frame = build_pointcloud_frame_from_observation(
                    observation,
                    camera_name=args.camera_name,
                    rgb_key=args.rgb_key,
                    depth_key=args.depth_key,
                )
                camera_frame_summary = _camera_frame_summary(frame)
                camera_frame_identity = _camera_frame_identity(camera_frame_summary)
                previous_camera_frame_identity = last_camera_frame_identity
                rgbd_reused_by_identity = (
                    camera_frame_identity is not None
                    and camera_frame_identity == previous_camera_frame_identity
                )
                if camera_frame_identity is not None:
                    last_camera_frame_identity = camera_frame_identity
                rgbd_reused = bool(rgbd_reused_by_adapter or rgbd_reused_by_identity)
                if rgbd_reused and not args.allow_reused_rgbd:
                    if rgbd_reused_by_adapter:
                        reuse_reason = (
                            f"{args.camera_name} RGB-D frame is marked reused; refusing to run policy."
                        )
                    else:
                        reuse_reason = (
                            f"{args.camera_name} RGB-D frame has same source identity as previous frame; "
                            "refusing to run policy."
                        )
                    summary = {
                        "step": step_idx,
                        "event": "reused_rgbd_before_inference",
                        "event_reason": f"{reuse_reason} Fix camera streaming before inference.",
                        "mode": args.mode,
                        "action_mode": args.action_mode,
                        "observation_source": observation_source,
                        "camera_frame": camera_frame_summary,
                        "rgbd_reused_by_adapter": rgbd_reused_by_adapter,
                        "rgbd_reused_by_identity": rgbd_reused_by_identity,
                        "source_frame_identity": _frame_identity_summary(camera_frame_identity),
                        "previous_source_frame_identity": _frame_identity_summary(
                            previous_camera_frame_identity
                        ),
                    }
                    robot_summary = _runtime_robot_summary(args, robot)
                    if robot_summary is not None:
                        summary["robot"] = robot_summary
                    if artifacts is not None:
                        summary["artifacts"] = artifacts
                    _add_cycle_timing_summary(summary, step_start, period_s)
                    _write_summary(summary, summary_file)
                    raise RuntimeError(f"{reuse_reason} Fix camera streaming before inference.")
                if visualization is None:
                    point_cloud, pc_meta = builder.from_live_frame(frame)
                else:
                    point_cloud, pc_meta, visualization_stages = (
                        builder.from_live_frame_with_stages(frame)
                    )
                pc_meta = _validate_policy_pointcloud_meta(
                    pc_meta,
                    expected_num_points=contract.pointcloud_points,
                )
                agent_pos = build_agent_pos(
                    observation,
                    default_gripper_state=_effective_default_gripper_state(args, robot),
                )
                if connect_robot:
                    _enforce_live_gripper_state_sources(args, robot, observation)
                agent_pos = validate_agent_pos(agent_pos, expected_dim=contract.state_dim)
                agent_pos_summary = _agent_pos_summary(agent_pos, observation=observation)
                pc_np = prepare_point_cloud(
                    point_cloud,
                    expected_num_points=contract.pointcloud_points,
                    expected_dim=contract.pointcloud_dim,
                )
                agent_history.append(agent_pos)
                point_cloud_history.append(pc_np)
                policy_obs = history_to_policy_obs(
                    agent_history,
                    point_cloud_history,
                    n_obs_steps=contract.n_obs_steps,
                    device=device,
                )

                with torch.no_grad():
                    result = policy.predict_action(policy_obs)
                action_seq = _policy_action_sequence(result, contract)
                predicted_at = time.monotonic()
                if args.action_mode == "chunk":
                    action_queue.extend(
                        _QueuedAction(
                            vector=action,
                            predicted_at=predicted_at,
                            chunk_index=idx,
                            chunk_size=len(action_seq),
                            camera_frame_age_ms_at_prediction=_summary_float(
                                camera_frame_summary,
                                "age_ms",
                            ),
                            camera_frame_timestamp=_summary_float(
                                camera_frame_summary,
                                "timestamp",
                            ),
                            camera_frame_wall_time=_summary_float(
                                camera_frame_summary,
                                "wall_time",
                            ),
                            camera_frame_index=_summary_int(
                                camera_frame_summary,
                                "global_frame_index",
                            ),
                            point_cloud_padded=bool(pc_meta.get("padded")),
                        )
                        for idx, action in enumerate(action_seq)
                    )
                else:
                    action_queue.clear()
                    action_queue.append(
                        _QueuedAction(
                            vector=action_seq[0],
                            predicted_at=predicted_at,
                            chunk_index=0,
                            chunk_size=len(action_seq),
                            camera_frame_age_ms_at_prediction=_summary_float(
                                camera_frame_summary,
                                "age_ms",
                            ),
                            camera_frame_timestamp=_summary_float(
                                camera_frame_summary,
                                "timestamp",
                            ),
                            camera_frame_wall_time=_summary_float(
                                camera_frame_summary,
                                "wall_time",
                            ),
                            camera_frame_index=_summary_int(
                                camera_frame_summary,
                                "global_frame_index",
                            ),
                            point_cloud_padded=bool(pc_meta.get("padded")),
                        )
                    )
                predicted_chunk = True
                point_cloud_summary = {
                    "stage": pc_meta.get("stage"),
                    "num_raw_points": pc_meta.get("num_raw_points"),
                    "num_cropped_points": pc_meta.get("num_cropped_points"),
                    "num_sampled_points": pc_meta.get("num_sampled_points"),
                    "num_channels": int(pc_np.shape[1]),
                    "crop_empty": pc_meta.get("crop_empty"),
                    "input_empty": pc_meta.get("input_empty"),
                    "padded": pc_meta.get("padded"),
                    "rgbd_reused": rgbd_reused,
                    "rgbd_reused_by_adapter": rgbd_reused_by_adapter,
                    "rgbd_reused_by_identity": rgbd_reused_by_identity,
                }
                dump_summary = _dump_sampled_pointcloud(
                    args,
                    step_idx=step_idx,
                    point_cloud=pc_np,
                    pointcloud_meta=pc_meta,
                )
                if dump_summary is not None:
                    point_cloud_summary["dump"] = dump_summary
                if visualization is not None and visualization_stages is not None:
                    point_cloud_summary["visualization"] = visualization.maybe_publish(
                        step_idx=step_idx,
                        depth=frame["depth"],
                        stages=visualization_stages,
                        sampled_point_cloud=pc_np,
                        pointcloud_meta=pc_meta,
                    )

            queued_action = action_queue.popleft()
            raw_action = queued_action.vector
            safe_action, diagnostics = filter_action_vector(raw_action, limits)
            action_dict = action_vector_to_flexiv_dict(safe_action)
            now = time.monotonic()
            policy_latency_ms = (now - step_start) * 1000.0
            action_age_ms = (now - queued_action.predicted_at) * 1000.0
            source_camera_frame_summary = _source_camera_frame_summary(queued_action, now)

            summary = {
                "step": step_idx,
                "mode": args.mode,
                "action_mode": args.action_mode,
                "observation_source": observation_source,
                "predicted_chunk": predicted_chunk,
                "chunk_index": queued_action.chunk_index,
                "chunk_size": queued_action.chunk_size,
                "queued_actions_remaining": len(action_queue),
                "policy_output": _policy_output_summary(queued_action, raw_action),
                "action_vector": _action_vector_summary(raw_action, safe_action),
                "flexiv_action": action_dict,
                "raw": summarize_action(raw_action),
                "safe": summarize_action(safe_action),
                "safety": diagnostics,
                "devices": _runtime_devices_summary(args, builder, device),
                "point_cloud": point_cloud_summary,
                "agent_pos": agent_pos_summary,
                "camera_frame": camera_frame_summary,
                "source_camera_frame": source_camera_frame_summary,
                "policy_latency_ms": policy_latency_ms,
                "action_age_ms": action_age_ms,
                "send_duration_ms": None,
                "send_status": "pending",
            }
            robot_summary = _runtime_robot_summary(args, robot)
            if robot_summary is not None:
                summary["robot"] = robot_summary
            if artifacts is not None:
                summary["artifacts"] = artifacts

            if stop_path is not None and stop_path.exists():
                summary["send_status"] = "skipped_stop_file_after_inference"
                summary["send_error"] = f"Stop file exists after inference: {stop_path}; skip send"
                _add_cycle_timing_summary(summary, step_start, period_s)
                _write_summary(summary, summary_file)
                LOGGER.warning("Stop file exists after inference at step %d: %s; skip send", step_idx, stop_path)
                break

            try:
                _enforce_inference_pointcloud_safety(args, queued_action)
            except RuntimeError as exc:
                summary["send_status"] = "skipped_pointcloud_safety"
                summary["send_error"] = str(exc)
                _add_cycle_timing_summary(summary, step_start, period_s)
                _write_summary(summary, summary_file)
                raise

            try:
                _enforce_inference_source_frame_identity(args, source_camera_frame_summary)
                _enforce_timing_safety(
                    args,
                    policy_latency_ms,
                    action_age_ms,
                    source_camera_frame_summary.get("age_ms"),
                )
            except RuntimeError as exc:
                summary["send_status"] = "skipped_timing_safety"
                summary["send_error"] = str(exc)
                _add_cycle_timing_summary(summary, step_start, period_s)
                _write_summary(summary, summary_file)
                raise

            try:
                send_start = time.monotonic()
                robot.send_action(action_dict)
            except Exception as exc:  # noqa: BLE001
                summary["send_duration_ms"] = (time.monotonic() - send_start) * 1000.0
                summary["send_status"] = "send_error"
                summary["send_error"] = str(exc)
                _add_cycle_timing_summary(summary, step_start, period_s)
                _write_summary(summary, summary_file)
                raise
            summary["send_duration_ms"] = (time.monotonic() - send_start) * 1000.0
            summary["send_status"] = "sent"
            _add_cycle_timing_summary(summary, step_start, period_s)
            runtime_safety_error = _inference_runtime_safety_error(args, summary)
            if runtime_safety_error is not None:
                summary["runtime_safety_error"] = runtime_safety_error
                _write_summary(summary, summary_file)
                raise RuntimeError(runtime_safety_error)
            _write_summary(summary, summary_file)

            elapsed = time.monotonic() - step_start
            if elapsed < period_s:
                time.sleep(period_s - elapsed)
    finally:
        active_error = sys.exc_info()[0] is not None
        release_error: Exception | None = None
        try:
            if connect_started:
                try:
                    robot.release()
                    LOGGER.info("Flexiv robot released")
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Flexiv robot release failed")
                    if not active_error:
                        release_error = exc
        finally:
            if visualization is not None:
                visualization.close()
            if summary_file is not None:
                summary_empty = summary_file.tell() == 0
                summary_path = Path(summary_file.name)
                summary_file.close()
                if summary_empty and not getattr(args, "overwrite_summary_jsonl", False):
                    summary_path.unlink(missing_ok=True)
        if release_error is not None:
            raise release_error
    return 0


def _start_live_visualization(
    args: argparse.Namespace,
    builder: PointCloudBuilder,
) -> LiveVisualizationPublisher | None:
    if not bool(getattr(args, "visualize_live", False)):
        return None

    camera = builder.camera
    intrinsics = camera.active_intrinsics
    config = ViewerConfig(
        title=f"DP3 Live Perception | {args.mode}",
        camera_width=int(camera.width),
        camera_height=int(camera.height),
        camera_fx=float(intrinsics.fx),
        camera_fy=float(intrinsics.fy),
        camera_cx=float(intrinsics.cx),
        camera_cy=float(intrinsics.cy),
        depth_scale=float(camera.depth_scale),
        point_size=float(args.visualization_point_size),
    )
    publisher = LiveVisualizationPublisher(
        rate_hz=float(args.visualization_rate_hz),
        max_raw_points=int(args.visualization_max_raw_points),
        max_cropped_points=int(args.visualization_max_cropped_points),
        viewer_config=config,
    )
    try:
        publisher.start()
    except Exception:  # noqa: BLE001
        LOGGER.exception("Live viewer failed to start; inference will continue without it")
        publisher.close()
        return None
    LOGGER.info(
        "Live viewer started pid=%s rate_hz=%.1f raw_display_max=%d cropped_display_max=%d",
        publisher.viewer_pid,
        publisher.rate_hz,
        publisher.max_raw_points,
        publisher.max_cropped_points,
    )
    return publisher


def _enforce_runtime_inference_gates(
    args: argparse.Namespace,
    robot: Any,
    limits: SafetyLimits,
    *,
    connect_robot: bool = False,
) -> None:
    if getattr(args, "mode", None) != "inference":
        return
    robot_config = getattr(robot, "config", None)
    if robot_config is None or getattr(robot_config, "debug", None) is not False:
        raise RuntimeError("--mode inference requires robot.config.debug=False so send_action is active")
    if getattr(robot_config, "use_gripper", True) is not True:
        raise RuntimeError("--mode inference requires robot.config.use_gripper=True so gripper commands are active")
    if bool(getattr(args, "allow_reused_rgbd", False)):
        raise RuntimeError("Inference never sends actions from reused RGB-D frames")
    if getattr(args, "default_gripper_state", None) is not None:
        raise RuntimeError("Inference requires live gripper state")
    if getattr(args, "camera_name", "head_rgb") != "head_rgb":
        raise RuntimeError("--mode inference requires --camera-name head_rgb")
    if getattr(args, "rgb_key", None) is not None or getattr(args, "depth_key", None) is not None:
        raise RuntimeError("Inference requires the default head RGB-D keys")
    if getattr(args, "summary_jsonl", None) is None:
        raise RuntimeError("--mode inference requires --summary-jsonl for action audit logging")
    summary_jsonl = getattr(args, "summary_jsonl", None)
    stop_file = getattr(args, "stop_file", None)
    if summary_jsonl is not None and stop_file is not None and _same_resolved_path(summary_jsonl, stop_file):
        raise RuntimeError(
            "--summary-jsonl and --stop-file must be different paths so the "
            "operator stop signal cannot collide with the audit log"
        )
    for name in (
        "max_policy_latency_ms",
        "max_camera_frame_age_ms",
        "max_action_age_ms",
        "max_send_duration_ms",
        "max_loop_overrun_ms",
    ):
        value = getattr(args, name, None)
        if value is None:
            raise RuntimeError(f"--mode inference requires --{name.replace('_', '-')}")
        _positive_runtime_float(value, label=f"--{name.replace('_', '-')}")
    low_speed_scale = _runtime_finite_float(limits.low_speed_scale, label="inference.low_speed_scale")
    if low_speed_scale <= 0.0 or low_speed_scale > 1.0:
        raise RuntimeError("inference.low_speed_scale must be in (0, 1]")
    _positive_runtime_float(limits.max_cartesian_delta, label="inference.max_cartesian_delta")
    _positive_runtime_float(limits.max_rotation_delta, label="inference.max_rotation_delta")


def _write_summary(summary: dict[str, Any], summary_file: Any) -> None:
    summary_line = json.dumps(summary, ensure_ascii=True, sort_keys=True)
    if summary_file is None or LOGGER.isEnabledFor(logging.DEBUG):
        print(summary_line)
    if summary_file is not None:
        summary_file.write(summary_line + "\n")
        summary_file.flush()


def _positive_runtime_float(value: Any, *, label: str) -> float:
    try:
        return _positive_float(value, label=label)
    except SystemExit as exc:
        message = str(exc)
        if not message:
            message = f"{label} must be positive"
        raise RuntimeError(message) from exc


def _runtime_finite_float(value: Any, *, label: str) -> float:
    try:
        return _finite_float_arg(value, label=label)
    except SystemExit as exc:
        message = str(exc)
        if not message:
            message = f"{label} must be a finite number"
        raise RuntimeError(message) from exc


def _add_cycle_timing_summary(
    summary: dict[str, Any],
    step_start: float,
    target_period_s: float,
) -> None:
    target_period_ms = float(target_period_s) * 1000.0
    cycle_time_ms = (time.monotonic() - step_start) * 1000.0
    summary["target_period_ms"] = target_period_ms
    summary["cycle_time_ms"] = cycle_time_ms
    summary["loop_overrun_ms"] = max(0.0, cycle_time_ms - target_period_ms)


def _open_summary_jsonl(path: Path | None, *, overwrite: bool = False):
    if path is None:
        return None
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved.open("w" if overwrite else "x", encoding="utf-8")


def _dump_sampled_pointcloud(
    args: argparse.Namespace,
    *,
    step_idx: int,
    point_cloud: np.ndarray,
    pointcloud_meta: Mapping[str, Any],
) -> dict[str, Any] | None:
    dump_dir = getattr(args, "sampled_pointcloud_dir", None)
    if dump_dir is None:
        return None
    every = int(getattr(args, "sampled_pointcloud_every", 1))
    if step_idx % every != 0:
        return None

    resolved_dir = Path(dump_dir).expanduser()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    pc = np.asarray(point_cloud, dtype=np.float32)

    step_path = resolved_dir / f"step_{step_idx:06d}_sampled.npy"
    np.save(step_path, pc)
    latest_path = resolved_dir / "latest_sampled.npy"
    latest_tmp_path = resolved_dir / "latest_sampled.npy.tmp"
    with latest_tmp_path.open("wb") as f:
        np.save(f, pc)
    latest_tmp_path.replace(latest_path)

    meta = {
        "step": int(step_idx),
        "shape": [int(value) for value in pc.shape],
        "dtype": str(pc.dtype),
        "point_cloud": {
            "stage": pointcloud_meta.get("stage"),
            "num_raw_points": pointcloud_meta.get("num_raw_points"),
            "num_cropped_points": pointcloud_meta.get("num_cropped_points"),
            "num_sampled_points": pointcloud_meta.get("num_sampled_points"),
            "crop_empty": pointcloud_meta.get("crop_empty"),
            "input_empty": pointcloud_meta.get("input_empty"),
            "padded": pointcloud_meta.get("padded"),
        },
        "npy": step_path.name,
    }
    step_meta_path = resolved_dir / f"step_{step_idx:06d}_sampled_meta.json"
    step_meta_path.write_text(json.dumps(meta, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    latest_meta_path = resolved_dir / "latest_sampled_meta.json"
    latest_meta_tmp_path = resolved_dir / "latest_sampled_meta.json.tmp"
    latest_meta_tmp_path.write_text(
        json.dumps(meta, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    latest_meta_tmp_path.replace(latest_meta_path)
    return {
        "dir": _display_path(resolved_dir),
        "path": _display_path(step_path),
        "latest_path": _display_path(latest_path),
        "every": every,
    }


def _agent_pos_summary(agent_pos: Any, *, observation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    values = agent_pos.reshape(-1)
    summary = {
        "dim": int(values.shape[0]),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "left_gripper_state_norm": float(values[13]) if values.shape[0] > 13 else None,
        "right_gripper_state_norm": float(values[27]) if values.shape[0] > 27 else None,
    }
    if observation is not None:
        for side in ("left", "right"):
            source = observation.get(f"{side}_gripper_state_source")
            if source is not None:
                summary[f"{side}_gripper_state_source"] = str(source)
    return summary


def _validate_policy_pointcloud_meta(
    pointcloud_meta: Any,
    *,
    expected_num_points: int,
) -> dict[str, Any]:
    if not isinstance(pointcloud_meta, Mapping):
        raise RuntimeError("PointCloudBuilder.from_live_frame() metadata must be a mapping")
    meta = dict(pointcloud_meta)
    stage = meta.get("stage")
    if stage != "sampled":
        raise RuntimeError(
            f"PointCloudBuilder.from_live_frame() returned stage {stage!r}; "
            "live policy input must come from the fixed-size sampled point cloud"
        )
    try:
        sampled_points = _optional_non_negative_int(
            meta.get("num_sampled_points"),
            label="point_cloud.num_sampled_points",
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if sampled_points is None:
        raise RuntimeError(
            "PointCloudBuilder.from_live_frame() metadata is missing point_cloud.num_sampled_points"
        )
    meta["num_sampled_points"] = sampled_points
    if sampled_points != int(expected_num_points):
        raise RuntimeError(
            f"PointCloudBuilder sampled {sampled_points} points, "
            f"but the checkpoint expects {int(expected_num_points)}"
        )
    point_counts: dict[str, int] = {}
    for key in ("num_raw_points", "num_cropped_points"):
        if key not in meta:
            raise RuntimeError(
                f"PointCloudBuilder.from_live_frame() metadata is missing point_cloud.{key}"
            )
        try:
            count = _optional_non_negative_int(meta.get(key), label=f"point_cloud.{key}")
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if count is None:
            raise RuntimeError(
                f"PointCloudBuilder.from_live_frame() metadata is missing point_cloud.{key}"
            )
        meta[key] = count
        point_counts[key] = count
    if point_counts["num_cropped_points"] > point_counts["num_raw_points"]:
        raise RuntimeError(
            "PointCloudBuilder metadata has point_cloud.num_cropped_points "
            f"{point_counts['num_cropped_points']} > point_cloud.num_raw_points "
            f"{point_counts['num_raw_points']}"
        )
    for key in ("crop_empty", "input_empty"):
        if key not in meta:
            raise RuntimeError(
                f"PointCloudBuilder.from_live_frame() metadata is missing point_cloud.{key}"
            )
        try:
            is_empty = _bool_scalar(meta.get(key), label=f"point_cloud.{key}")
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if is_empty:
            raise RuntimeError(
                f"PointCloudBuilder reported point_cloud.{key}=true; "
                "refusing to run policy on an empty point cloud"
            )
        meta[key] = is_empty
    if point_counts["num_raw_points"] <= 0:
        raise RuntimeError("point_cloud.num_raw_points must be positive when input_empty=false")
    if point_counts["num_cropped_points"] <= 0:
        raise RuntimeError("point_cloud.num_cropped_points must be positive when crop_empty=false")
    if "padded" not in meta:
        raise RuntimeError("PointCloudBuilder.from_live_frame() metadata is missing point_cloud.padded")
    try:
        meta["padded"] = _bool_scalar(meta.get("padded"), label="point_cloud.padded")
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return meta


def _enforce_live_gripper_state_sources(
    args: argparse.Namespace,
    robot: Any,
    observation: Mapping[str, Any],
) -> None:
    robot_config = getattr(robot, "config", None)
    if robot_config is None or not bool(getattr(robot_config, "use_gripper", True)):
        return
    requires_hardware_width = bool(getattr(args, "mode", None) == "inference") or bool(
        getattr(args, "read_gripper_state", False)
    )
    if not requires_hardware_width:
        return
    for side in ("left", "right"):
        key = f"{side}_gripper_state_source"
        source = observation.get(key)
        if source != "hardware_width":
            raise RuntimeError(
                f"{key} must be hardware_width for inference-ready live inference, got {source!r}; "
                "the observation must prove a hardware width read"
            )


def _camera_frame_summary(frame: dict[str, Any]) -> dict[str, Any]:
    timestamp = _optional_float(frame.get("timestamp"), label="camera_frame.timestamp")
    wall_time = _optional_float(frame.get("wall_time"), label="camera_frame.wall_time")
    global_frame_index = _optional_non_negative_int(
        frame.get("global_frame_index"),
        label="camera_frame.global_frame_index",
    )
    if wall_time is not None:
        age_ms = _wall_time_age_ms(wall_time, label="camera_frame.wall_time")
    else:
        age_ms = None
    return {
        "timestamp": timestamp,
        "wall_time": wall_time,
        "global_frame_index": global_frame_index,
        "age_ms": age_ms,
    }


def _camera_frame_identity(camera_frame: dict[str, Any] | None) -> tuple[str, float] | None:
    if camera_frame is None:
        return None
    timestamp = _optional_float(camera_frame.get("timestamp"), label="camera_frame.timestamp")
    if timestamp is not None:
        return ("timestamp", timestamp)
    wall_time = _optional_float(camera_frame.get("wall_time"), label="camera_frame.wall_time")
    if wall_time is not None:
        return ("wall_time", wall_time)
    global_frame_index = _summary_int(camera_frame, "global_frame_index")
    if global_frame_index is not None:
        return ("global_frame_index", float(global_frame_index))
    return None


def _frame_identity_summary(identity: tuple[str, float] | None) -> dict[str, Any] | None:
    if identity is None:
        return None
    kind, value = identity
    return {"kind": kind, "value": float(value)}


def _source_camera_frame_summary(queued_action: _QueuedAction, now_monotonic: float) -> dict[str, Any]:
    if queued_action.camera_frame_wall_time is not None:
        wall_time = _optional_float(
            queued_action.camera_frame_wall_time,
            label="source_camera_frame.wall_time",
        )
        age_ms = _wall_time_age_ms(wall_time, label="source_camera_frame.wall_time")
    elif queued_action.camera_frame_age_ms_at_prediction is not None:
        recorded_age_ms = _non_negative_float(
            queued_action.camera_frame_age_ms_at_prediction,
            label="Source camera frame recorded age",
        )
        prediction_age_ms = _non_negative_float(
            (float(now_monotonic) - queued_action.predicted_at) * 1000.0,
            label="Source camera frame prediction age",
        )
        age_ms = recorded_age_ms + prediction_age_ms
    else:
        age_ms = None
    return {
        "timestamp": queued_action.camera_frame_timestamp,
        "wall_time": queued_action.camera_frame_wall_time,
        "global_frame_index": queued_action.camera_frame_index,
        "age_ms": age_ms,
    }


def _policy_output_summary(queued_action: _QueuedAction, action: Any) -> dict[str, Any]:
    values = action.reshape(-1) if hasattr(action, "reshape") else torch.as_tensor(action).reshape(-1)
    return {
        "horizon": int(queued_action.chunk_size),
        "selected_index": int(queued_action.chunk_index),
        "action_dim": int(values.shape[0]),
        "action_fields": list(ACTION_FIELD_NAMES),
    }


def _action_vector_summary(raw_action: Any, safe_action: Any) -> dict[str, Any]:
    return {
        "fields": list(ACTION_FIELD_NAMES),
        "raw": _action_values(raw_action),
        "safe": _action_values(safe_action),
    }


def _action_values(action: Any) -> list[float]:
    values = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
    return [float(value) for value in values.tolist()]


def _runtime_devices_summary(
    args: argparse.Namespace,
    builder: PointCloudBuilder,
    policy_device: torch.device,
) -> dict[str, str | None]:
    builder_config = getattr(builder, "config", None)
    return {
        "policy_device": str(policy_device),
        "pointcloud_device": str(getattr(builder, "device", "unknown")),
        "pointcloud_config_device": str(getattr(builder_config, "device", "unknown")),
        "pointcloud_device_override": getattr(args, "pointcloud_device", None),
    }


def _summary_float(summary: dict[str, Any] | None, key: str) -> float | None:
    if summary is None:
        return None
    return _optional_float(summary.get(key), label=key)


def _summary_int(summary: dict[str, Any] | None, key: str) -> int | None:
    if summary is None:
        return None
    return _optional_non_negative_int(summary.get(key), label=key)


def _optional_float(value: Any, *, label: str = "value") -> float | None:
    if value is None:
        return None
    scalar = _scalar_value(value, label=label)
    if isinstance(scalar, (bool, np.bool_, str, bytes)):
        raise ValueError(f"{label} must be a finite float")
    try:
        value_f = float(scalar)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a finite float") from None
    if not math.isfinite(value_f):
        raise ValueError(f"{label} must be a finite float")
    return value_f


def _non_negative_float(value: Any, *, label: str) -> float:
    try:
        value_f = _optional_float(value, label=label)
    except ValueError as exc:
        raise RuntimeError(f"{label} must be a finite non-negative value; skip send") from exc
    if value_f is None or value_f < 0.0:
        raise RuntimeError(f"{label} must be a finite non-negative value; skip send")
    return value_f


def _wall_time_age_ms(wall_time: float, *, label: str) -> float:
    age_s = time.time() - wall_time
    if age_s < -FUTURE_WALL_TIME_TOLERANCE_S:
        raise ValueError(
            f"{label} is {-age_s * 1000.0:.1f} ms in the future; refusing to treat frame as fresh"
        )
    return max(0.0, age_s * 1000.0)


def _optional_non_negative_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    scalar = _scalar_value(value, label=label)
    if isinstance(scalar, (bool, np.bool_, str, bytes)):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        value_f = float(scalar)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a non-negative integer") from None
    if not math.isfinite(value_f) or value_f < 0 or int(value_f) != value_f:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(value_f)


def _bool_scalar(value: Any, *, label: str) -> bool:
    scalar = _scalar_value(value, label=label)
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    raise ValueError(f"{label} must be a boolean scalar")


def _scalar_value(value: Any, *, label: str) -> Any:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.shape == ():
        return array.item()
    flat = array.reshape(-1)
    if flat.size != 1:
        raise ValueError(f"{label} must be a scalar")
    scalar = flat[0]
    return scalar.item() if hasattr(scalar, "item") else scalar


def _policy_action_sequence(result: dict[str, Any], contract: Any):
    if "action" not in result:
        raise RuntimeError("policy.predict_action() result is missing key 'action'")
    action = result["action"]
    if not hasattr(action, "detach"):
        action = torch.as_tensor(action)
    action_np = action.detach().cpu().numpy()
    if action_np.ndim != 3:
        raise RuntimeError(f"policy action must have shape (B,T,D); got {action_np.shape}")
    if action_np.shape[0] != 1:
        raise RuntimeError(f"policy action batch must be 1 for live inference; got {action_np.shape[0]}")
    action_seq = action_np[0]
    if action_seq.shape[0] <= 0:
        raise RuntimeError("policy action sequence is empty")
    if action_seq.shape[1] != int(contract.action_dim):
        raise RuntimeError(
            f"policy action dim {action_seq.shape[1]} != expected {int(contract.action_dim)}"
        )
    return action_seq


def _enforce_timing_safety(
    args: argparse.Namespace,
    policy_latency_ms: float,
    action_age_ms: float,
    camera_frame_age_ms: float | None = None,
) -> None:
    if getattr(args, "mode", None) != "inference":
        return
    policy_latency_ms = _non_negative_float(policy_latency_ms, label="Policy latency")
    action_age_ms = _non_negative_float(action_age_ms, label="Queued action age")
    max_policy_latency_ms = getattr(args, "max_policy_latency_ms", None)
    if max_policy_latency_ms is not None and policy_latency_ms > float(max_policy_latency_ms):
        raise RuntimeError(
            f"Policy latency {policy_latency_ms:.1f} ms exceeded "
            f"--max-policy-latency-ms={float(max_policy_latency_ms):.1f}; skip send"
        )
    max_camera_age_ms = getattr(args, "max_camera_frame_age_ms", None)
    if max_camera_age_ms is not None:
        if camera_frame_age_ms is None:
            raise RuntimeError("Source camera frame age is unavailable; skip send")
        camera_frame_age_ms = _non_negative_float(
            camera_frame_age_ms,
            label="Source camera frame age",
        )
        if float(camera_frame_age_ms) > float(max_camera_age_ms):
            raise RuntimeError(
                f"Source camera frame age {float(camera_frame_age_ms):.1f} ms exceeded "
                f"--max-camera-frame-age-ms={float(max_camera_age_ms):.1f}; skip send"
            )
    max_action_age_ms = getattr(args, "max_action_age_ms", None)
    if max_action_age_ms is not None and action_age_ms > float(max_action_age_ms):
        raise RuntimeError(
            f"Queued action age {action_age_ms:.1f} ms exceeded "
            f"--max-action-age-ms={float(max_action_age_ms):.1f}; skip send"
            )


def _enforce_inference_source_frame_identity(
    args: argparse.Namespace,
    source_camera_frame_summary: Mapping[str, Any] | None,
) -> None:
    if getattr(args, "mode", None) != "inference":
        return
    if _camera_frame_identity(source_camera_frame_summary) is None:
        raise RuntimeError("Source camera frame identity is unavailable; skip send")


def _enforce_inference_pointcloud_safety(
    args: argparse.Namespace,
    queued_action: _QueuedAction,
) -> None:
    if getattr(args, "mode", None) != "inference":
        return
    if queued_action.point_cloud_padded:
        raise RuntimeError(
            "PointCloudBuilder padded the fixed-size policy input; skip send and "
            "adjust crop/camera coverage before inference"
        )


def _enforce_inference_timing_safety(
    args: argparse.Namespace,
    policy_latency_ms: float,
    action_age_ms: float,
    camera_frame_age_ms: float | None = None,
) -> None:
    _enforce_timing_safety(
        args,
        policy_latency_ms,
        action_age_ms,
        camera_frame_age_ms,
    )


def _inference_runtime_safety_error(args: argparse.Namespace, summary: dict[str, Any]) -> str | None:
    if args.mode != "inference" or summary.get("send_status") != "sent":
        return None
    max_send_duration_ms = getattr(args, "max_send_duration_ms", None)
    if max_send_duration_ms is not None:
        send_duration_ms = _non_negative_float(
            summary.get("send_duration_ms"),
            label="send_duration_ms",
        )
        if send_duration_ms > float(max_send_duration_ms):
            return (
                f"send_action duration {send_duration_ms:.1f} ms exceeds "
                f"--max-send-duration-ms={float(max_send_duration_ms):.1f}; stop inference"
            )
    max_loop_overrun_ms = getattr(args, "max_loop_overrun_ms", None)
    if max_loop_overrun_ms is not None:
        loop_overrun_ms = _non_negative_float(
            summary.get("loop_overrun_ms"),
            label="loop_overrun_ms",
        )
        if loop_overrun_ms > float(max_loop_overrun_ms):
            return (
                f"Loop overrun {loop_overrun_ms:.1f} ms exceeds "
                f"--max-loop-overrun-ms={float(max_loop_overrun_ms):.1f}; stop inference"
            )
    return None


def _resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"Requested {device}, but torch.cuda.is_available() is false")
    return torch.device(device)


def _validate_pointcloud_device_arg(device: Any) -> str:
    device_str = str(device).strip()
    if not device_str:
        raise SystemExit("--pointcloud-device must be a non-empty torch device or 'auto'")
    if device_str.lower() == "auto":
        return "auto"
    try:
        torch.device(device_str)
    except (RuntimeError, TypeError):
        raise SystemExit(f"--pointcloud-device is not a valid torch device: {device_str}") from None
    return device_str


def _load_pointcloud_builder(config_path: Path, pointcloud_device: str | None) -> PointCloudBuilder:
    config = load_pointcloud_config(config_path)
    if pointcloud_device is not None:
        config = replace(config, device=pointcloud_device)
    return PointCloudBuilder(config)


def _camera_base_name(camera_name: str) -> str:
    if camera_name.endswith("_rgb"):
        return camera_name.removesuffix("_rgb")
    if camera_name.endswith("_image"):
        return camera_name.removesuffix("_image")
    return camera_name


def _observation_rgbd_reused(observation: dict[str, Any], camera_name: str) -> bool:
    base_name = _camera_base_name(camera_name)
    for key in (f"{base_name}_rgbd_reused", f"{camera_name}_rgbd_reused"):
        if key in observation:
            return _bool_scalar(observation[key], label=key)
    return False


def _validate_builder_contract(builder: PointCloudBuilder, contract: Any) -> None:
    contract_dim = _positive_int(
        getattr(contract, "pointcloud_dim", None),
        label="checkpoint pointcloud_dim",
    )
    contract_points = _positive_int(
        getattr(contract, "pointcloud_points", None),
        label="checkpoint pointcloud_points",
    )
    builder_dim = _builder_output_dim(builder)
    if contract_dim == 6 and builder_dim < 6:
        raise SystemExit(
            "Checkpoint expects xyzrgb point clouds, but the PointCloudBuilder "
            "config has pointcloud.use_rgb=false. Use data_rgb_config.yaml or "
            "another config that outputs RGB channels."
        )
    if contract_dim == 3 and builder_dim > 3:
        LOGGER.warning(
            "Checkpoint expects xyz point clouds but builder outputs xyzrgb; "
            "RGB channels will be truncated before policy inference."
        )
    builder_points = _builder_output_points(builder)
    if builder_points is None:
        raise SystemExit(
            "Checkpoint inference requires PointCloudBuilder sampling.enabled=true "
            "so live point clouds have a fixed point count."
        )
    if builder_points != contract_points:
        raise SystemExit(
            f"Checkpoint expects {contract_points} point-cloud points, "
            f"but PointCloudBuilder sampling.num_points={builder_points}."
        )


def _builder_output_dim(builder: PointCloudBuilder) -> int:
    pointcloud_cfg = getattr(getattr(builder, "config", None), "pointcloud", None)
    return 6 if bool(getattr(pointcloud_cfg, "use_rgb", False)) else 3


def _builder_output_points(builder: PointCloudBuilder) -> int | None:
    sampling_cfg = getattr(getattr(builder, "config", None), "sampling", None)
    if not bool(getattr(sampling_cfg, "enabled", False)):
        return None
    return _positive_int(
        getattr(sampling_cfg, "num_points", None),
        label="PointCloudBuilder sampling.num_points",
    )


def _validate_robot_camera_contract(robot_config: Any, builder: PointCloudBuilder, camera_name: str) -> None:
    camera_cfg = robot_config.cameras.get(camera_name)
    if camera_cfg is None:
        available = ", ".join(sorted(robot_config.cameras)) or "<none>"
        raise SystemExit(
            f"Robot config has no camera named {camera_name!r}. Available cameras: {available}."
        )
    builder_camera = getattr(builder, "camera", None)
    builder_width = _positive_int(
        getattr(builder_camera, "width", None),
        label="PointCloudBuilder camera width",
    )
    builder_height = _positive_int(
        getattr(builder_camera, "height", None),
        label="PointCloudBuilder camera height",
    )
    camera_width = _positive_int(
        getattr(camera_cfg, "width", None),
        label="robot camera width",
    )
    camera_height = _positive_int(
        getattr(camera_cfg, "height", None),
        label="robot camera height",
    )
    if (camera_width, camera_height) != (builder_width, builder_height):
        raise SystemExit(
            f"Robot camera {camera_name!r} is configured as {camera_width}x{camera_height}, "
            f"but PointCloudBuilder expects {builder_width}x{builder_height}. "
            "Use matching --camera-width/--camera-height or a matching point-cloud config."
        )


def _validate_adapter_feature_contract(
    robot: Any,
    camera_name: str,
    *,
    default_gripper_state: float | None = None,
) -> None:
    observation_features = _adapter_feature_mapping(robot, "observation_features")
    action_features = _adapter_feature_mapping(robot, "action_features")
    extra_features = _adapter_feature_mapping(robot, "dataset_extra_features", default={})
    robot_config = getattr(robot, "config", None)
    robot_uses_gripper = bool(getattr(robot_config, "use_gripper", True))
    base_name = _camera_base_name(camera_name)

    missing_obs = []
    for key in STATE_FIELD_NAMES:
        if key in observation_features:
            continue
        if (
            key.endswith("_gripper_state_norm")
            and default_gripper_state is not None
            and not robot_uses_gripper
        ):
            continue
        missing_obs.append(key)
    if missing_obs:
        raise SystemExit(
            "Flexiv adapter observation_features are missing DP3 state fields: "
            + ", ".join(missing_obs)
        )

    missing_action = []
    for key in ACTION_FIELD_NAMES:
        if key in action_features:
            continue
        if key.endswith("_gripper_cmd") and not robot_uses_gripper:
            continue
        missing_action.append(key)
    if missing_action:
        raise SystemExit(
            "Flexiv adapter action_features are missing DP3 action fields: "
            + ", ".join(missing_action)
        )

    if camera_name not in observation_features:
        raise SystemExit(
            f"Flexiv adapter observation_features are missing RGB camera key {camera_name!r}."
        )
    _validate_adapter_feature_shape(
        observation_features[camera_name],
        key=camera_name,
        expected_rank=3,
        expected_last_dim=3,
    )

    depth_key = f"sidecar.{base_name}_depth"
    if depth_key not in extra_features:
        raise SystemExit(
            f"Flexiv adapter dataset_extra_features are missing depth sidecar {depth_key!r}. "
            "The inference script requires save_depth_sidecar=true."
        )
    _validate_adapter_feature_shape(extra_features[depth_key], key=depth_key, expected_rank=2)
    metadata_keys = (
        f"{base_name}_rgbd_timestamp",
        f"{base_name}_rgbd_wall_time",
        f"{base_name}_rgbd_reused",
        "global_frame_index",
    )
    missing_metadata = [key for key in metadata_keys if key not in extra_features]
    if missing_metadata:
        raise SystemExit(
            "Flexiv adapter dataset_extra_features are missing RGB-D metadata fields: "
            + ", ".join(missing_metadata)
            + ". The inference script requires save_rgbd_timestamps=true for stale-frame checks."
        )
    for key in metadata_keys:
        _validate_adapter_feature_shape(extra_features[key], key=key, expected_shape=(1,))


def _adapter_feature_mapping(robot: Any, attr_name: str, *, default: Any = None) -> dict[str, Any]:
    value = getattr(robot, attr_name, default)
    if not isinstance(value, Mapping):
        raise SystemExit(f"Flexiv adapter {attr_name} must be a mapping")
    return dict(value)


def _validate_adapter_feature_shape(
    feature: Any,
    *,
    key: str,
    expected_rank: int | None = None,
    expected_shape: tuple[int, ...] | None = None,
    expected_last_dim: int | None = None,
) -> tuple[int, ...]:
    raw_shape = feature.get("shape") if isinstance(feature, Mapping) else feature
    if not isinstance(raw_shape, (tuple, list)):
        raise SystemExit(f"Flexiv adapter feature {key!r} must declare a shape")
    shape = tuple(
        _positive_int(dim, label=f"Flexiv adapter feature {key!r} shape dim")
        for dim in raw_shape
    )
    if expected_shape is not None and shape != expected_shape:
        raise SystemExit(
            f"Flexiv adapter feature {key!r} shape {shape} != expected {expected_shape}"
        )
    if expected_rank is not None and len(shape) != int(expected_rank):
        raise SystemExit(
            f"Flexiv adapter feature {key!r} rank {len(shape)} != expected {int(expected_rank)}"
        )
    if expected_last_dim is not None and (not shape or shape[-1] != int(expected_last_dim)):
        raise SystemExit(
            f"Flexiv adapter feature {key!r} last dim {shape[-1] if shape else None} "
            f"!= expected {int(expected_last_dim)}"
        )
    return shape


def _add_import_path(path: Path | None) -> None:
    if path is None:
        return
    resolved = Path(path).expanduser()
    if resolved.exists():
        sys.path.insert(0, str(resolved))


def _make_flexiv_robot(args: argparse.Namespace, builder: PointCloudBuilder):
    FlexivDualArmConfig, FlexivDualArm = _load_flexiv_interface(FLEXIV_INTERFACE_DIR)
    config = _load_flexiv_config(FlexivDualArmConfig, args)
    _validate_robot_camera_contract(config, builder, args.camera_name)
    robot = FlexivDualArm(config)
    _validate_adapter_feature_contract(
        robot,
        args.camera_name,
        default_gripper_state=args.default_gripper_state,
    )
    return robot


def _load_flexiv_interface(interface_dir: Path):
    package_name = "_dp3_flexiv_interface"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(interface_dir)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
    try:
        config_mod = importlib.import_module(f"{package_name}.config_flexiv")
        flexiv_mod = importlib.import_module(f"{package_name}.flexiv_dual_arm")
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise RuntimeError(
            "Flexiv adapter import failed because the current environment is missing "
            f"`{missing}`. Run inference in an environment that has both DP3 and "
            "Flexiv/LeRobot runtime dependencies, or install the missing robot-side "
            "packages into the dp3 environment. The script can add LeRobot source "
            "with --lerobot-src, but it cannot supply that source tree's Python "
            "dependencies."
        ) from exc
    return config_mod.FlexivDualArmConfig, flexiv_mod.FlexivDualArm


def _load_flexiv_config(FlexivDualArmConfig: Any, args: argparse.Namespace):
    raw = _load_yaml(args.robot_config)
    robot_raw = dict(_mapping_section(raw, "robot", default_to_root=True))
    camera_raw = dict(_mapping_section(raw, "cameras"))
    field_names = {field.name for field in fields(FlexivDualArmConfig)}
    kwargs = {key: value for key, value in robot_raw.items() if key in field_names and key != "cameras"}

    def set_if_supported(key: str, value: Any) -> None:
        if key in field_names:
            kwargs[key] = value

    robot_debug = bool(args.robot_debug)
    if robot_debug:
        raise SystemExit("Inference requires robot.debug=false so Flexiv send_action is active")
    set_if_supported("debug", bool(robot_debug))
    set_if_supported("control_mode", "oculus")
    set_if_supported("save_depth_sidecar", True)
    set_if_supported("save_rgbd_timestamps", True)
    set_if_supported("save_ir_sidecar", False)
    if getattr(args, "max_cartesian_delta", None) is not None:
        set_if_supported("max_cartesian_delta", float(args.max_cartesian_delta))
    if getattr(args, "max_rotation_delta", None) is not None:
        set_if_supported("max_rotation_delta", float(args.max_rotation_delta))
    connect_motion_allowed = _connect_motion_allowed(args)
    if not connect_motion_allowed:
        set_if_supported("enable_on_connect", False)
        set_if_supported("clear_fault_on_connect", False)
        set_if_supported("go_home_on_connect", False)
        set_if_supported("reset_go_home", False)
        set_if_supported("switch_tool_on_connect", False)
        set_if_supported("initialize_gripper_on_connect", False)
        set_if_supported("open_grippers_on_connect", False)
        set_if_supported("reset_opens_grippers", False)
        set_if_supported("zero_ft_sensor_on_connect", False)
        set_if_supported("switch_cartesian_mode_on_connect", False)
        set_if_supported("use_cartesian_servo_thread", False)
        set_if_supported("camera_hardware_reset_on_connect", False)
        set_if_supported("camera_hardware_reset_on_release", False)
    if _switch_cartesian_mode_allowed(args):
        set_if_supported("switch_cartesian_mode_on_connect", True)
    set_if_supported("read_gripper_state_in_debug", bool(args.read_gripper_state and robot_debug))

    head_serial = args.head_camera_serial or camera_raw.get("head_cam_serial")
    if not head_serial:
        raise SystemExit("Missing head camera serial. Set --head-camera-serial or cameras.head_cam_serial.")
    width = _positive_int(
        args.camera_width if args.camera_width is not None else camera_raw.get("width", 640),
        label="camera width",
    )
    height = _positive_int(
        args.camera_height if args.camera_height is not None else camera_raw.get("height", 480),
        label="camera height",
    )
    fps = _positive_int(args.camera_fps, label="camera fps")
    set_if_supported(
        "cameras",
        {
            args.camera_name: _make_realsense_config(
                serial_number_or_name=str(head_serial),
                width=width,
                height=height,
                fps=fps,
            )
        },
    )
    config = FlexivDualArmConfig(**kwargs)
    if getattr(config, "use_gripper", True) is not True:
        raise SystemExit("Inference requires robot.use_gripper: true in the robot config")
    return config


def _mapping_section(raw: Mapping[str, Any], key: str, *, default_to_root: bool = False) -> Mapping[str, Any]:
    if key not in raw:
        return raw if default_to_root else {}
    value = raw[key]
    if not isinstance(value, Mapping):
        raise SystemExit(f"`{key}` section in robot config must be a mapping")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise SystemExit(f"{label} must be a finite positive integer")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise SystemExit(f"{label} must be a finite positive integer") from None
    value_f = float(value)
    if not math.isfinite(value_f) or value_f <= 0 or int(value_f) != value_f:
        raise SystemExit(f"{label} must be a finite positive integer")
    return int(value_f)


def _finite_float_arg(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise SystemExit(f"{label} must be a finite number")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise SystemExit(f"{label} must be a finite number")
    value_f = float(value)
    if not math.isfinite(value_f):
        raise SystemExit(f"{label} must be a finite number")
    return value_f


def _positive_float(value: Any, *, label: str) -> float:
    value_f = _finite_float_arg(value, label=label)
    if value_f <= 0.0:
        raise SystemExit(f"{label} must be positive")
    return value_f


def _bounded_float(value: Any, *, label: str, minimum: float, maximum: float) -> float:
    value_f = _finite_float_arg(value, label=label)
    if value_f < float(minimum) or value_f > float(maximum):
        raise SystemExit(f"{label} must be in [{minimum:g}, {maximum:g}]")
    return value_f


def _make_realsense_config(*, serial_number_or_name: str, width: int, height: int, fps: int):
    try:
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    except ImportError:  # pragma: no cover - depends on LeRobot version
        from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
    from lerobot.cameras.configs import ColorMode, Cv2Rotation

    return RealSenseCameraConfig(
        serial_number_or_name=serial_number_or_name,
        fps=fps,
        width=width,
        height=height,
        color_mode=ColorMode.RGB,
        use_depth=True,
        use_ir=False,
        rotation=Cv2Rotation.NO_ROTATION,
    )


def _config_check_summary(
    args: argparse.Namespace,
    cfg: Any,
    contract: Any,
    builder: PointCloudBuilder,
    robot_config: Any,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "config_check": True,
        "mode": args.mode,
        "checkpoint": _display_path(args.ckpt),
        "artifacts": artifacts,
        "flexiv_interface": {
            "path": _display_path(FLEXIV_INTERFACE_DIR),
        },
        "task_name": str(_config_value(cfg, "task_name", default="unknown")),
        "n_obs_steps": contract.n_obs_steps,
        "state_dim": contract.state_dim,
        "action_dim": contract.action_dim,
        "point_cloud": {
            "points": contract.pointcloud_points,
            "dim": contract.pointcloud_dim,
            "builder_config": _display_path(args.pointcloud_config),
            "builder_config_device": str(getattr(getattr(builder, "config", None), "device", "unknown")),
            "builder_device": str(builder.device),
            "builder_device_override": args.pointcloud_device,
            "builder_output_dim": _builder_output_dim(builder),
            "builder_output_points": _builder_output_points(builder),
        },
        "action_limits": {
            "low_speed_scale": _optional_config_check_float(args.low_speed_scale),
            "max_cartesian_delta": _optional_config_check_float(args.max_cartesian_delta),
            "max_rotation_delta": _optional_config_check_float(args.max_rotation_delta),
        },
        "robot": _robot_config_summary(args, robot_config, include_cameras=True),
    }
    if args.mode == "inference":
        summary["inference_watchdogs"] = {
            "max_policy_latency_ms": _optional_config_check_float(args.max_policy_latency_ms),
            "max_camera_frame_age_ms": _optional_config_check_float(args.max_camera_frame_age_ms),
            "max_action_age_ms": _optional_config_check_float(args.max_action_age_ms),
            "max_send_duration_ms": _optional_config_check_float(args.max_send_duration_ms),
            "max_loop_overrun_ms": _optional_config_check_float(args.max_loop_overrun_ms),
        }
    return summary


def _runtime_robot_summary(args: argparse.Namespace, robot: Any) -> dict[str, Any] | None:
    robot_config = getattr(robot, "config", None)
    if robot_config is None:
        return None
    return _robot_config_summary(args, robot_config, include_cameras=False)


def _robot_config_summary(
    args: argparse.Namespace,
    robot_config: Any,
    *,
    include_cameras: bool,
) -> dict[str, Any]:
    summary = {
        "debug": bool(getattr(robot_config, "debug", False)),
        "use_gripper": bool(getattr(robot_config, "use_gripper", False)),
        "read_gripper_state_in_debug": bool(
            getattr(robot_config, "read_gripper_state_in_debug", False)
        ),
        "gripper_state_source": _gripper_state_source(args, robot_config),
        "enable_on_connect": bool(getattr(robot_config, "enable_on_connect", False)),
        "clear_fault_on_connect": bool(getattr(robot_config, "clear_fault_on_connect", False)),
        "switch_tool_on_connect": bool(getattr(robot_config, "switch_tool_on_connect", False)),
        "go_home_on_connect": bool(getattr(robot_config, "go_home_on_connect", False)),
        "reset_go_home": bool(getattr(robot_config, "reset_go_home", False)),
        "initialize_gripper_on_connect": bool(
            getattr(robot_config, "initialize_gripper_on_connect", False)
        ),
        "open_grippers_on_connect": bool(
            getattr(robot_config, "open_grippers_on_connect", False)
        ),
        "reset_opens_grippers": bool(
            getattr(robot_config, "reset_opens_grippers", False)
        ),
        "zero_ft_sensor_on_connect": bool(getattr(robot_config, "zero_ft_sensor_on_connect", False)),
        "switch_cartesian_mode_on_connect": bool(
            getattr(robot_config, "switch_cartesian_mode_on_connect", False)
        ),
        "use_cartesian_servo_thread": bool(
            getattr(robot_config, "use_cartesian_servo_thread", False)
        ),
        "camera_hardware_reset_on_connect": bool(
            getattr(robot_config, "camera_hardware_reset_on_connect", False)
        ),
        "camera_hardware_reset_on_release": bool(
            getattr(robot_config, "camera_hardware_reset_on_release", False)
        ),
        "max_cartesian_delta": _optional_robot_float(
            getattr(robot_config, "max_cartesian_delta", None)
        ),
        "max_rotation_delta": _optional_robot_float(
            getattr(robot_config, "max_rotation_delta", None)
        ),
        "save_depth_sidecar": bool(getattr(robot_config, "save_depth_sidecar", False)),
        "save_rgbd_timestamps": bool(getattr(robot_config, "save_rgbd_timestamps", False)),
    }
    if include_cameras:
        cameras = {}
        for name, camera_cfg in getattr(robot_config, "cameras", {}).items():
            cameras[name] = {
                "type": getattr(camera_cfg, "type", camera_cfg.__class__.__name__),
                "width": getattr(camera_cfg, "width", None),
                "height": getattr(camera_cfg, "height", None),
                "fps": getattr(camera_cfg, "fps", None),
                "serial_configured": bool(getattr(camera_cfg, "serial_number_or_name", None)),
                "use_depth": getattr(camera_cfg, "use_depth", None),
            }
        summary["cameras"] = cameras
    return summary


def _optional_config_check_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_robot_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _artifact_audit(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = {
        "checkpoint": _file_audit(args.ckpt),
        "inference_config": _file_audit(args.config_path),
        "pointcloud_config": _file_audit(args.pointcloud_config),
    }
    robot_config = getattr(args, "robot_config", None)
    if robot_config is not None:
        artifacts["robot_config"] = _file_audit(robot_config)
    return artifacts


def _file_audit(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    stat = resolved.stat()
    return {
        "name": resolved.name,
        "size_bytes": int(stat.st_size),
        "sha256": _sha256_file(resolved),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: str | Path) -> str:
    raw = Path(path)
    expanded = raw.expanduser()
    if expanded.is_absolute():
        try:
            return str(expanded.relative_to(REPO_ROOT))
        except ValueError:
            pass
        home = Path.home()
        try:
            return str(Path("~") / expanded.relative_to(home))
        except ValueError:
            return str(expanded)
    return str(raw)


def _same_resolved_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve(strict=False) == Path(right).expanduser().resolve(strict=False)


def _gripper_state_source(args: argparse.Namespace, robot_config: Any) -> str:
    if not bool(getattr(robot_config, "use_gripper", False)):
        if args.default_gripper_state is not None:
            return "cli_default_gripper_state"
        return "missing_without_default_gripper_state"
    if bool(getattr(robot_config, "debug", False)):
        if bool(getattr(robot_config, "read_gripper_state_in_debug", False)):
            return "hardware_width_in_debug"
        return "cached_command_in_debug"
    return "hardware_width"


def _effective_default_gripper_state(args: argparse.Namespace, robot: Any) -> float | None:
    if args.default_gripper_state is None:
        return None
    robot_config = getattr(robot, "config", None)
    if robot_config is None:
        return None
    if bool(getattr(robot_config, "use_gripper", True)):
        return None
    return float(args.default_gripper_state)


def _load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return data


def _config_value(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


if __name__ == "__main__":
    raise SystemExit(main())
