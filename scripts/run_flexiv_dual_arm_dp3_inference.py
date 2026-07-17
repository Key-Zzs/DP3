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
import site
import sys
import time
import types
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, fields
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

for path in (REPO_ROOT, DP3_ROOT, POINTCLOUD_BUILDER_ROOT):
    if path.exists():
        sys.path.insert(0, str(path))

from diffusion_policy_3d.real_world.flexiv_dual_arm_dp3 import (  # noqa: E402
    ACTION_FIELD_NAMES,
    PointCloudRuntimeContract,
    STATE_FIELD_NAMES,
    SafetyLimits,
    action_vector_to_flexiv_dict,
    build_agent_pos,
    build_pointcloud_frame_from_observation,
    configure_policy_action_steps,
    configure_policy_inference_scheduler,
    filter_action_vector,
    history_to_policy_obs,
    load_dp3_policy_from_checkpoint,
    policy_contract_from_cfg,
    prepare_point_cloud,
    pointcloud_runtime_contract_from_builder,
    summarize_action,
    validate_agent_pos,
    validate_flexiv_normalizer_contract,
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
    camera_frame_source_index: int | None
    camera_frame_left_ir_timestamp: float | None
    camera_frame_right_ir_timestamp: float | None
    camera_frame_left_ir_index: int | None
    camera_frame_right_ir_index: int | None
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
        mode="inference",
        gpu_id=int(inference["gpu_id"]),
        device=str(inference["device"]),
        duration_seconds=duration_seconds,
        num_steps=num_steps,
        rate_hz=rate_hz,
        action_mode=str(inference["action_mode"]),
        n_action_steps=cfg["policy"]["n_action_steps"],
        use_ema=bool(cfg["use_ema"]),
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
        max_consecutive_timing_skips=int(
            inference.get("max_consecutive_timing_skips", 3)
        ),
        i_understand_this_moves_robot=True,
        enable_on_connect=bool(robot["enable_on_connect"]),
        clear_fault_on_connect=bool(robot["clear_fault_on_connect"]),
        go_home_on_connect=bool(robot["go_home_on_connect"]),
        switch_tool_on_connect=bool(robot["switch_tool_on_connect"]),
        initialize_gripper_on_connect=bool(robot["initialize_gripper_on_connect"]),
        switch_cartesian_mode_on_connect=bool(
            robot["switch_cartesian_mode_on_connect"]
        ),
        use_cartesian_servo_thread=bool(robot["use_cartesian_servo_thread"]),
        robot_debug=False,
        inference_scheduler=str(inference.get("scheduler", "checkpoint")),
        scheduler_clip_sample=bool(cfg["policy"]["noise_scheduler"]["clip_sample"]),
        num_inference_steps=int(inference["num_inference_steps"]),
        policy_warmup_steps=int(inference.get("policy_warmup_steps", 2)),
        pointcloud_warmup_steps=int(inference.get("pointcloud_warmup_steps", 2)),
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
    ("obs_as_global_cond", "policy.obs_as_global_cond"),
    ("policy.condition_type", "policy.condition_type"),
    ("policy.use_down_condition", "policy.use_down_condition"),
    ("policy.use_mid_condition", "policy.use_mid_condition"),
    ("policy.use_up_condition", "policy.use_up_condition"),
    ("policy.diffusion_step_embed_dim", "policy.diffusion_step_embed_dim"),
    ("policy.down_dims", "policy.down_dims"),
    ("policy.encoder_output_dim", "policy.encoder_output_dim"),
    ("policy.kernel_size", "policy.kernel_size"),
    ("policy.n_groups", "policy.n_groups"),
    ("policy.noise_scheduler._target_", "policy.noise_scheduler._target_"),
    ("policy.noise_scheduler.num_train_timesteps", "policy.noise_scheduler.num_train_timesteps"),
    ("policy.noise_scheduler.beta_start", "policy.noise_scheduler.beta_start"),
    ("policy.noise_scheduler.beta_end", "policy.noise_scheduler.beta_end"),
    ("policy.noise_scheduler.beta_schedule", "policy.noise_scheduler.beta_schedule"),
    ("policy.noise_scheduler.prediction_type", "policy.noise_scheduler.prediction_type"),
    ("policy.use_pc_color", "policy.use_pc_color"),
    ("policy.pointnet_type", "policy.pointnet_type"),
    ("policy.pointcloud_encoder_cfg.in_channels", "policy.pointcloud_encoder_cfg.in_channels"),
    ("policy.pointcloud_encoder_cfg.out_channels", "policy.pointcloud_encoder_cfg.out_channels"),
    ("policy.pointcloud_encoder_cfg.use_layernorm", "policy.pointcloud_encoder_cfg.use_layernorm"),
    ("policy.pointcloud_encoder_cfg.final_norm", "policy.pointcloud_encoder_cfg.final_norm"),
    ("shape_meta.obs.point_cloud.shape", "shape_meta.obs.point_cloud.shape"),
    ("shape_meta.obs.agent_pos.shape", "shape_meta.obs.agent_pos.shape"),
    ("shape_meta.action.shape", "shape_meta.action.shape"),
)

_FLEXIV_CONTRACT_FIELDS = (
    ("task.dataset.state_schema", "flexiv_contract.state_schema"),
    ("task.dataset.normalizer_schema", "flexiv_contract.normalizer_schema"),
    (
        "task.dataset.state_rotation_representation",
        "flexiv_contract.state_rotation_representation",
    ),
    ("task.dataset.state_rotation_reference", "flexiv_contract.state_rotation_reference"),
    ("task.dataset.rotation6d_convention", "flexiv_contract.rotation6d_convention"),
    (
        "task.dataset.action_rotation_representation",
        "flexiv_contract.action_rotation_representation",
    ),
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
    for checkpoint_key, inference_key in _FLEXIV_CONTRACT_FIELDS:
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
    try:
        policy, cfg, _workspace = load_dp3_policy_from_checkpoint(
            args.ckpt,
            device,
            use_ema=args.use_ema,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _validate_checkpoint_inference_config(cfg, args.inference_config)
    try:
        max_action_steps = configure_policy_action_steps(
            policy,
            horizon=OmegaConf.select(cfg, "horizon"),
            n_obs_steps=OmegaConf.select(cfg, "n_obs_steps"),
            n_action_steps=args.n_action_steps,
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid inference policy.n_action_steps: {exc}") from exc
    scheduler_class = configure_policy_inference_scheduler(
        policy,
        args.inference_scheduler,
        clip_sample=args.scheduler_clip_sample,
    )
    if args.num_inference_steps is not None:
        policy.num_inference_steps = int(args.num_inference_steps)
    contract = policy_contract_from_cfg(cfg)
    validate_policy_contract(contract)
    try:
        normalizer_audit = validate_flexiv_normalizer_contract(
            policy,
            normalizer_schema=OmegaConf.select(cfg, "task.dataset.normalizer_schema"),
            state_schema=OmegaConf.select(cfg, "task.dataset.state_schema"),
            rotation6d_convention=OmegaConf.select(
                cfg,
                "task.dataset.rotation6d_convention",
            ),
            action_rotation_representation=OmegaConf.select(
                cfg,
                "task.dataset.action_rotation_representation",
            ),
            clip_actions_to_execution_limits=OmegaConf.select(
                cfg,
                "task.dataset.clip_actions_to_execution_limits",
            ),
            action_xyz_limit=OmegaConf.select(cfg, "task.dataset.action_xyz_limit"),
            action_rotation_limit=OmegaConf.select(
                cfg,
                "task.dataset.action_rotation_limit",
            ),
            state_joint_range_floor=OmegaConf.select(
                cfg,
                "task.dataset.state_joint_range_floor",
            ),
            state_ee_position_range_floor=OmegaConf.select(
                cfg,
                "task.dataset.state_ee_position_range_floor",
            ),
        )
    except ValueError as exc:
        raise SystemExit(f"Unsafe Flexiv checkpoint normalizer: {exc}") from exc
    LOGGER.info(
        "Loaded checkpoint=%s n_obs_steps=%d n_action_steps=%d (max=%d) "
        "point_cloud=(%d,%d) state_dim=%d action_dim=%d use_ema=%s device=%s "
        "normalizer=%s max_agent_pos_scale=%.6g",
        args.ckpt,
        contract.n_obs_steps,
        policy.n_action_steps,
        max_action_steps,
        contract.pointcloud_points,
        contract.pointcloud_dim,
        contract.state_dim,
        contract.action_dim,
        args.use_ema,
        device,
        normalizer_audit["schema"],
        normalizer_audit["max_agent_pos_scale"],
    )

    if not args.check_config:
        _warmup_policy(
            policy,
            contract,
            device,
            steps=args.policy_warmup_steps,
        )

    builder = _load_pointcloud_builder(args.pointcloud_config)
    try:
        runtime_contract = pointcloud_runtime_contract_from_builder(builder)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid PointCloudBuilder runtime contract: {exc}") from exc
    _validate_builder_contract(builder, contract, runtime_contract)
    if not args.check_config:
        _warmup_pointcloud_builder(
            builder,
            runtime_contract,
            steps=args.pointcloud_warmup_steps,
        )
    LOGGER.info(
        "Loaded PointCloudBuilder config=%s device=%s depth_source=%s output_format=%s "
        "points=%d ffs_backend=%s",
        args.pointcloud_config,
        builder.device,
        runtime_contract.depth_source,
        runtime_contract.output_format,
        runtime_contract.num_points,
        runtime_contract.ffs_backend or "none",
    )
    artifacts = _artifact_audit(args, runtime_contract)

    if args.check_config:
        return _run_config_check(
            args=args,
            cfg=cfg,
            contract=contract,
            builder=builder,
            runtime_contract=runtime_contract,
            artifacts=artifacts,
        )

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
        f"action_mode={args.action_mode} scheduler={scheduler_class} "
        f"diffusion_steps={policy.num_inference_steps} "
        f"n_action_steps={policy.n_action_steps}"
    )
    robot = _make_flexiv_robot(args, builder, runtime_contract)
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
            runtime_contract=runtime_contract,
        )
    except KeyboardInterrupt:
        print("\n[inference] stopped by user; robot cleanup complete")
        return 0


def _run_config_check(
    *,
    args: argparse.Namespace,
    cfg: Any,
    contract: Any,
    builder: PointCloudBuilder,
    runtime_contract: PointCloudRuntimeContract,
    artifacts: dict[str, Any],
) -> int:
    """Validate the complete no-hardware adapter branch used by --check-config."""

    FlexivDualArmConfig, FlexivDualArm = _load_flexiv_interface(FLEXIV_INTERFACE_DIR)
    robot_config = _load_flexiv_config(FlexivDualArmConfig, args, runtime_contract)
    _validate_robot_camera_contract(robot_config, builder, args.camera_name)
    robot_probe = FlexivDualArm(robot_config)
    _validate_adapter_feature_contract(
        robot_probe,
        args.camera_name,
        runtime_contract=runtime_contract,
        default_gripper_state=args.default_gripper_state,
    )
    print(
        json.dumps(
            _config_check_summary(
                args,
                cfg,
                contract,
                builder,
                robot_config,
                artifacts,
                runtime_contract,
            ),
            ensure_ascii=True,
            sort_keys=True,
        )
    )
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
    args.n_action_steps = _positive_int(
        args.n_action_steps,
        label="policy.n_action_steps",
    )
    if args.gpu_id < 0:
        raise SystemExit("inference.gpu_id must be a non-negative integer")
    if args.device.startswith("cuda") and args.device != "cuda:0":
        raise SystemExit(
            "inference.device must be cuda:0 after gpu_id masks the selected physical GPU"
        )
    if args.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise SystemExit("inference.log_level must be DEBUG, INFO, WARNING, or ERROR")
    if args.policy_warmup_steps < 0:
        raise SystemExit("inference.policy_warmup_steps must be a non-negative integer")
    if args.pointcloud_warmup_steps < 0:
        raise SystemExit("inference.pointcloud_warmup_steps must be a non-negative integer")
    args.max_consecutive_timing_skips = _positive_int(
        args.max_consecutive_timing_skips,
        label="inference.max_consecutive_timing_skips",
    )
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
    args.inference_scheduler = str(args.inference_scheduler).strip().lower()
    if args.inference_scheduler not in {"checkpoint", "ddim"}:
        raise SystemExit("inference.scheduler must be 'checkpoint' or 'ddim'")
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
    runtime_contract: PointCloudRuntimeContract | None = None,
) -> int:
    if runtime_contract is None:
        runtime_contract = pointcloud_runtime_contract_from_builder(builder)
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
    consecutive_timing_safety_skips = 0

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
            pointcloud_build_ms = None
            policy_predict_ms = None

            needs_policy_prediction = args.action_mode == "receding" or not action_queue
            if args.action_mode in {"receding", "chunk"}:
                observation = robot.get_observation()
                frame_reused_by_adapter = _observation_rgbd_reused(observation, args.camera_name)
                frame = build_pointcloud_frame_from_observation(
                    observation,
                    camera_name=args.camera_name,
                    rgb_key=args.rgb_key,
                    depth_key=args.depth_key,
                    runtime_contract=runtime_contract,
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
                frame_reused = bool(frame_reused_by_adapter or rgbd_reused_by_identity)
                if frame_reused and not args.allow_reused_rgbd:
                    source_label = (
                        "RGB-D"
                        if runtime_contract.depth_source == "native_depth"
                        else "stereo IR"
                    )
                    if frame_reused_by_adapter:
                        reuse_reason = (
                            f"{args.camera_name} {source_label} frame is marked reused; "
                            "refusing to run policy."
                        )
                    else:
                        reuse_reason = (
                            f"{args.camera_name} {source_label} frame has same source identity as previous frame; "
                            "refusing to run policy."
                        )
                    summary = {
                        "step": step_idx,
                        "event": (
                            "reused_rgbd_before_inference"
                            if runtime_contract.depth_source == "native_depth"
                            else "reused_ffs_stereo_before_inference"
                        ),
                        "event_reason": f"{reuse_reason} Fix camera streaming before inference.",
                        "mode": args.mode,
                        "action_mode": args.action_mode,
                        "observation_source": observation_source,
                        "depth_source": runtime_contract.depth_source,
                        "ffs_backend": runtime_contract.ffs_backend,
                        "camera_frame": camera_frame_summary,
                        "rgbd_reused_by_adapter": frame_reused_by_adapter,
                        "frame_reused_by_adapter": frame_reused_by_adapter,
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
                pointcloud_started = time.monotonic()
                if visualization is None:
                    point_cloud, pc_meta = builder.from_live_frame(frame)
                else:
                    point_cloud, pc_meta, visualization_stages = (
                        builder.from_live_frame_with_stages(frame)
                    )
                pointcloud_build_ms = (time.monotonic() - pointcloud_started) * 1000.0
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
                if needs_policy_prediction:
                    policy_obs = history_to_policy_obs(
                        agent_history,
                        point_cloud_history,
                        n_obs_steps=contract.n_obs_steps,
                        device=device,
                    )

                    policy_started = time.monotonic()
                    with torch.no_grad():
                        result = policy.predict_action(policy_obs)
                    action_seq = _policy_action_sequence(result, contract)
                    policy_predict_ms = (time.monotonic() - policy_started) * 1000.0
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
                                camera_frame_source_index=_summary_int(
                                    camera_frame_summary,
                                    "frame_index",
                                ),
                                camera_frame_left_ir_timestamp=_summary_float(
                                    camera_frame_summary,
                                    "left_ir_timestamp",
                                ),
                                camera_frame_right_ir_timestamp=_summary_float(
                                    camera_frame_summary,
                                    "right_ir_timestamp",
                                ),
                                camera_frame_left_ir_index=_summary_int(
                                    camera_frame_summary,
                                    "left_ir_frame_index",
                                ),
                                camera_frame_right_ir_index=_summary_int(
                                    camera_frame_summary,
                                    "right_ir_frame_index",
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
                                camera_frame_source_index=_summary_int(
                                    camera_frame_summary,
                                    "frame_index",
                                ),
                                camera_frame_left_ir_timestamp=_summary_float(
                                    camera_frame_summary,
                                    "left_ir_timestamp",
                                ),
                                camera_frame_right_ir_timestamp=_summary_float(
                                    camera_frame_summary,
                                    "right_ir_timestamp",
                                ),
                                camera_frame_left_ir_index=_summary_int(
                                    camera_frame_summary,
                                    "left_ir_frame_index",
                                ),
                                camera_frame_right_ir_index=_summary_int(
                                    camera_frame_summary,
                                    "right_ir_frame_index",
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
                    "depth_source": runtime_contract.depth_source,
                    "ffs_backend": runtime_contract.ffs_backend,
                    "build_ms": pointcloud_build_ms,
                    "backend_timing_ms": _pointcloud_backend_timing_ms(pc_meta),
                    "rgbd_reused": frame_reused,
                    "rgbd_reused_by_adapter": frame_reused_by_adapter,
                    "frame_reused_by_adapter": frame_reused_by_adapter,
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
                        depth=_visualization_depth_from_observation(
                            observation,
                            camera_name=args.camera_name,
                        ),
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
                "inference_scheduler": args.inference_scheduler,
                "observation_source": observation_source,
                "depth_source": runtime_contract.depth_source,
                "ffs_backend": runtime_contract.ffs_backend,
                "pointcloud_output_format": runtime_contract.output_format,
                "pointcloud_frame_keys": {
                    "left": runtime_contract.ffs_left_key,
                    "right": runtime_contract.ffs_right_key,
                },
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
                "policy_predict_ms": policy_predict_ms,
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
                action_queue.clear()
                consecutive_timing_safety_skips += 1
                summary["consecutive_timing_safety_skips"] = (
                    consecutive_timing_safety_skips
                )
                summary["timing_safety_recovery"] = (
                    "stop"
                    if consecutive_timing_safety_skips
                    >= args.max_consecutive_timing_skips
                    else "repredict"
                )
                _add_cycle_timing_summary(summary, step_start, period_s)
                _write_summary(summary, summary_file)
                if (
                    consecutive_timing_safety_skips
                    >= args.max_consecutive_timing_skips
                ):
                    raise RuntimeError(
                        f"{exc}; reached "
                        "inference.max_consecutive_timing_skips="
                        f"{args.max_consecutive_timing_skips}"
                    ) from exc
                LOGGER.warning(
                    "Timing safety skipped step %d (%d/%d): %s; "
                    "discard queued chunk and re-predict from a fresh observation",
                    step_idx,
                    consecutive_timing_safety_skips,
                    args.max_consecutive_timing_skips,
                    exc,
                )
                continue

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
            consecutive_timing_safety_skips = 0
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


def _warmup_policy(
    policy: Any,
    contract: Any,
    device: torch.device,
    *,
    steps: int,
) -> None:
    """Initialize policy CUDA kernels before robot connection or action send."""

    if steps <= 0:
        return
    policy_obs = {
        "point_cloud": torch.zeros(
            (
                1,
                int(contract.n_obs_steps),
                int(contract.pointcloud_points),
                int(contract.pointcloud_dim),
            ),
            dtype=torch.float32,
            device=device,
        ),
        "agent_pos": torch.zeros(
            (1, int(contract.n_obs_steps), int(contract.state_dim)),
            dtype=torch.float32,
            device=device,
        ),
    }
    latencies_ms: list[float] = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.monotonic()
        with torch.no_grad():
            policy.predict_action(policy_obs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies_ms.append((time.monotonic() - started) * 1000.0)
    print(
        "[inference] policy warmup before robot connect: "
        + ", ".join(f"{latency:.1f} ms" for latency in latencies_ms)
    )


def _warmup_pointcloud_builder(
    builder: PointCloudBuilder,
    runtime_contract: PointCloudRuntimeContract,
    *,
    steps: int,
) -> None:
    """Initialize PointCloudBuilder and FFS CUDA kernels before robot connection."""

    if steps <= 0:
        return
    camera = builder.camera
    frame: dict[str, Any] = {}
    if runtime_contract.depth_source == "ffs_stereo":
        height = int(runtime_contract.ffs_height)
        width = int(runtime_contract.ffs_width)
        left_key = str(runtime_contract.ffs_left_key)
        right_key = str(runtime_contract.ffs_right_key)
        frame[left_key] = np.zeros((height, width), dtype=np.uint8)
        frame[right_key] = np.zeros((height, width), dtype=np.uint8)
    else:
        height = int(camera.height)
        width = int(camera.width)
        depth_scale = float(camera.depth_scale)
        depth_raw = max(1, int(round(0.6 / depth_scale)))
        frame["depth"] = np.full((height, width), depth_raw, dtype=np.uint16)
    if runtime_contract.use_rgb:
        color = camera.color_intrinsics
        frame["rgb"] = np.zeros(
            (int(color.height), int(color.width), 3),
            dtype=np.uint8,
        )

    latencies_ms: list[float] = []
    for _ in range(steps):
        if builder.device.type == "cuda":
            torch.cuda.synchronize(builder.device)
        started = time.monotonic()
        point_cloud, _ = builder.from_live_frame(frame)
        if builder.device.type == "cuda":
            torch.cuda.synchronize(builder.device)
        expected_shape = (
            int(runtime_contract.num_points),
            int(runtime_contract.pointcloud_dim),
        )
        if tuple(point_cloud.shape) != expected_shape:
            raise RuntimeError(
                "PointCloudBuilder warmup returned shape "
                f"{tuple(point_cloud.shape)}, expected {expected_shape}"
            )
        latencies_ms.append((time.monotonic() - started) * 1000.0)
    print(
        "[inference] point-cloud warmup before robot connect: "
        + ", ".join(f"{latency:.1f} ms" for latency in latencies_ms)
    )


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
    left_rotation = values[10:16]
    right_rotation = values[27:33]
    summary = {
        "dim": int(values.shape[0]),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "left_gripper_state_norm": float(values[16]) if values.shape[0] > 16 else None,
        "right_gripper_state_norm": float(values[33]) if values.shape[0] > 33 else None,
        "left_rotation6d_c0_norm": float(np.linalg.norm(left_rotation[:3])),
        "left_rotation6d_c1_norm": float(np.linalg.norm(left_rotation[3:])),
        "left_rotation6d_c0_c1_dot": float(np.dot(left_rotation[:3], left_rotation[3:])),
        "right_rotation6d_c0_norm": float(np.linalg.norm(right_rotation[:3])),
        "right_rotation6d_c1_norm": float(np.linalg.norm(right_rotation[3:])),
        "right_rotation6d_c0_c1_dot": float(np.dot(right_rotation[:3], right_rotation[3:])),
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
    frame_index = _optional_non_negative_int(
        frame.get("frame_index"),
        label="camera_frame.frame_index",
    )
    left_ir_timestamp = _optional_float(
        frame.get("left_ir_timestamp"),
        label="camera_frame.left_ir_timestamp",
    )
    right_ir_timestamp = _optional_float(
        frame.get("right_ir_timestamp"),
        label="camera_frame.right_ir_timestamp",
    )
    left_ir_frame_index = _optional_non_negative_int(
        frame.get("left_ir_frame_index"),
        label="camera_frame.left_ir_frame_index",
    )
    right_ir_frame_index = _optional_non_negative_int(
        frame.get("right_ir_frame_index"),
        label="camera_frame.right_ir_frame_index",
    )
    if wall_time is not None:
        age_ms = _wall_time_age_ms(wall_time, label="camera_frame.wall_time")
    else:
        age_ms = None
    return {
        "timestamp": timestamp,
        "wall_time": wall_time,
        "global_frame_index": global_frame_index,
        "frame_index": frame_index,
        "left_ir_timestamp": left_ir_timestamp,
        "right_ir_timestamp": right_ir_timestamp,
        "left_ir_frame_index": left_ir_frame_index,
        "right_ir_frame_index": right_ir_frame_index,
        "depth_source": frame.get("depth_source"),
        "ffs_backend": frame.get("ffs_backend"),
        "ffs_left_key": frame.get("ffs_left_key"),
        "ffs_right_key": frame.get("ffs_right_key"),
        "age_ms": age_ms,
    }


def _camera_frame_identity(camera_frame: dict[str, Any] | None) -> tuple[str, float] | None:
    if camera_frame is None:
        return None
    source_frame_index = _optional_non_negative_int(
        camera_frame.get("frame_index"),
        label="camera_frame.frame_index",
    )
    if source_frame_index is not None:
        return ("frame_index", float(source_frame_index))
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
        "frame_index": queued_action.camera_frame_source_index,
        "left_ir_timestamp": queued_action.camera_frame_left_ir_timestamp,
        "right_ir_timestamp": queued_action.camera_frame_right_ir_timestamp,
        "left_ir_frame_index": queued_action.camera_frame_left_ir_index,
        "right_ir_frame_index": queued_action.camera_frame_right_ir_index,
        "age_ms": age_ms,
    }


def _visualization_depth_from_observation(
    observation: Mapping[str, Any],
    *,
    camera_name: str,
) -> Any:
    """Return the diagnostic native depth image used by the existing viewer.

    FFS never receives this field.  It is retained in the adapter observation
    only so the established warmup and monitor remain useful while the builder
    computes its policy depth from stereo IR.
    """

    base_name = _camera_base_name(camera_name)
    key = _first_present_key(
        observation,
        (f"sidecar.{base_name}_depth", f"sidecar.{camera_name}_depth"),
    )
    if key is None:
        if "depth" in observation:
            return observation["depth"]
        raise KeyError(
            f"Missing diagnostic native depth for camera {camera_name!r}; "
            "the live adapter must keep save_depth_sidecar=true for the viewer"
        )
    return observation[key]


def _pointcloud_backend_timing_ms(
    pointcloud_meta: Mapping[str, Any],
) -> dict[str, float] | None:
    ffs = pointcloud_meta.get("ffs")
    if not isinstance(ffs, Mapping):
        return None
    timing = ffs.get("timing_ms")
    if not isinstance(timing, Mapping):
        return None
    return {
        str(key): _non_negative_float(value, label=f"pointcloud timing {key}")
        for key, value in timing.items()
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


def _load_pointcloud_builder(config_path: Path) -> PointCloudBuilder:
    try:
        config = load_pointcloud_config(config_path)
        return PointCloudBuilder(config)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"PointCloudBuilder configuration/backend preflight failed for {config_path}: {exc}"
        ) from exc


def _camera_base_name(camera_name: str) -> str:
    if camera_name.endswith("_rgb"):
        return camera_name.removesuffix("_rgb")
    if camera_name.endswith("_image"):
        return camera_name.removesuffix("_image")
    return camera_name


def _first_present_key(
    mapping: Mapping[str, Any],
    candidates: tuple[str, ...],
) -> str | None:
    """Return the first candidate key present in a mapping."""

    for key in candidates:
        if key in mapping:
            return key
    return None


def _observation_rgbd_reused(observation: dict[str, Any], camera_name: str) -> bool:
    base_name = _camera_base_name(camera_name)
    for key in (f"{base_name}_rgbd_reused", f"{camera_name}_rgbd_reused"):
        if key in observation:
            return _bool_scalar(observation[key], label=key)
    return False


def _validate_builder_contract(
    builder: PointCloudBuilder,
    contract: Any,
    runtime_contract: PointCloudRuntimeContract | None = None,
) -> None:
    runtime_contract = runtime_contract or pointcloud_runtime_contract_from_builder(builder)
    contract_dim = _positive_int(
        getattr(contract, "pointcloud_dim", None),
        label="checkpoint pointcloud_dim",
    )
    contract_points = _positive_int(
        getattr(contract, "pointcloud_points", None),
        label="checkpoint pointcloud_points",
    )
    builder_dim = runtime_contract.pointcloud_dim
    if contract_dim != builder_dim:
        raise SystemExit(
            "Checkpoint point-cloud shape does not match the PointCloudBuilder output: "
            f"checkpoint dim={contract_dim}, Builder output_format={runtime_contract.output_format!r} "
            f"(dim={builder_dim}). Use the matching checkpoint and Builder YAML "
            "(xyz=3, xyzrgb=6)."
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
    builder_camera_name = str(getattr(builder_camera, "name", "")).strip()
    if _camera_base_name(builder_camera_name) != _camera_base_name(camera_name):
        raise SystemExit(
            f"PointCloudBuilder camera.name={builder_camera_name!r} does not match the "
            f"formal Flexiv camera {camera_name!r}. Use the same camera in the Builder "
            "YAML and robot observation contract."
        )
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
            "Use matching robot and PointCloudBuilder YAML camera dimensions."
        )


def _validate_adapter_feature_contract(
    robot: Any,
    camera_name: str,
    *,
    runtime_contract: PointCloudRuntimeContract | None = None,
    default_gripper_state: float | None = None,
) -> None:
    observation_features = _adapter_feature_mapping(robot, "observation_features")
    action_features = _adapter_feature_mapping(robot, "action_features")
    extra_features = _adapter_feature_mapping(robot, "dataset_extra_features", default={})
    robot_config = getattr(robot, "config", None)
    robot_uses_gripper = bool(getattr(robot_config, "use_gripper", True))
    if runtime_contract is None:
        runtime_contract = PointCloudRuntimeContract(
            depth_source="native_depth",
            output_format="xyz",
            use_rgb=False,
            num_points=1,
            camera_name=_camera_base_name(camera_name),
        )
    base_name = _camera_base_name(camera_name)
    state_feature_order = tuple(
        key for key in observation_features if key in set(STATE_FIELD_NAMES)
    )
    if state_feature_order != STATE_FIELD_NAMES:
        raise SystemExit(
            "Flexiv adapter observation_features state order does not match the DP3 v2 "
            f"contract: expected {list(STATE_FIELD_NAMES)!r}, got {list(state_feature_order)!r}"
        )
    action_feature_order = tuple(
        key for key in action_features if key in set(ACTION_FIELD_NAMES)
    )
    if action_feature_order != ACTION_FIELD_NAMES:
        raise SystemExit(
            "Flexiv adapter action_features order does not match the DP3 14D "
            f"contract: expected {list(ACTION_FIELD_NAMES)!r}, got {list(action_feature_order)!r}"
        )

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
    camera_cfg = getattr(robot_config, "cameras", {}).get(camera_name)
    if camera_cfg is None:
        raise SystemExit(f"Flexiv adapter config is missing camera {camera_name!r}")
    if bool(getattr(camera_cfg, "use_depth", False)) is not True:
        raise SystemExit(
            f"Flexiv camera {camera_name!r} must keep use_depth=true for the live "
            "RGB-D warmup/diagnostic contract"
        )
    if runtime_contract.depth_source == "ffs_stereo":
        if bool(getattr(camera_cfg, "use_ir", False)) is not True:
            raise SystemExit(
                f"Flexiv camera {camera_name!r} must set use_ir=true for "
                "depth_source='ffs_stereo'"
            )
        ir_keys = (
            f"sidecar.{base_name}_left_ir",
            f"sidecar.{base_name}_right_ir",
        )
        missing_ir = [key for key in ir_keys if key not in extra_features]
        if missing_ir:
            raise SystemExit(
                "Flexiv adapter dataset_extra_features are missing the FFS stereo IR "
                "sidecar fields: "
                + ", ".join(missing_ir)
                + ". Set save_ir_sidecar=true for the Builder's ffs_stereo route."
            )
        for key in ir_keys:
            _validate_adapter_feature_shape(extra_features[key], key=key, expected_rank=2)
        if runtime_contract.ffs_left_key is None or runtime_contract.ffs_right_key is None:
            raise SystemExit("FFS runtime contract is missing Builder left_key/right_key")
    elif bool(getattr(camera_cfg, "use_ir", False)):
        raise SystemExit(
            f"Flexiv camera {camera_name!r} unexpectedly enables IR while "
            "depth_source='native_depth' is active"
        )
    metadata_keys = (
        f"{base_name}_rgbd_timestamp",
        f"{base_name}_rgbd_wall_time",
        f"{base_name}_rgbd_frame_index",
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
    if runtime_contract.depth_source == "ffs_stereo":
        pair_metadata = (
            f"{base_name}_left_ir_timestamp",
            f"{base_name}_right_ir_timestamp",
            f"{base_name}_left_ir_frame_index",
            f"{base_name}_right_ir_frame_index",
        )
        missing_pair_metadata = [key for key in pair_metadata if key not in extra_features]
        if missing_pair_metadata:
            raise SystemExit(
                "Flexiv adapter dataset_extra_features are missing FFS stereo identity "
                "metadata: "
                + ", ".join(missing_pair_metadata)
            )
        for key in pair_metadata:
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


def _make_flexiv_robot(
    args: argparse.Namespace,
    builder: PointCloudBuilder,
    runtime_contract: PointCloudRuntimeContract | None = None,
):
    runtime_contract = runtime_contract or pointcloud_runtime_contract_from_builder(builder)
    FlexivDualArmConfig, FlexivDualArm = _load_flexiv_interface(FLEXIV_INTERFACE_DIR)
    config = _load_flexiv_config(FlexivDualArmConfig, args, runtime_contract)
    _validate_robot_camera_contract(config, builder, args.camera_name)
    robot = FlexivDualArm(config)
    _validate_adapter_feature_contract(
        robot,
        args.camera_name,
        runtime_contract=runtime_contract,
        default_gripper_state=args.default_gripper_state,
    )
    return robot


def _load_flexiv_interface(interface_dir: Path):
    package_name = _ensure_flexiv_interface_package(interface_dir)
    try:
        config_mod = importlib.import_module(f"{package_name}.config_flexiv")
        flexiv_mod = importlib.import_module(f"{package_name}.flexiv_dual_arm")
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise RuntimeError(
            "Standalone Flexiv adapter import failed because the active environment "
            f"is missing `{missing}`. Install the robot-side dependencies listed in "
            "third_party/real/dual_flexiv_rizon4s/requirements-runtime.txt."
        ) from exc
    return config_mod.FlexivDualArmConfig, flexiv_mod.FlexivDualArm


def _ensure_flexiv_interface_package(interface_dir: Path) -> str:
    package_name = "_dp3_flexiv_interface"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(interface_dir)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
    return package_name


def _load_flexiv_config(
    FlexivDualArmConfig: Any,
    args: argparse.Namespace,
    runtime_contract: PointCloudRuntimeContract | None = None,
):
    if runtime_contract is None:
        runtime_contract = PointCloudRuntimeContract(
            depth_source="native_depth",
            output_format="xyz",
            use_rgb=False,
            num_points=1,
            camera_name=_camera_base_name(args.camera_name),
        )
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
    set_if_supported(
        "save_ir_sidecar",
        runtime_contract.depth_source == "ffs_stereo",
    )
    if getattr(args, "max_cartesian_delta", None) is not None:
        set_if_supported("max_cartesian_delta", float(args.max_cartesian_delta))
    if getattr(args, "max_rotation_delta", None) is not None:
        set_if_supported("max_rotation_delta", float(args.max_rotation_delta))
    for key in (
        "enable_on_connect",
        "clear_fault_on_connect",
        "go_home_on_connect",
        "switch_tool_on_connect",
        "initialize_gripper_on_connect",
        "switch_cartesian_mode_on_connect",
        "use_cartesian_servo_thread",
    ):
        set_if_supported(key, bool(getattr(args, key)))
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
                use_ir=runtime_contract.depth_source == "ffs_stereo",
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


def _make_realsense_config(
    *,
    serial_number_or_name: str,
    width: int,
    height: int,
    fps: int,
    use_ir: bool = False,
):
    package_name = _ensure_flexiv_interface_package(FLEXIV_INTERFACE_DIR)
    camera_mod = importlib.import_module(f"{package_name}.realsense_camera")
    return camera_mod.RealSenseCameraConfig(
        serial_number_or_name=serial_number_or_name,
        fps=fps,
        width=width,
        height=height,
        color_mode=camera_mod.ColorMode.RGB,
        use_depth=True,
        use_ir=bool(use_ir),
        rotation=camera_mod.Cv2Rotation.NO_ROTATION,
    )


def _config_check_summary(
    args: argparse.Namespace,
    cfg: Any,
    contract: Any,
    builder: PointCloudBuilder,
    robot_config: Any,
    artifacts: dict[str, Any] | None = None,
    runtime_contract: PointCloudRuntimeContract | None = None,
) -> dict[str, Any]:
    runtime_contract = runtime_contract or pointcloud_runtime_contract_from_builder(builder)
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
        "state_schema": contract.state_schema,
        "state_rotation_representation": contract.state_rotation_representation,
        "state_rotation_reference": contract.state_rotation_reference,
        "rotation6d_convention": contract.rotation6d_convention,
        "action_rotation_representation": contract.action_rotation_representation,
        "point_cloud": {
            "points": contract.pointcloud_points,
            "dim": contract.pointcloud_dim,
            "depth_source": runtime_contract.depth_source,
            "output_format": runtime_contract.output_format,
            "ffs_backend": runtime_contract.ffs_backend,
            "ffs_left_key": runtime_contract.ffs_left_key,
            "ffs_right_key": runtime_contract.ffs_right_key,
            "builder_config": _display_path(args.pointcloud_config),
            "builder_config_device": str(getattr(getattr(builder, "config", None), "device", "unknown")),
            "builder_device": str(builder.device),
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
        summary["rate_hz"] = float(args.rate_hz)
        summary["duration_seconds"] = args.duration_seconds
        summary["action_mode"] = args.action_mode
        summary["n_action_steps"] = int(args.n_action_steps)
        summary["use_ema"] = bool(args.use_ema)
        summary["inference_scheduler"] = args.inference_scheduler
        summary["scheduler_clip_sample"] = bool(args.scheduler_clip_sample)
        summary["num_inference_steps"] = int(args.num_inference_steps)
        summary["policy_warmup_steps"] = int(args.policy_warmup_steps)
        summary["pointcloud_warmup_steps"] = int(args.pointcloud_warmup_steps)
        summary["inference_watchdogs"] = {
            "max_policy_latency_ms": _optional_config_check_float(args.max_policy_latency_ms),
            "max_camera_frame_age_ms": _optional_config_check_float(args.max_camera_frame_age_ms),
            "max_action_age_ms": _optional_config_check_float(args.max_action_age_ms),
            "max_send_duration_ms": _optional_config_check_float(args.max_send_duration_ms),
            "max_loop_overrun_ms": _optional_config_check_float(args.max_loop_overrun_ms),
            "max_consecutive_timing_skips": int(
                args.max_consecutive_timing_skips
            ),
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
        "save_ir_sidecar": bool(getattr(robot_config, "save_ir_sidecar", False)),
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
                "use_ir": getattr(camera_cfg, "use_ir", None),
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


def _artifact_audit(
    args: argparse.Namespace,
    runtime_contract: PointCloudRuntimeContract | None = None,
) -> dict[str, Any]:
    artifacts = {
        "checkpoint": _file_audit(args.ckpt),
        "inference_config": _file_audit(args.config_path),
        "pointcloud_config": _file_audit(args.pointcloud_config),
    }
    robot_config = getattr(args, "robot_config", None)
    if robot_config is not None:
        artifacts["robot_config"] = _file_audit(robot_config)
    if runtime_contract is not None and runtime_contract.depth_source == "ffs_stereo":
        artifacts["ffs"] = _preflight_ffs_artifacts(runtime_contract)
    return artifacts


def _preflight_ffs_artifacts(runtime_contract: PointCloudRuntimeContract) -> dict[str, Any]:
    """Reuse the exporter/FFS manifest preflight before any robot is created."""

    if runtime_contract.depth_source != "ffs_stereo" or runtime_contract.ffs_config is None:
        return {"depth_source": runtime_contract.depth_source}
    if not runtime_contract.ffs_artifact_id:
        raise SystemExit(
            "FFS Builder YAML must declare depth_source.ffs.artifact_id before live inference"
        )
    try:
        from tools import export_lerobot_to_dp3_zarr as exporter

        ffs = {
            field.name: getattr(runtime_contract.ffs_config, field.name)
            for field in fields(type(runtime_contract.ffs_config))
        }
        return exporter._preflight_ffs_artifacts(
            ffs,
            config_dir=Path.cwd(),
            backend=str(runtime_contract.ffs_backend),
            artifact_id=runtime_contract.ffs_artifact_id,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, OSError) as exc:
        raise SystemExit(f"FFS live artifact preflight failed: {exc}") from exc


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
