from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))
sys.path.insert(0, str(ROOT / "PointCloudBuilder"))

MODULE_PATH = ROOT / "scripts" / "run_flexiv_dual_arm_dp3_inference.py"
SPEC = importlib.util.spec_from_file_location("flexiv_dp3_inference_ffs_tests", MODULE_PATH)
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
assert SPEC.loader is not None
SPEC.loader.exec_module(launcher)

from diffusion_policy_3d.real_world.flexiv_dual_arm_dp3 import (  # noqa: E402
    PointCloudRuntimeContract,
    build_pointcloud_frame_from_observation,
    pointcloud_runtime_contract_from_builder,
)
from pointcloud_builder.config import load_config  # noqa: E402


H, W = 480, 640


def _contract(
    *,
    depth_source: str = "ffs_stereo",
    output_format: str = "xyz",
    left_key: str = "left_ir",
    right_key: str = "right_ir",
    backend: str = "pytorch",
) -> PointCloudRuntimeContract:
    return PointCloudRuntimeContract(
        depth_source=depth_source,
        output_format=output_format,
        use_rgb=output_format == "xyzrgb",
        num_points=2048,
        camera_name="head",
        ffs_backend=backend if depth_source == "ffs_stereo" else None,
        ffs_left_key=left_key if depth_source == "ffs_stereo" else None,
        ffs_right_key=right_key if depth_source == "ffs_stereo" else None,
        ffs_width=W if depth_source == "ffs_stereo" else None,
        ffs_height=H if depth_source == "ffs_stereo" else None,
        ffs_artifact_id="fp16_o3" if depth_source == "ffs_stereo" else None,
    )


def _observation(*, custom_keys: bool = False) -> dict[str, object]:
    left_key = "custom_left" if custom_keys else "sidecar.head_left_ir"
    right_key = "custom_right" if custom_keys else "sidecar.head_right_ir"
    return {
        "head_rgb": np.zeros((H, W, 3), dtype=np.uint8),
        "sidecar.head_depth": np.full((H, W), 1000, dtype=np.uint16),
        left_key: np.zeros((H, W), dtype=np.uint8),
        right_key: np.ones((H, W), dtype=np.uint8),
        "head_rgbd_timestamp": 12.5,
        "head_rgbd_wall_time": 1_700_000_000.0,
        "head_rgbd_frame_index": 7,
        "global_frame_index": 11,
        "head_left_ir_timestamp": 12.5,
        "head_right_ir_timestamp": 12.5,
        "head_left_ir_frame_index": 7,
        "head_right_ir_frame_index": 7,
    }


def _builder_for_contract(contract: PointCloudRuntimeContract) -> SimpleNamespace:
    ffs = SimpleNamespace(
        backend=contract.ffs_backend,
        left_key=contract.ffs_left_key,
        right_key=contract.ffs_right_key,
        width=contract.ffs_width,
        height=contract.ffs_height,
        artifact_id=contract.ffs_artifact_id,
    )
    config = SimpleNamespace(
        camera=SimpleNamespace(name="head"),
        pointcloud=SimpleNamespace(
            use_rgb=contract.use_rgb,
            output_format=contract.output_format,
        ),
        sampling=SimpleNamespace(enabled=True, num_points=contract.num_points),
        depth_source=SimpleNamespace(
            mode="ffs_stereo" if contract.depth_source == "ffs_stereo" else "frame",
            ffs=ffs if contract.depth_source == "ffs_stereo" else None,
        ),
    )
    return SimpleNamespace(
        config=config,
        camera=SimpleNamespace(name="head", width=W, height=H),
        device="cpu",
    )


def test_native_depth_frame_keeps_depth_and_does_not_require_ir() -> None:
    frame = build_pointcloud_frame_from_observation(
        _observation(),
        camera_name="head_rgb",
        runtime_contract=_contract(depth_source="native_depth"),
    )

    assert set(frame) >= {"depth", "rgb", "timestamp", "global_frame_index"}
    assert "left_ir" not in frame
    assert frame["depth_source"] == "native_depth"


def test_ffs_frame_uses_builder_keys_and_never_inserts_native_depth() -> None:
    frame = build_pointcloud_frame_from_observation(
        _observation(custom_keys=True),
        camera_name="head_rgb",
        runtime_contract=_contract(left_key="custom_left", right_key="custom_right", output_format="xyzrgb"),
    )

    assert set(frame) >= {
        "custom_left",
        "custom_right",
        "rgb",
        "timestamp",
        "global_frame_index",
    }
    assert "depth" not in frame
    assert frame["ffs_backend"] == "pytorch"


@pytest.mark.parametrize("missing", ["left", "right"])
def test_ffs_missing_ir_fails_fast_without_depth_fallback(missing: str) -> None:
    observation = _observation()
    observation.pop(f"sidecar.head_{missing}_ir")

    with pytest.raises(KeyError, match=f"{missing} IR.*ffs_stereo"):
        build_pointcloud_frame_from_observation(
            observation,
            camera_name="head_rgb",
            runtime_contract=_contract(),
        )


@pytest.mark.parametrize("kind", ["shape", "timestamp", "frame_index"])
def test_ffs_pair_shape_and_identity_mismatch_fails_fast(kind: str) -> None:
    observation = _observation()
    if kind == "shape":
        observation["sidecar.head_right_ir"] = np.ones((H, W - 1), dtype=np.uint8)
    elif kind == "timestamp":
        observation["head_right_ir_timestamp"] = 13.5
    else:
        observation["head_right_ir_frame_index"] = 8

    with pytest.raises(ValueError, match="FFS stereo IR"):
        build_pointcloud_frame_from_observation(
            observation,
            camera_name="head_rgb",
            runtime_contract=_contract(),
        )


def test_ffs_accepts_small_cross_stream_timestamp_skew_and_stream_local_indices() -> None:
    observation = _observation()
    rgb_timestamp_ms = 1_784_271_074_134.6362
    ir_timestamp_ms = 1_784_271_074_134.62
    observation["head_rgbd_timestamp"] = rgb_timestamp_ms
    observation["head_left_ir_timestamp"] = ir_timestamp_ms
    observation["head_right_ir_timestamp"] = ir_timestamp_ms
    observation["head_rgbd_frame_index"] = 7
    observation["head_left_ir_frame_index"] = 42
    observation["head_right_ir_frame_index"] = 42

    frame = build_pointcloud_frame_from_observation(
        observation,
        camera_name="head_rgb",
        runtime_contract=_contract(),
    )

    assert frame["left_ir_timestamp"] == ir_timestamp_ms
    assert frame["right_ir_timestamp"] == ir_timestamp_ms
    assert frame["frame_index"] == 7
    assert frame["left_ir_frame_index"] == frame["right_ir_frame_index"] == 42


def test_ffs_rejects_cross_stream_timestamp_skew_beyond_tolerance() -> None:
    observation = _observation()
    observation["head_left_ir_timestamp"] = 13.5
    observation["head_right_ir_timestamp"] = 13.5

    with pytest.raises(ValueError, match="does not match the RGB frame"):
        build_pointcloud_frame_from_observation(
            observation,
            camera_name="head_rgb",
            runtime_contract=_contract(),
        )


@pytest.mark.parametrize(
    "missing_key",
    [
        "head_left_ir_timestamp",
        "head_right_ir_timestamp",
        "head_left_ir_frame_index",
        "head_right_ir_frame_index",
    ],
)
def test_ffs_pair_identity_metadata_is_required(missing_key: str) -> None:
    observation = _observation()
    observation.pop(missing_key)

    with pytest.raises(ValueError, match="FFS stereo IR"):
        build_pointcloud_frame_from_observation(
            observation,
            camera_name="head_rgb",
            runtime_contract=_contract(),
        )


@pytest.mark.parametrize(
    "backend",
    ["pytorch", "tensorrt_single", "tensorrt_two_stage", "tensorrt_plugin"],
)
def test_all_ffs_backends_share_one_builder_runtime_contract_path(backend: str) -> None:
    builder = _builder_for_contract(_contract(backend=backend))

    runtime = pointcloud_runtime_contract_from_builder(builder)

    assert runtime.depth_source == "ffs_stereo"
    assert runtime.ffs_backend == backend
    assert runtime.ffs_left_key == "left_ir"
    assert runtime.ffs_right_key == "right_ir"


def test_ffs_pointcloud_warmup_runs_before_live_frames(
    capsys: pytest.CaptureFixture[str],
) -> None:
    frames: list[dict[str, object]] = []

    def from_live_frame(frame: dict[str, object]):
        frames.append(frame)
        return np.zeros((2048, 3), dtype=np.float32), {}

    builder = SimpleNamespace(
        camera=SimpleNamespace(
            width=W,
            height=H,
            depth_scale=0.001,
            color_intrinsics=SimpleNamespace(width=W, height=H),
        ),
        device=SimpleNamespace(type="cpu"),
        from_live_frame=from_live_frame,
    )

    launcher._warmup_pointcloud_builder(builder, _contract(), steps=2)

    assert len(frames) == 2
    assert frames[0]["left_ir"].shape == (H, W)
    assert frames[0]["right_ir"].shape == (H, W)
    assert "depth" not in frames[0]
    assert "before robot connect" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("output_format", "checkpoint_dim"),
    [("xyz", 3), ("xyzrgb", 6)],
)
def test_checkpoint_and_builder_output_dimensions_must_match(
    output_format: str,
    checkpoint_dim: int,
) -> None:
    builder = _builder_for_contract(_contract(output_format=output_format))
    contract = SimpleNamespace(pointcloud_dim=checkpoint_dim, pointcloud_points=2048)

    launcher._validate_builder_contract(builder, contract)

    wrong = SimpleNamespace(pointcloud_dim=9 - checkpoint_dim, pointcloud_points=2048)
    with pytest.raises(SystemExit, match="does not match"):
        launcher._validate_builder_contract(builder, wrong)


def test_camera_config_is_builder_driven_for_native_and_ffs() -> None:
    native = launcher._make_realsense_config(
        serial_number_or_name="1234",
        width=W,
        height=H,
        fps=30,
        use_ir=False,
    )
    ffs = launcher._make_realsense_config(
        serial_number_or_name="1234",
        width=W,
        height=H,
        fps=30,
        use_ir=True,
    )

    assert native.use_depth is True
    assert native.use_ir is False
    assert ffs.use_depth is True
    assert ffs.use_ir is True


def test_visualization_depth_resolves_rgb_camera_sidecar_key() -> None:
    depth = np.full((H, W), 1234, dtype=np.uint16)
    observation = {
        "sidecar.head_depth": depth,
        "depth": np.zeros((H, W), dtype=np.uint16),
    }

    resolved = launcher._visualization_depth_from_observation(
        observation,
        camera_name="head_rgb",
    )

    assert resolved is depth


def test_visualization_depth_falls_back_to_generic_depth_key() -> None:
    depth = np.full((H, W), 1234, dtype=np.uint16)

    resolved = launcher._visualization_depth_from_observation(
        {"depth": depth},
        camera_name="head_rgb",
    )

    assert resolved is depth


def test_visualization_depth_missing_field_fails_with_actionable_error() -> None:
    with pytest.raises(KeyError, match="save_depth_sidecar=true"):
        launcher._visualization_depth_from_observation(
            {},
            camera_name="head_rgb",
        )


def test_pointcloud_backend_timing_is_preserved_for_runtime_diagnosis() -> None:
    timing = launcher._pointcloud_backend_timing_ms(
        {
            "ffs": {
                "timing_ms": {
                    "inference": 10.5,
                    "sampling": 82.25,
                }
            }
        }
    )

    assert timing == {"inference": 10.5, "sampling": 82.25}


def test_builder_camera_name_must_match_formal_flexiv_camera() -> None:
    builder = _builder_for_contract(_contract())
    builder.camera = SimpleNamespace(name="wrist", width=W, height=H)
    robot_config = SimpleNamespace(
        cameras={"head_rgb": SimpleNamespace(width=W, height=H)},
    )

    with pytest.raises(SystemExit, match="camera.name.*does not match"):
        launcher._validate_robot_camera_contract(robot_config, builder, "head_rgb")


@pytest.mark.parametrize(
    ("config_path", "expected_source", "expected_backend", "expected_format"),
    [
        (
            ROOT / "third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml",
            "ffs_stereo",
            "tensorrt_two_stage",
            "xyz",
        ),
        (
            ROOT / "third_party/real/dual_flexiv_rizon4s/configs/data_rgb_config.yaml",
            "ffs_stereo",
            "tensorrt_plugin",
            "xyzrgb",
        ),
    ],
)
def test_checked_in_builder_yaml_has_a_static_live_contract_and_artifacts(
    config_path: Path,
    expected_source: str,
    expected_backend: str,
    expected_format: str,
) -> None:
    builder = SimpleNamespace(config=load_config(config_path), device="cpu")
    runtime = pointcloud_runtime_contract_from_builder(builder)

    assert runtime.depth_source == expected_source
    assert runtime.ffs_backend == expected_backend
    assert runtime.output_format == expected_format
    assert runtime.num_points == 2048
    audit = launcher._preflight_ffs_artifacts(runtime)
    assert audit["contract"]["backend"] == expected_backend
    assert audit["contract"]["height"] == H
    assert audit["contract"]["width"] == W


def _loader_args(robot_config: Path) -> SimpleNamespace:
    return SimpleNamespace(
        mode="inference",
        ckpt=Path("checkpoint.ckpt"),
        config_path=Path("inference.yaml"),
        pointcloud_config=Path("builder.yaml"),
        robot_config=robot_config,
        camera_name="head_rgb",
        robot_debug=False,
        read_gripper_state=False,
        max_cartesian_delta=0.01,
        max_rotation_delta=0.02,
        enable_on_connect=True,
        clear_fault_on_connect=True,
        go_home_on_connect=True,
        switch_tool_on_connect=True,
        initialize_gripper_on_connect=True,
        switch_cartesian_mode_on_connect=True,
        use_cartesian_servo_thread=True,
        head_camera_serial=None,
        camera_width=None,
        camera_height=None,
        camera_fps=30,
        default_gripper_state=None,
        low_speed_scale=1.0,
        max_policy_latency_ms=250.0,
        max_camera_frame_age_ms=1000.0,
        max_action_age_ms=500.0,
        max_send_duration_ms=500.0,
        max_loop_overrun_ms=250.0,
        max_consecutive_timing_skips=3,
        rate_hz=10.0,
        duration_seconds=None,
        action_mode="chunk",
        n_action_steps=4,
        use_ema=True,
        inference_scheduler="ddim",
        scheduler_clip_sample=True,
        num_inference_steps=10,
        policy_warmup_steps=2,
        pointcloud_warmup_steps=2,
    )


def test_adapter_sidecar_and_camera_features_follow_depth_source(tmp_path: Path) -> None:
    robot_config_path = tmp_path / "robot.yaml"
    robot_config_path.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "left_robot_sn": "left",
                    "right_robot_sn": "right",
                    "use_gripper": True,
                },
                "cameras": {"head_cam_serial": "1234", "width": W, "height": H},
            }
        ),
        encoding="utf-8",
    )
    config_cls, robot_cls = launcher._load_flexiv_interface(launcher.FLEXIV_INTERFACE_DIR)
    args = _loader_args(robot_config_path)

    native_config = launcher._load_flexiv_config(
        config_cls,
        args,
        _contract(depth_source="native_depth"),
    )
    ffs_config = launcher._load_flexiv_config(config_cls, args, _contract())

    assert native_config.save_ir_sidecar is False
    assert native_config.cameras["head_rgb"].use_ir is False
    assert ffs_config.save_ir_sidecar is True
    assert ffs_config.cameras["head_rgb"].use_ir is True

    native_robot = robot_cls(native_config)
    ffs_robot = robot_cls(ffs_config)
    launcher._validate_adapter_feature_contract(
        native_robot,
        "head_rgb",
        runtime_contract=_contract(depth_source="native_depth"),
    )
    launcher._validate_adapter_feature_contract(
        ffs_robot,
        "head_rgb",
        runtime_contract=_contract(),
    )
    assert "sidecar.head_left_ir" in ffs_robot.dataset_extra_features
    assert "head_rgbd_frame_index" in ffs_robot.dataset_extra_features
    assert "head_left_ir_timestamp" in ffs_robot.dataset_extra_features

    observation: dict[str, object] = {}
    ffs_robot._add_camera_frame_observation(
        observation,
        "head_rgb",
        {
            "rgb": np.zeros((H, W, 3), dtype=np.uint8),
            "depth": np.ones((H, W), dtype=np.uint16),
            "left_ir": np.zeros((H, W), dtype=np.uint8),
            "right_ir": np.ones((H, W), dtype=np.uint8),
            "timestamp": 12.5,
            "wall_time": 1_700_000_000.0,
            "frame_index": 7,
            "left_ir_timestamp": 12.5,
            "right_ir_timestamp": 12.5,
            "left_ir_frame_index": 7,
            "right_ir_frame_index": 7,
            "reused": False,
        },
    )
    assert observation["sidecar.head_left_ir"].shape == (H, W)
    assert observation["sidecar.head_right_ir"].shape == (H, W)
    assert observation["head_rgbd_frame_index"] == 7
    assert observation["head_left_ir_frame_index"] == 7


@pytest.mark.parametrize("depth_source", ["native_depth", "ffs_stereo"])
def test_check_config_branch_validates_native_and_ffs_without_connecting(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    depth_source: str,
) -> None:
    robot_config_path = tmp_path / "robot.yaml"
    robot_config_path.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "left_robot_sn": "left",
                    "right_robot_sn": "right",
                    "use_gripper": True,
                },
                "cameras": {"head_cam_serial": "1234", "width": W, "height": H},
            }
        ),
        encoding="utf-8",
    )
    args = _loader_args(robot_config_path)
    runtime = _contract(depth_source=depth_source)
    builder = _builder_for_contract(runtime)
    policy_contract = SimpleNamespace(
        n_obs_steps=2,
        state_dim=34,
        action_dim=14,
        pointcloud_points=2048,
        pointcloud_dim=runtime.pointcloud_dim,
        state_schema=None,
        state_rotation_representation=None,
        state_rotation_reference=None,
        rotation6d_convention=None,
        action_rotation_representation=None,
    )

    assert (
        launcher._run_config_check(
            args=args,
            cfg={},
            contract=policy_contract,
            builder=builder,
            runtime_contract=runtime,
            artifacts={"depth_source": depth_source},
        )
        == 0
    )
    output = yaml.safe_load(capsys.readouterr().out)
    assert output["config_check"] is True
    assert output["point_cloud"]["depth_source"] == depth_source
    camera = output["robot"]["cameras"]["head_rgb"]
    assert camera["use_depth"] is True
    assert camera["use_ir"] is (depth_source == "ffs_stereo")


def test_config_check_summary_reports_native_and_ffs_routes() -> None:
    args = SimpleNamespace(
        mode="inference",
        default_gripper_state=None,
        ckpt=Path("checkpoint.ckpt"),
        config_path=Path("inference.yaml"),
        pointcloud_config=Path("builder.yaml"),
        low_speed_scale=1.0,
        max_cartesian_delta=0.01,
        max_rotation_delta=0.02,
        rate_hz=10.0,
        duration_seconds=None,
        action_mode="chunk",
        n_action_steps=4,
        use_ema=True,
        inference_scheduler="ddim",
        scheduler_clip_sample=True,
        num_inference_steps=10,
        policy_warmup_steps=2,
        pointcloud_warmup_steps=2,
        max_policy_latency_ms=250.0,
        max_camera_frame_age_ms=1000.0,
        max_action_age_ms=500.0,
        max_send_duration_ms=500.0,
        max_loop_overrun_ms=250.0,
        max_consecutive_timing_skips=3,
    )
    robot_config = SimpleNamespace(
        debug=False,
        use_gripper=True,
        cameras={},
        save_depth_sidecar=True,
        save_ir_sidecar=True,
        save_rgbd_timestamps=True,
    )
    policy_contract = SimpleNamespace(
        n_obs_steps=2,
        state_dim=34,
        action_dim=14,
        pointcloud_points=2048,
        pointcloud_dim=6,
        state_schema=None,
        state_rotation_representation=None,
        state_rotation_reference=None,
        rotation6d_convention=None,
        action_rotation_representation=None,
    )
    builder = _builder_for_contract(_contract(output_format="xyzrgb"))

    summary = launcher._config_check_summary(
        args,
        {},
        policy_contract,
        builder,
        robot_config,
        artifacts={"ffs": {"manifest_sha256": "a" * 64}},
    )

    assert summary["point_cloud"]["depth_source"] == "ffs_stereo"
    assert summary["point_cloud"]["ffs_backend"] == "pytorch"
    assert summary["artifacts"]["ffs"]["manifest_sha256"] == "a" * 64
    assert summary["inference_watchdogs"]["max_consecutive_timing_skips"] == 3
