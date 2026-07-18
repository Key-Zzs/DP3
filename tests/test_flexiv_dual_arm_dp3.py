from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))

from diffusion_policy_3d.common.flexiv_state_contract import (  # noqa: E402
    FLEXIV_RAW_FORCE_DROPPED_STATE_NAMES,
)
from diffusion_policy_3d.real_world.flexiv_dual_arm_dp3 import (  # noqa: E402
    ACTION_FIELD_NAMES,
    FLEXIV_ROTATION6D_CONVENTION,
    FLEXIV_STATE_SCHEMA,
    PolicyContract,
    SafetyLimits,
    STATE_FIELD_NAMES,
    action_vector_to_flexiv_dict,
    build_agent_pos,
    configure_policy_action_steps,
    configure_policy_inference_scheduler,
    filter_action_vector,
    validate_flexiv_normalizer_contract,
    validate_agent_pos,
    validate_policy_contract,
)
from diffusion_policy_3d.model.common.normalizer import (  # noqa: E402
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


def test_configure_policy_action_steps_accepts_inference_rollout_override() -> None:
    policy = SimpleNamespace(n_action_steps=7)

    max_action_steps = configure_policy_action_steps(
        policy,
        horizon=8,
        n_obs_steps=2,
        n_action_steps=4,
    )

    assert max_action_steps == 7
    assert policy.n_action_steps == 4


def test_configure_policy_action_steps_rejects_slice_past_horizon() -> None:
    policy = SimpleNamespace(n_action_steps=7)

    with pytest.raises(ValueError, match="horizon - n_obs_steps \\+ 1"):
        configure_policy_action_steps(
            policy,
            horizon=8,
            n_obs_steps=2,
            n_action_steps=8,
        )

    assert policy.n_action_steps == 7


def test_configure_ddim_scheduler_from_checkpoint_schedule() -> None:
    diffusers = pytest.importorskip("diffusers")
    ddpm = diffusers.DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    policy = SimpleNamespace(
        noise_scheduler=ddpm,
        noise_scheduler_pc=ddpm,
    )

    scheduler_class = configure_policy_inference_scheduler(policy, "ddim")

    assert scheduler_class == "DDIMScheduler"
    assert type(policy.noise_scheduler).__name__ == "DDIMScheduler"
    assert type(policy.noise_scheduler_pc).__name__ == "DDIMScheduler"
    assert policy.noise_scheduler.config.num_train_timesteps == 100
    assert policy.noise_scheduler.config.prediction_type == "epsilon"


def test_checkpoint_scheduler_selection_keeps_original_object() -> None:
    scheduler = object()
    policy = SimpleNamespace(noise_scheduler=scheduler)

    assert configure_policy_inference_scheduler(policy, "checkpoint") == "object"
    assert policy.noise_scheduler is scheduler


def test_checkpoint_scheduler_allows_inference_clip_override() -> None:
    diffusers = pytest.importorskip("diffusers")
    ddpm = diffusers.DDPMScheduler(clip_sample=True)
    policy = SimpleNamespace(
        noise_scheduler=ddpm,
        noise_scheduler_pc=ddpm,
    )

    scheduler_class = configure_policy_inference_scheduler(
        policy,
        "checkpoint",
        clip_sample=False,
    )

    assert scheduler_class == "DDPMScheduler"
    assert policy.noise_scheduler.config.clip_sample is False
    assert policy.noise_scheduler_pc.config.clip_sample is False


def test_rejects_unknown_inference_scheduler() -> None:
    policy = SimpleNamespace(noise_scheduler=object())

    with pytest.raises(ValueError, match="Unsupported inference scheduler"):
        configure_policy_inference_scheduler(policy, "unknown")


def _manual_field(scale: list[float], offset: list[float]) -> SingleFieldLinearNormalizer:
    scale_array = np.asarray(scale, dtype=np.float32)
    zeros = np.zeros_like(scale_array)
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale_array,
        offset=np.asarray(offset, dtype=np.float32),
        input_stats_dict={
            "min": zeros.copy(),
            "max": zeros.copy(),
            "mean": zeros.copy(),
            "std": zeros.copy(),
        },
    )


def _policy_with_normalizer(*, legacy_action: bool = False) -> SimpleNamespace:
    normalizer = LinearNormalizer()
    action_scale = (
        [1.0] * 14
        if legacy_action
        else [*([50.0] * 3), *([25.0] * 3), *([50.0] * 3), *([25.0] * 3), 2.0, 2.0]
    )
    action_offset = [0.0] * 12 + ([-1.0, -1.0] if not legacy_action else [0.0, -1.0])
    normalizer["action"] = _manual_field(action_scale, action_offset)
    state_offset = [0.0] * 34
    state_offset[16] = -1.0
    state_offset[33] = -1.0
    normalizer["agent_pos"] = _manual_field(
        [10.0] * 7
        + [20.0] * 3
        + [1.0] * 6
        + [2.0]
        + [10.0] * 7
        + [20.0] * 3
        + [1.0] * 6
        + [2.0],
        state_offset,
    )
    return SimpleNamespace(normalizer=normalizer)


def _validate_normalizer(policy: SimpleNamespace):
    return validate_flexiv_normalizer_contract(
        policy,
        normalizer_schema="flexiv_abs_rot6d_v2",
        state_schema=FLEXIV_STATE_SCHEMA,
        rotation6d_convention=FLEXIV_ROTATION6D_CONVENTION,
        action_rotation_representation="rotvec",
        clip_actions_to_execution_limits=True,
        action_xyz_limit=0.02,
        action_rotation_limit=0.04,
        state_joint_range_floor=0.20,
        state_ee_position_range_floor=0.10,
    )


def test_flexiv_normalizer_contract_accepts_physical_scales() -> None:
    summary = _validate_normalizer(_policy_with_normalizer())

    assert summary["schema"] == "flexiv_abs_rot6d_v2"
    assert summary["max_agent_pos_scale"] == pytest.approx(20.0)
    assert summary["rotation6d_scale"] == pytest.approx(1.0)
    assert summary["rotation6d_offset"] == pytest.approx(0.0)


def test_flexiv_normalizer_contract_rejects_legacy_static_action_scale() -> None:
    with pytest.raises(ValueError, match="action normalizer"):
        _validate_normalizer(_policy_with_normalizer(legacy_action=True))


def test_flexiv_normalizer_contract_rejects_wrong_state_gripper_mapping() -> None:
    policy = _policy_with_normalizer()
    policy.normalizer["agent_pos"].params_dict["offset"][16] = 0.0
    with pytest.raises(ValueError, match="state gripper normalizer"):
        _validate_normalizer(policy)


def test_flexiv_state_field_and_action_contracts_are_34d_and_14d() -> None:
    assert len(STATE_FIELD_NAMES) == 34
    assert len(ACTION_FIELD_NAMES) == 14
    assert STATE_FIELD_NAMES[10:16] == (
        "left_ee_rotation_6d.c0x",
        "left_ee_rotation_6d.c0y",
        "left_ee_rotation_6d.c0z",
        "left_ee_rotation_6d.c1x",
        "left_ee_rotation_6d.c1y",
        "left_ee_rotation_6d.c1z",
    )
    assert ACTION_FIELD_NAMES[3:6] == (
        "left_delta_ee_pose.rx",
        "left_delta_ee_pose.ry",
        "left_delta_ee_pose.rz",
    )


def test_live_agent_pos_checks_rotation6d_and_gripper_indices() -> None:
    observation = {key: 0.0 for key in STATE_FIELD_NAMES}
    for side in ("left", "right"):
        observation[f"{side}_ee_rotation_6d.c0x"] = 1.0
        observation[f"{side}_ee_rotation_6d.c1y"] = 1.0
    observation["left_gripper_state_norm"] = 0.2
    observation["right_gripper_state_norm"] = 0.8
    state = build_agent_pos(observation)
    assert state.shape == (34,)
    assert validate_agent_pos(state).shape == (34,)
    with pytest.raises(ValueError, match="c0 is not unit length"):
        bad = state.copy()
        bad[10] = 0.0
        validate_agent_pos(bad)


def test_live_agent_pos_ignores_v3_force_fields_without_changing_target_state() -> None:
    observation = {key: 0.0 for key in STATE_FIELD_NAMES}
    for side in ("left", "right"):
        observation[f"{side}_ee_rotation_6d.c0x"] = 1.0
        observation[f"{side}_ee_rotation_6d.c1y"] = 1.0
    observation["left_gripper_state_norm"] = 0.2
    observation["right_gripper_state_norm"] = 0.8
    baseline = build_agent_pos(observation)

    with_force = dict(observation)
    with_force.update(
        {
            name: value
            for name, value in zip(
                FLEXIV_RAW_FORCE_DROPPED_STATE_NAMES,
                [
                    1e30,
                    -1e30,
                    7.0,
                    -8.0,
                    1e-30,
                    -1e-30,
                    123.0,
                    -456.0,
                    0.0,
                    1.0,
                    -2.0,
                    3.0,
                    -4.0,
                    5.0,
                ],
                strict=True,
            )
        }
    )
    np.testing.assert_array_equal(build_agent_pos(with_force), baseline)
    with pytest.raises(KeyError, match=r"left_joint_1\.pos"):
        missing = dict(with_force)
        missing.pop("left_joint_1.pos")
        build_agent_pos(missing)


def test_action_contract_and_safety_filter_remain_14d_delta_rotvec() -> None:
    action = np.asarray(
        [
            0.03,
            0.04,
            0.0,
            0.0,
            0.0,
            0.08,
            0.0,
            -0.03,
            0.04,
            0.06,
            0.0,
            0.08,
            -0.5,
            1.5,
        ],
        dtype=np.float32,
    )
    mapped = action_vector_to_flexiv_dict(action)
    assert mapped["left_delta_ee_pose.rx"] == pytest.approx(0.0)
    assert mapped["right_delta_ee_pose.rz"] == pytest.approx(0.08)
    assert mapped["left_gripper_cmd"] == pytest.approx(-0.5)
    safe, _diagnostics = filter_action_vector(
        action,
        SafetyLimits(
            low_speed_scale=1.0,
            max_cartesian_delta=0.02,
            max_rotation_delta=0.04,
        ),
    )
    np.testing.assert_allclose(safe[:3], [0.012, 0.016, 0.0], atol=1e-6)
    np.testing.assert_allclose(safe[3:6], [0.0, 0.0, 0.04], atol=1e-6)
    np.testing.assert_allclose(safe[6:9], [0.0, -0.012, 0.016], atol=1e-6)
    np.testing.assert_allclose(safe[9:12], [0.024, 0.0, 0.032], atol=1e-6)
    np.testing.assert_allclose(safe[12:14], [0.0, 1.0], atol=1e-6)


def test_v1_policy_contract_is_rejected_before_hardware() -> None:
    with pytest.raises(ValueError, match="state_dim must be 34"):
        validate_policy_contract(
            PolicyContract(
                n_obs_steps=2,
                state_dim=28,
                action_dim=14,
                pointcloud_points=1024,
                pointcloud_dim=3,
            )
        )

    with pytest.raises(ValueError, match="state_schema"):
        validate_policy_contract(
            PolicyContract(
                n_obs_steps=2,
                state_dim=34,
                action_dim=14,
                pointcloud_points=1024,
                pointcloud_dim=3,
                state_schema="flexiv_physical_v1",
                state_rotation_representation="absolute_rotvec",
                rotation6d_convention=None,
                action_rotation_representation="rotvec",
            )
        )
