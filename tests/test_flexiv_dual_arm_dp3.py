from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))

from diffusion_policy_3d.real_world.flexiv_dual_arm_dp3 import (  # noqa: E402
    configure_policy_action_steps,
    configure_policy_inference_scheduler,
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
