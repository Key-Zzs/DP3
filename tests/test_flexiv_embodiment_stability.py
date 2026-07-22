import os
import sys
from pathlib import Path
import numpy as np
from flexiv_sapien_test_utils import open_flexiv_env

ROOT = Path(__file__).resolve().parents[1]
RMBENCH = ROOT / "third_party/sim/RMBench"
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))
sys.path.insert(0, str(RMBENCH))
os.chdir(RMBENCH)

from envs.flexiv_embodiment_smoke import FlexivEmbodimentSmoke


def test_home_stays_bounded_for_1000_physics_steps_and_resets_repeatably():
    env = open_flexiv_env(FlexivEmbodimentSmoke)
    try:
        env.home(settle_steps=600)
        before = np.r_[env.left_arm.arm_qpos(), env.right_arm.arm_qpos()]
        env._settle(1000)
        after = np.r_[env.left_arm.arm_qpos(), env.right_arm.arm_qpos()]
        assert np.max(np.abs(after - before)) < 0.02
        states = [env.reset(seed=0)[0] for _ in range(3)]
        assert np.max(np.ptp(np.stack(states), axis=0)) < 1e-3
    finally:
        env.close()
