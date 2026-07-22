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

from diffusion_policy_3d.common.flexiv_state_contract import validate_flexiv_state_rotation6d
from envs.flexiv_embodiment_smoke import FlexivEmbodimentSmoke


def test_simulation_state_is_exactly_34d_and_rot6d_is_orthonormal():
    env = open_flexiv_env(FlexivEmbodimentSmoke)
    try:
        state = env.state()
        assert state.shape == (34,)
        assert state.dtype == np.float32
        validate_flexiv_state_rotation6d(state)
        for start in (10, 27):
            c0, c1 = state[start : start + 3], state[start + 3 : start + 6]
            np.testing.assert_allclose(np.linalg.norm(c0), 1.0, atol=1e-4)
            np.testing.assert_allclose(np.linalg.norm(c1), 1.0, atol=1e-4)
            np.testing.assert_allclose(np.dot(c0, c1), 0.0, atol=1e-4)
        assert np.logical_and(state[[16, 33]] >= 0, state[[16, 33]] <= 1).all()
    finally:
        env.close()
