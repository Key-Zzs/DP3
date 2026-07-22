import numpy as np
from scipy.spatial.transform import Rotation

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))

from diffusion_policy_3d.sim.flexiv.action_adapter import FlexivActionAdapter
from diffusion_policy_3d.sim.flexiv.frames import RigidPose


def test_14d_order_limits_and_left_multiply_rotation():
    adapter = FlexivActionAdapter()
    current = RigidPose(np.zeros(3), Rotation.from_euler("y", 0.7).as_matrix())
    action = np.array([0.03, 0, 0, 0, 0, 0.08, 0, 0, 0, 0, 0, 0, 0.5, 0.25], dtype=np.float32)
    decoded = adapter.decode(action, left_current_base=current, right_current_base=current)
    assert decoded.clipped_action.shape == (14,)
    assert np.linalg.norm(decoded.clipped_action[:3]) <= 0.02 + 1e-7
    assert np.linalg.norm(decoded.clipped_action[3:6]) <= 0.04 + 1e-7
    expected = Rotation.from_rotvec(decoded.clipped_action[3:6]).as_matrix() @ current.rotation
    np.testing.assert_allclose(decoded.left_base.rotation, expected, atol=1e-7)
    assert decoded.clipped_action[12:].tolist() == [0.5, 0.25]


def test_action_rejects_out_of_range_without_explicit_clipping():
    adapter = FlexivActionAdapter()
    current = RigidPose(np.zeros(3), np.eye(3))
    try:
        adapter.decode(np.r_[np.zeros(12), 1.2, 0.0], left_current_base=current, right_current_base=current, clip=False)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range gripper command was silently accepted")
