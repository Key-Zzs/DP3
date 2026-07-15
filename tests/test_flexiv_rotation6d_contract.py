from __future__ import annotations

import importlib
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
DP3_ROOT = ROOT / "3D-Diffusion-Policy"
sys.path.insert(0, str(DP3_ROOT))

from diffusion_policy_3d.common.flexiv_state_contract import (  # noqa: E402
    flexiv_state_names,
    rdk_pose7_to_absolute_xyz_rot6d,
)


def _load_standalone_adapter():
    package_name = "_test_flexiv_rotation6d_interface"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [
            str(ROOT / "third_party/real/dual_flexiv_rizon4s/interface")
        ]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.flexiv_dual_arm")


adapter = _load_standalone_adapter()


def _pose7(rotation: Rotation, position=(0.1, -0.2, 0.3)) -> np.ndarray:
    quat_xyzw = rotation.as_quat()
    return np.asarray((*position, quat_xyzw[3], *quat_xyzw[:3]), dtype=float)


def test_collection_export_live_rotation6d_parity() -> None:
    pose7 = _pose7(Rotation.from_euler("zyx", [1.1, -0.5, 0.2]))
    canonical = rdk_pose7_to_absolute_xyz_rot6d(pose7)
    live = adapter._pose7_to_absolute_xyz_rot6d(pose7)
    np.testing.assert_allclose(live, canonical, rtol=0.0, atol=1e-7)
    matrix = Rotation.from_quat([pose7[4], pose7[5], pose7[6], pose7[3]]).as_matrix()
    exported = np.concatenate((pose7[:3], matrix[:, 0], matrix[:, 1])).astype(np.float32)
    np.testing.assert_allclose(exported, canonical, rtol=0.0, atol=1e-7)


def test_standalone_adapter_emits_exact_v2_state_fields() -> None:
    class MockStates:
        q = np.arange(7, dtype=float)
        tcp_pose = _pose7(Rotation.from_euler("z", 0.4))

    class MockRobot:
        def states(self):
            return MockStates()

    robot = adapter.FlexivDualArm.__new__(adapter.FlexivDualArm)
    robot._left_robot_lock = threading.Lock()
    robot._num_joints_per_arm = 7
    robot._cached_left_pose7 = np.zeros(7, dtype=float)
    robot.config = SimpleNamespace(
        use_gripper=True,
        gripper_min_width=0.0,
        gripper_max_open=0.1,
    )
    robot._left_gripper_cmd = 1.0
    robot._left_gripper = None
    observation: dict[str, float] = {}
    robot._add_arm_observation(observation, "left", MockRobot())

    state_keys = [key for key in observation if key in flexiv_state_names()]
    assert state_keys == flexiv_state_names()[:17]
    assert not any(name.startswith("left_ee_pose.r") for name in observation)
    np.testing.assert_allclose(
        [observation[name] for name in flexiv_state_names()[10:16]],
        rdk_pose7_to_absolute_xyz_rot6d(MockStates.tcp_pose)[3:],
        atol=1e-7,
    )


def test_adapter_action_contract_remains_14d_delta_rotvec() -> None:
    robot = adapter.FlexivDualArm.__new__(adapter.FlexivDualArm)
    robot._num_joints_per_arm = 7
    robot.config = SimpleNamespace(control_mode="oculus", use_gripper=True)
    assert list(robot.action_features) == list(adapter.ACTION_FIELD_NAMES)
    assert len(robot.action_features) == 14
    assert tuple(adapter.ACTION_FIELD_NAMES[3:6]) == (
        "left_delta_ee_pose.rx",
        "left_delta_ee_pose.ry",
        "left_delta_ee_pose.rz",
    )
