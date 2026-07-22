"""SAPIEN-side two-articulation Flexiv embodiment and bounded numerical IK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .frames import RigidPose, pose_from_sapien
from .gripper_adapter import GripperMapping


@dataclass
class IKResult:
    success: bool
    qpos: np.ndarray
    iterations: int
    position_error_m: float
    rotation_error_rad: float
    reason: str = ""


class SapienFlexivArm:
    def __init__(self, articulation: Any, *, arm_joints: list[str], tcp_link: str, gripper: GripperMapping):
        self.articulation = articulation
        self.arm_joints = arm_joints
        self.tcp_link = tcp_link
        self.gripper = gripper
        active = articulation.get_active_joints()
        self._q_index = {joint.get_name(): index for index, joint in enumerate(active)}
        missing = [name for name in [*arm_joints, gripper.base_joint, *gripper.mimic_joints] if name not in self._q_index]
        if missing:
            raise ValueError(f"active SAPIEN joint mapping is incomplete: {missing}")
        self.arm_indices = [self._q_index[name] for name in arm_joints]

    def qpos(self) -> np.ndarray:
        return np.asarray(self.articulation.get_qpos(), dtype=np.float64).copy()

    def arm_qpos(self) -> np.ndarray:
        return self.qpos()[self.arm_indices]

    def tcp_world(self) -> RigidPose:
        link = self.articulation.find_link_by_name(self.tcp_link)
        if link is None:
            raise ValueError(f"TCP link not found: {self.tcp_link}")
        return pose_from_sapien(link.get_entity_pose())

    def set_arm_drive_targets(self, qpos: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.shape != (7,):
            raise ValueError("arm qpos target must have shape (7,)")
        for name, value in zip(self.arm_joints, qpos):
            joint = self.articulation.find_joint_by_name(name)
            joint.set_drive_target(float(value))
            joint.set_drive_velocity_target(0.0)

    def set_gripper(self, value: float, *, clip: bool = False) -> dict[str, float]:
        targets = self.gripper.normalized_to_joint_targets(value, clip=clip)
        for name, target in targets.items():
            joint = self.articulation.find_joint_by_name(name)
            joint.set_drive_target(float(target))
            joint.set_drive_velocity_target(0.0)
        return targets


class DampedLeastSquaresIK:
    def __init__(self, arm: SapienFlexivArm, *, damping: float = 1e-4, max_iterations: int = 80, position_tolerance: float = 5e-4, rotation_tolerance: float = 5e-3):
        self.arm = arm
        self.damping = float(damping)
        self.max_iterations = int(max_iterations)
        self.position_tolerance = float(position_tolerance)
        self.rotation_tolerance = float(rotation_tolerance)

    def _error(self, target: RigidPose) -> tuple[np.ndarray, float, float]:
        current = self.arm.tcp_world()
        position_error = target.position - current.position
        rotation_error = Rotation.from_matrix(target.rotation @ current.rotation.T).as_rotvec()
        return np.r_[position_error, rotation_error], float(np.linalg.norm(position_error)), float(np.linalg.norm(rotation_error))

    def solve(self, target: RigidPose) -> IKResult:
        original = self.arm.qpos()
        q = original.copy()
        eps = 1e-5
        try:
            for iteration in range(1, self.max_iterations + 1):
                self.arm.articulation.set_qpos(q)
                error, position_norm, rotation_norm = self._error(target)
                if position_norm <= self.position_tolerance and rotation_norm <= self.rotation_tolerance:
                    return IKResult(True, q.copy(), iteration, position_norm, rotation_norm)
                jacobian = np.zeros((6, 7), dtype=np.float64)
                for column, index in enumerate(self.arm.arm_indices):
                    q_probe = q.copy()
                    q_probe[index] += eps
                    self.arm.articulation.set_qpos(q_probe)
                    probe_error, _, _ = self._error(target)
                    jacobian[:, column] = (probe_error - error) / eps
                self.arm.articulation.set_qpos(q)
                # ``jacobian`` above is d(target-current_error)/dq, hence the
                # update solves J*dq = -error.
                step = -jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + self.damping * np.eye(6), error)
                step_norm = float(np.linalg.norm(step))
                if step_norm > 0.20:
                    step *= 0.20 / step_norm
                q[self.arm.arm_indices] += step
                for joint_name, index in zip(self.arm.arm_joints, self.arm.arm_indices):
                    joint = self.arm.articulation.find_joint_by_name(joint_name)
                    limit = np.asarray(joint.get_limits(), dtype=float).reshape(-1)
                    if limit.size >= 2 and np.isfinite(limit[:2]).all():
                        q[index] = np.clip(q[index], limit[0], limit[1])
            self.arm.articulation.set_qpos(q)
            _, position_norm, rotation_norm = self._error(target)
            return IKResult(False, q.copy(), self.max_iterations, position_norm, rotation_norm, "iteration_limit")
        finally:
            self.arm.articulation.set_qpos(original)
