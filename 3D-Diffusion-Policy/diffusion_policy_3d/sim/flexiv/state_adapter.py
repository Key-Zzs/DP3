"""Build the canonical 34D Flexiv simulation state from SAPIEN articulations."""

from __future__ import annotations

from typing import Any

import numpy as np

from diffusion_policy_3d.common.flexiv_state_contract import (
    FLEXIV_STATE_DIM,
    flexiv_state_names,
    rotation_matrix_to_rot6d,
    validate_flexiv_state_rotation6d,
)

from .frames import RigidPose, pose_from_sapien, rotation_matrix_to_rot6d as _rotation_matrix_to_rot6d
from .gripper_adapter import GripperMapping


def _active_qpos(articulation: Any) -> dict[str, float]:
    qpos = np.asarray(articulation.get_qpos(), dtype=np.float64).reshape(-1)
    return {joint.get_name(): float(qpos[index]) for index, joint in enumerate(articulation.get_active_joints())}


class FlexivStateAdapter:
    def __init__(
        self,
        *,
        left_articulation: Any,
        right_articulation: Any,
        left_arm_joints: list[str],
        right_arm_joints: list[str],
        left_tcp_link: str,
        right_tcp_link: str,
        left_gripper: GripperMapping,
        right_gripper: GripperMapping,
        left_base_world: RigidPose,
        right_base_world: RigidPose,
    ):
        if len(left_arm_joints) != 7 or len(right_arm_joints) != 7:
            raise ValueError("Flexiv state requires exactly seven arm joints per side")
        self.left_articulation = left_articulation
        self.right_articulation = right_articulation
        self.left_arm_joints = left_arm_joints
        self.right_arm_joints = right_arm_joints
        self.left_tcp_link = left_tcp_link
        self.right_tcp_link = right_tcp_link
        self.left_gripper = left_gripper
        self.right_gripper = right_gripper
        self.left_base_world = left_base_world
        self.right_base_world = right_base_world

    @staticmethod
    def field_names() -> list[str]:
        return flexiv_state_names()

    @staticmethod
    def _tcp_base(articulation: Any, tcp_link: str, base_world: RigidPose) -> RigidPose:
        link = articulation.find_link_by_name(tcp_link)
        if link is None:
            raise ValueError(f"TCP link not found: {tcp_link}")
        from .frames import world_to_base_pose

        return world_to_base_pose(base_world, pose_from_sapien(link.get_entity_pose()))

    def state(self) -> np.ndarray:
        left_q = _active_qpos(self.left_articulation)
        right_q = _active_qpos(self.right_articulation)
        left_pose = self._tcp_base(self.left_articulation, self.left_tcp_link, self.left_base_world)
        right_pose = self._tcp_base(self.right_articulation, self.right_tcp_link, self.right_base_world)
        left_g = self.left_gripper.base_to_normalized(left_q[self.left_gripper.base_joint])
        right_g = self.right_gripper.base_to_normalized(right_q[self.right_gripper.base_joint])
        state = np.concatenate(
            (
                np.asarray([left_q[name] for name in self.left_arm_joints]),
                left_pose.position,
                _rotation_matrix_to_rot6d(left_pose.rotation),
                [left_g],
                np.asarray([right_q[name] for name in self.right_arm_joints]),
                right_pose.position,
                _rotation_matrix_to_rot6d(right_pose.rotation),
                [right_g],
            )
        ).astype(np.float32, copy=False)
        if state.shape != (FLEXIV_STATE_DIM,):
            raise AssertionError(f"Flexiv simulation state shape drifted: {state.shape}")
        validate_flexiv_state_rotation6d(state, context="Flexiv simulation state")
        if not np.logical_and(state[[16, 33]] >= 0.0, state[[16, 33]] <= 1.0).all():
            raise ValueError("Flexiv simulation gripper state escaped [0,1]")
        if not np.isfinite(state).all():
            raise ValueError("Flexiv simulation state contains NaN/Inf")
        return state

    def describe(self) -> str:
        state = self.state()
        return "\n".join(f"{index:02d} {name} = {value:.7g}" for index, (name, value) in enumerate(zip(self.field_names(), state))) + "\n"
