"""Explicit world/base/TCP frame transforms for the Flexiv simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RigidPose:
    position: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64).reshape(-1)
        rotation = np.asarray(self.rotation, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(f"pose position must have shape (3,), got {position.shape}")
        if rotation.shape != (3, 3):
            raise ValueError(f"pose rotation must have shape (3,3), got {rotation.shape}")
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("pose contains NaN or Inf")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "rotation", rotation)


def rotation_from_quat_wxyz(quaternion: Any) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError(f"invalid wxyz quaternion: {q}")
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def rotation_to_quat_wxyz(rotation: Any) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    quat_xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def pose_from_sapien(pose: Any) -> RigidPose:
    return RigidPose(np.asarray(pose.p, dtype=np.float64), rotation_from_quat_wxyz(pose.q))


def base_to_world_pose(base_pose_world: RigidPose, pose_base: RigidPose) -> RigidPose:
    return RigidPose(
        base_pose_world.position + base_pose_world.rotation @ pose_base.position,
        base_pose_world.rotation @ pose_base.rotation,
    )


def world_to_base_pose(base_pose_world: RigidPose, pose_world: RigidPose) -> RigidPose:
    rotation_inv = base_pose_world.rotation.T
    return RigidPose(
        rotation_inv @ (pose_world.position - base_pose_world.position),
        rotation_inv @ pose_world.rotation,
    )


def left_multiply_rotation(current_rotation: Any, delta_rotvec: Any) -> np.ndarray:
    current = np.asarray(current_rotation, dtype=np.float64)
    delta = np.asarray(delta_rotvec, dtype=np.float64).reshape(-1)
    if current.shape != (3, 3) or delta.shape != (3,):
        raise ValueError("current rotation must be (3,3) and delta rotvec must be (3,)")
    return Rotation.from_rotvec(delta).as_matrix() @ current


def rotation_matrix_to_rot6d(rotation: Any) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    return np.concatenate((matrix[:, 0], matrix[:, 1])).astype(np.float32, copy=False)
