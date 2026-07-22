"""Strict 14D Flexiv Cartesian-delta action adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .frames import RigidPose, base_to_world_pose, left_multiply_rotation


ACTION_DIM = 14
ACTION_XYZ_LIMIT_M = 0.02
ACTION_ROTVEC_LIMIT_RAD = 0.04


def _clip_norm(vector: np.ndarray, limit: float) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm == 0.0:
        return vector.copy(), False
    return vector * (limit / norm), True


@dataclass(frozen=True)
class FlexivActionTargets:
    raw_action: np.ndarray
    clipped_action: np.ndarray
    left_base: RigidPose
    right_base: RigidPose
    diagnostics: dict[str, Any]


class FlexivActionAdapter:
    """Decode the real 14D order without changing frame or rotation semantics."""

    def __init__(self, *, xyz_limit_m: float = ACTION_XYZ_LIMIT_M, rotvec_limit_rad: float = ACTION_ROTVEC_LIMIT_RAD):
        self.xyz_limit_m = float(xyz_limit_m)
        self.rotvec_limit_rad = float(rotvec_limit_rad)

    def decode(
        self,
        action: Any,
        *,
        left_current_base: RigidPose,
        right_current_base: RigidPose,
        clip: bool = True,
    ) -> FlexivActionTargets:
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw.shape != (ACTION_DIM,):
            raise ValueError(f"Flexiv action must have shape (14,), got {raw.shape}")
        if not np.isfinite(raw).all():
            raise ValueError("Flexiv action contains NaN/Inf")
        clipped = raw.copy()
        diagnostics: dict[str, Any] = {"raw_action": raw.tolist(), "xyz_clipped": {}, "rotvec_clipped": {}}
        for label, start, limit, destination in (
            ("left", 0, self.xyz_limit_m, "xyz_clipped"),
            ("right", 6, self.xyz_limit_m, "xyz_clipped"),
            ("left", 3, self.rotvec_limit_rad, "rotvec_clipped"),
            ("right", 9, self.rotvec_limit_rad, "rotvec_clipped"),
        ):
            vector, changed = _clip_norm(clipped[start : start + 3], limit)
            if changed and not clip:
                raise ValueError(f"{label} action vector exceeds limit {limit}")
            clipped[start : start + 3] = vector
            diagnostics[destination][label] = changed
        if not np.logical_and(clipped[12:14] >= 0.0, clipped[12:14] <= 1.0).all():
            if not clip:
                raise ValueError("Flexiv gripper commands must be in [0,1]")
            diagnostics["gripper_clipped"] = True
            clipped[12:14] = np.clip(clipped[12:14], 0.0, 1.0)
        else:
            diagnostics["gripper_clipped"] = False
        left_base = RigidPose(
            left_current_base.position + clipped[0:3],
            left_multiply_rotation(left_current_base.rotation, clipped[3:6]),
        )
        right_base = RigidPose(
            right_current_base.position + clipped[6:9],
            left_multiply_rotation(right_current_base.rotation, clipped[9:12]),
        )
        diagnostics["clipped_action"] = clipped.tolist()
        diagnostics["rotation_composition"] = "R_target = Exp(delta_rotvec) @ R_current"
        return FlexivActionTargets(raw, clipped, left_base, right_base, diagnostics)

    @staticmethod
    def to_world(target: RigidPose, base_pose_world: RigidPose) -> RigidPose:
        return base_to_world_pose(base_pose_world, target)
