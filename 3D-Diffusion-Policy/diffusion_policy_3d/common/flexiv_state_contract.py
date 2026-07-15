"""Canonical Flexiv dual-arm state/action contract helpers.

The state orientation is an absolute RDK TCP orientation represented by the
first two *columns* of its rotation matrix.  Keeping this definition in one
small, dependency-light module prevents exporter, dataset, and inference code
from silently adopting different reshape conventions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


FLEXIV_STATE_SCHEMA = "flexiv_abs_rot6d_v2"
FLEXIV_LEGACY_STATE_SCHEMA = "flexiv_abs_rotvec_v1"
FLEXIV_STATE_DIM = 34
FLEXIV_LEGACY_STATE_DIM = 28
FLEXIV_ACTION_DIM = 14
FLEXIV_STATE_ROTATION_REPRESENTATION = "rotation_6d"
FLEXIV_STATE_ROTATION_REFERENCE = "absolute_rdk_world_base"
FLEXIV_ROTATION6D_CONVENTION = "matrix_columns_0_1"
FLEXIV_ROTATION6D_ORDER = (
    "c0x",
    "c0y",
    "c0z",
    "c1x",
    "c1y",
    "c1z",
)

STATE_POSITION_FIELDS = ("x", "y", "z")
STATE_ROTATION_6D_FIELDS = FLEXIV_ROTATION6D_ORDER
ACTION_DELTA_POSE_FIELDS = ("x", "y", "z", "rx", "ry", "rz")

STATE_JOINT_INDICES = np.asarray(
    [*range(0, 7), *range(17, 24)],
    dtype=np.int64,
)
STATE_EE_POSITION_INDICES = np.asarray(
    [*range(7, 10), *range(24, 27)],
    dtype=np.int64,
)
STATE_EE_ROTATION_6D_INDICES = np.asarray(
    [*range(10, 16), *range(27, 33)],
    dtype=np.int64,
)
STATE_GRIPPER_INDICES = np.asarray([16, 33], dtype=np.int64)

LEGACY_STATE_JOINT_INDICES = np.asarray(
    [*range(0, 7), *range(14, 21)],
    dtype=np.int64,
)
LEGACY_STATE_EE_POSITION_INDICES = np.asarray(
    [*range(7, 10), *range(21, 24)],
    dtype=np.int64,
)
LEGACY_STATE_EE_ROTATION_INDICES = np.asarray(
    [*range(10, 13), *range(24, 27)],
    dtype=np.int64,
)
LEGACY_STATE_GRIPPER_INDICES = np.asarray([13, 27], dtype=np.int64)


def flexiv_state_names() -> list[str]:
    names: list[str] = []
    for side in ("left", "right"):
        names.extend(f"{side}_joint_{index}.pos" for index in range(1, 8))
        names.extend(f"{side}_ee_pose.{axis}" for axis in STATE_POSITION_FIELDS)
        names.extend(
            f"{side}_ee_rotation_6d.{component}"
            for component in STATE_ROTATION_6D_FIELDS
        )
        names.append(f"{side}_gripper_state_norm")
    return names


def flexiv_legacy_state_names() -> list[str]:
    names: list[str] = []
    for side in ("left", "right"):
        names.extend(f"{side}_joint_{index}.pos" for index in range(1, 8))
        names.extend(f"{side}_ee_pose.{axis}" for axis in ACTION_DELTA_POSE_FIELDS)
        names.append(f"{side}_gripper_state_norm")
    return names


def flexiv_action_names() -> list[str]:
    names: list[str] = []
    for side in ("left", "right"):
        names.extend(f"{side}_delta_ee_pose.{axis}" for axis in ACTION_DELTA_POSE_FIELDS)
    names.extend(("left_gripper_cmd", "right_gripper_cmd"))
    return names


def build_flexiv_state_schema() -> dict[str, Any]:
    """Return the complete JSON-serializable v2 metadata contract."""

    return {
        "state_schema": FLEXIV_STATE_SCHEMA,
        "state_dim": FLEXIV_STATE_DIM,
        "action_dim": FLEXIV_ACTION_DIM,
        "state_rotation_representation": FLEXIV_STATE_ROTATION_REPRESENTATION,
        "state_rotation_reference": FLEXIV_STATE_ROTATION_REFERENCE,
        "rotation6d_convention": FLEXIV_ROTATION6D_CONVENTION,
        "rotation6d_order": list(FLEXIV_ROTATION6D_ORDER),
        "action_rotation_representation": "rotvec",
        "state_names": flexiv_state_names(),
        "action_names": flexiv_action_names(),
    }


def build_flexiv_legacy_state_schema() -> dict[str, Any]:
    """Return metadata used to identify the supported 28D legacy source."""

    return {
        "state_schema": FLEXIV_LEGACY_STATE_SCHEMA,
        "state_dim": FLEXIV_LEGACY_STATE_DIM,
        "action_dim": FLEXIV_ACTION_DIM,
        "state_rotation_representation": "absolute_rotvec",
        "state_rotation_reference": FLEXIV_STATE_ROTATION_REFERENCE,
        "state_names": flexiv_legacy_state_names(),
        "action_names": flexiv_action_names(),
    }


def rotation_matrix_to_rot6d(rotation_matrix: Any) -> np.ndarray:
    """Convert a 3x3 matrix to ``[R[:, 0], R[:, 1]]`` without reshaping."""

    matrix = np.asarray(rotation_matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in (3, 3), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("rotation matrix contains NaN or Inf")
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1).astype(
        np.float32,
        copy=False,
    )


def rdk_pose7_to_absolute_xyz_rot6d(pose7: Any) -> np.ndarray:
    """Convert an RDK ``[xyz, qw, qx, qy, qz]`` TCP pose to v2 state values."""

    pose = np.asarray(pose7, dtype=np.float64).reshape(-1)
    if pose.shape != (7,):
        raise ValueError(f"RDK TCP pose must have shape (7,), got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError("RDK TCP pose contains NaN or Inf")
    quat_xyzw = np.asarray((pose[4], pose[5], pose[6], pose[3]), dtype=np.float64)
    if np.linalg.norm(quat_xyzw) < 1e-12:
        raise ValueError("RDK TCP quaternion has zero norm")
    rotation = Rotation.from_quat(quat_xyzw)
    return np.concatenate((pose[:3], rotation_matrix_to_rot6d(rotation.as_matrix())))


def validate_rotation6d(
    rotation_6d: Any,
    *,
    unit_tolerance: float = 1e-4,
    orthogonality_tolerance: float = 1e-4,
    context: str = "rotation-6D",
) -> np.ndarray:
    """Validate finite, unit-length, mutually orthogonal rotation columns."""

    values = np.asarray(rotation_6d, dtype=np.float64)
    if values.shape[-1:] != (6,):
        raise ValueError(f"{context} must end in dimension 6, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{context} contains NaN or Inf")
    c0 = values[..., :3]
    c1 = values[..., 3:]
    c0_norm = np.linalg.norm(c0, axis=-1)
    c1_norm = np.linalg.norm(c1, axis=-1)
    dot = np.sum(c0 * c1, axis=-1)
    if np.any(np.abs(c0_norm - 1.0) > float(unit_tolerance)):
        raise ValueError(f"{context} c0 is not unit length within tolerance")
    if np.any(np.abs(c1_norm - 1.0) > float(unit_tolerance)):
        raise ValueError(f"{context} c1 is not unit length within tolerance")
    if np.any(np.abs(dot) > float(orthogonality_tolerance)):
        raise ValueError(f"{context} c0/c1 are not orthogonal within tolerance")
    return values.astype(np.float32, copy=False)


def validate_flexiv_state_rotation6d(
    state: Any,
    *,
    unit_tolerance: float = 1e-4,
    orthogonality_tolerance: float = 1e-4,
    context: str = "Flexiv state",
) -> np.ndarray:
    """Validate both arm rotations in a v2 state vector or batch."""

    values = np.asarray(state, dtype=np.float64)
    if values.shape[-1:] != (FLEXIV_STATE_DIM,):
        raise ValueError(
            f"{context} must end in dimension {FLEXIV_STATE_DIM}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{context} contains NaN or Inf")
    for side, indices in (
        ("left", slice(10, 16)),
        ("right", slice(27, 33)),
    ):
        validate_rotation6d(
            values[..., indices],
            unit_tolerance=unit_tolerance,
            orthogonality_tolerance=orthogonality_tolerance,
            context=f"{context} {side} rotation-6D",
        )
    return values.astype(np.float32, copy=False)


def convert_legacy_abs_rotvec_state(state: Any) -> np.ndarray:
    """Convert explicit Flexiv v1 absolute-rotvec states to v2 rotation-6D."""

    values = np.asarray(state, dtype=np.float64)
    if values.shape[-1:] != (FLEXIV_LEGACY_STATE_DIM,):
        raise ValueError(
            "legacy Flexiv state must end in dimension "
            f"{FLEXIV_LEGACY_STATE_DIM}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("legacy Flexiv state contains NaN or Inf")

    flat = values.reshape(-1, FLEXIV_LEGACY_STATE_DIM)
    converted = np.empty((flat.shape[0], FLEXIV_STATE_DIM), dtype=np.float64)
    converted[:, 0:7] = flat[:, 0:7]
    converted[:, 7:10] = flat[:, 7:10]
    converted[:, 16] = flat[:, 13]
    converted[:, 17:24] = flat[:, 14:21]
    converted[:, 24:27] = flat[:, 21:24]
    converted[:, 33] = flat[:, 27]

    left_matrix = Rotation.from_rotvec(flat[:, 10:13]).as_matrix()
    right_matrix = Rotation.from_rotvec(flat[:, 24:27]).as_matrix()
    converted[:, 10:16] = np.concatenate(
        (left_matrix[:, :, 0], left_matrix[:, :, 1]),
        axis=1,
    )
    converted[:, 27:33] = np.concatenate(
        (right_matrix[:, :, 0], right_matrix[:, :, 1]),
        axis=1,
    )
    output = converted.reshape(values.shape[:-1] + (FLEXIV_STATE_DIM,)).astype(
        np.float32,
        copy=False,
    )
    validate_flexiv_state_rotation6d(output, context="converted Flexiv state")
    return output


def validate_contract_metadata(
    metadata: Mapping[str, Any],
    *,
    source: str,
    require_v2: bool = True,
) -> None:
    """Reject conflicting persisted metadata while allowing absent optional keys."""

    expected = build_flexiv_state_schema() if require_v2 else build_flexiv_legacy_state_schema()
    mismatches = []
    for key, expected_value in expected.items():
        if key in metadata and metadata[key] != expected_value:
            mismatches.append(
                f"{key}: expected {expected_value!r}, got {metadata[key]!r}"
            )
    if mismatches:
        raise ValueError(f"{source} Flexiv schema metadata mismatch: {'; '.join(mismatches)}")
