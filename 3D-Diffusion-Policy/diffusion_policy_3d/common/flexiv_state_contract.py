"""Canonical Flexiv dual-arm state/action contract helpers.

The state orientation is an absolute RDK TCP orientation represented by the
first two *columns* of its rotation matrix.  Keeping this definition in one
small, dependency-light module prevents exporter, dataset, and inference code
from silently adopting different reshape conventions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


FLEXIV_STATE_SCHEMA = "flexiv_abs_rot6d_v2"
FLEXIV_LEGACY_STATE_SCHEMA = "flexiv_abs_rotvec_v1"
FLEXIV_RAW_FORCE_STATE_SCHEMA = "flexiv_abs_rot6d_raw_force_v3"
FLEXIV_STATE_DIM = 34
FLEXIV_LEGACY_STATE_DIM = 28
FLEXIV_RAW_FORCE_STATE_DIM = 48
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
FLEXIV_RAW_FORCE_DROPPED_STATE_NAMES = (
    "left_ee_ext_wrench_in_tcp_raw.fx",
    "left_ee_ext_wrench_in_tcp_raw.fy",
    "left_ee_ext_wrench_in_tcp_raw.fz",
    "left_ee_ext_wrench_in_tcp_raw.mx",
    "left_ee_ext_wrench_in_tcp_raw.my",
    "left_ee_ext_wrench_in_tcp_raw.mz",
    "left_gripper_force",
    "right_ee_ext_wrench_in_tcp_raw.fx",
    "right_ee_ext_wrench_in_tcp_raw.fy",
    "right_ee_ext_wrench_in_tcp_raw.fz",
    "right_ee_ext_wrench_in_tcp_raw.mx",
    "right_ee_ext_wrench_in_tcp_raw.my",
    "right_ee_ext_wrench_in_tcp_raw.mz",
    "right_gripper_force",
)
FLEXIV_RAW_FORCE_TO_V2_TRANSFORM = "drop_raw_force_fields_v3_to_v2_by_name"
FLEXIV_LEGACY_TO_V2_TRANSFORM = "legacy_abs_rotvec_to_abs_rot6d"

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


def flexiv_raw_force_state_names() -> list[str]:
    """Return the complete acquisition-side v3/48D field order."""

    return [*flexiv_state_names(), *FLEXIV_RAW_FORCE_DROPPED_STATE_NAMES]


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


FLEXIV_RAW_FORCE_STATE_NAMES = tuple(flexiv_raw_force_state_names())


@dataclass(frozen=True)
class FlexivSourceStateContract:
    """Validated source-side state metadata and its target-state transform."""

    schema: str
    transform: str
    state_dim: int
    state_names: tuple[str, ...]
    target_projection_indices: tuple[int, ...] | None = None
    dropped_state_names: tuple[str, ...] = ()

    @property
    def is_source_v3(self) -> bool:
        return self.schema == FLEXIV_RAW_FORCE_STATE_SCHEMA


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


def build_flexiv_raw_force_state_schema() -> dict[str, Any]:
    """Return the acquisition-side v3/48D schema, including raw force fields."""

    return {
        "state_schema": FLEXIV_RAW_FORCE_STATE_SCHEMA,
        "state_dim": FLEXIV_RAW_FORCE_STATE_DIM,
        "action_dim": FLEXIV_ACTION_DIM,
        "state_names": list(FLEXIV_RAW_FORCE_STATE_NAMES),
        "action_names": flexiv_action_names(),
        "state_rotation_representation": FLEXIV_STATE_ROTATION_REPRESENTATION,
        "state_rotation_reference": FLEXIV_STATE_ROTATION_REFERENCE,
        "rotation6d_convention": FLEXIV_ROTATION6D_CONVENTION,
        "rotation6d_order": list(FLEXIV_ROTATION6D_ORDER),
        "action_rotation_representation": "rotvec",
        "dropped_state_names": list(FLEXIV_RAW_FORCE_DROPPED_STATE_NAMES),
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


def _source_feature_shape(feature: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(feature, Mapping) or "shape" not in feature:
        raise ValueError(f"LeRobot metadata is missing {label} feature shape")
    try:
        return tuple(int(value) for value in feature["shape"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LeRobot metadata has invalid {label} feature shape") from exc


def _source_feature_names(feature: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(feature, Mapping) or not isinstance(feature.get("names"), (list, tuple)):
        raise ValueError(
            f"LeRobot metadata is missing exact {label} names/order; "
            "dimension-only schema detection is forbidden"
        )
    names = tuple(str(name) for name in feature["names"])
    if any(not name for name in names):
        raise ValueError(f"LeRobot metadata contains an empty {label} field name")
    if len(set(names)) != len(names):
        raise ValueError(f"LeRobot metadata contains duplicate {label} field names")
    return names


def _validate_source_dtype(feature: Any, *, label: str, required: bool) -> None:
    raw_dtype = feature.get("dtype") if isinstance(feature, Mapping) else None
    if raw_dtype is None:
        if required:
            raise ValueError(f"LeRobot metadata is missing {label} feature dtype")
        return
    try:
        dtype = np.dtype(raw_dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LeRobot metadata has invalid {label} feature dtype: {raw_dtype!r}") from exc
    if not (
        np.issubdtype(dtype, np.bool_)
        or np.issubdtype(dtype, np.integer)
        or np.issubdtype(dtype, np.floating)
    ):
        raise ValueError(
            f"LeRobot {label} feature dtype {dtype} cannot be converted to float32"
        )


def _source_schema_metadata(info: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    for key in ("robot_state_schema", "state_schema_metadata"):
        value = info.get(key)
        if value is not None:
            if isinstance(value, Mapping):
                candidates.append(value)
            elif isinstance(value, str) and value.strip():
                candidates.append({"state_schema": value.strip()})
            else:
                raise ValueError(f"LeRobot metadata {key!r} must be a mapping or schema string")
    value = info.get("state_schema")
    if value is not None:
        if isinstance(value, Mapping):
            candidates.append(value)
        elif isinstance(value, str) and value.strip():
            candidates.append({"state_schema": value.strip()})
        else:
            raise ValueError("LeRobot metadata 'state_schema' must be a mapping or schema string")
    top_level_contract = {
        key: info[key]
        for key in (
            "state_dim",
            "action_dim",
            "state_names",
            "action_names",
            "state_rotation_representation",
            "state_rotation_reference",
            "rotation6d_convention",
            "rotation6d_order",
            "action_rotation_representation",
            "dropped_state_names",
        )
        if key in info
    }
    if top_level_contract:
        candidates.append(top_level_contract)
    return candidates


def _validate_source_metadata(
    metadata: Mapping[str, Any],
    *,
    source_schema: str,
    state_names: tuple[str, ...],
    action_names: tuple[str, ...],
) -> None:
    legacy_aliases = {FLEXIV_LEGACY_STATE_SCHEMA, "flexiv_abs_rotvec", "flexiv_physical_v1"}
    allowed_schemas = (
        {FLEXIV_STATE_SCHEMA}
        if source_schema == FLEXIV_STATE_SCHEMA
        else {FLEXIV_RAW_FORCE_STATE_SCHEMA}
        if source_schema == FLEXIV_RAW_FORCE_STATE_SCHEMA
        else legacy_aliases
    )
    declared_schema = metadata.get("state_schema")
    if declared_schema is not None and declared_schema not in allowed_schemas:
        raise ValueError(
            "LeRobot state schema metadata conflicts with feature names: "
            f"declared={declared_schema!r}, detected={source_schema!r}"
        )
    expected_names = {
        FLEXIV_STATE_SCHEMA: tuple(flexiv_state_names()),
        FLEXIV_RAW_FORCE_STATE_SCHEMA: FLEXIV_RAW_FORCE_STATE_NAMES,
        FLEXIV_LEGACY_STATE_SCHEMA: tuple(flexiv_legacy_state_names()),
    }[source_schema]
    if "state_names" in metadata and tuple(metadata["state_names"]) != expected_names:
        raise ValueError("LeRobot state schema metadata state_names/order is not exact")
    if "action_names" in metadata and tuple(metadata["action_names"]) != tuple(action_names):
        raise ValueError("LeRobot state schema metadata action_names/order is not exact")
    expected_common = (
        ("state_dim", FLEXIV_STATE_DIM)
        if source_schema == FLEXIV_STATE_SCHEMA
        else ("state_dim", FLEXIV_RAW_FORCE_STATE_DIM)
        if source_schema == FLEXIV_RAW_FORCE_STATE_SCHEMA
        else ("state_dim", FLEXIV_LEGACY_STATE_DIM),
        ("action_dim", FLEXIV_ACTION_DIM),
    )
    for key, expected in expected_common:
        if key in metadata and metadata[key] != expected:
            raise ValueError(
                f"LeRobot state schema metadata {key}={metadata[key]!r} does not match {expected!r}"
            )
    if source_schema in {FLEXIV_STATE_SCHEMA, FLEXIV_RAW_FORCE_STATE_SCHEMA}:
        for key, expected in (
            ("state_rotation_representation", FLEXIV_STATE_ROTATION_REPRESENTATION),
            ("state_rotation_reference", FLEXIV_STATE_ROTATION_REFERENCE),
            ("rotation6d_convention", FLEXIV_ROTATION6D_CONVENTION),
            ("rotation6d_order", list(FLEXIV_ROTATION6D_ORDER)),
            ("action_rotation_representation", "rotvec"),
        ):
            if key in metadata and metadata[key] != expected:
                raise ValueError(
                    f"LeRobot state schema metadata {key}={metadata[key]!r} does not match {expected!r}"
                )
    else:
        for key, expected in (
            ("state_rotation_representation", "absolute_rotvec"),
            ("state_rotation_reference", FLEXIV_STATE_ROTATION_REFERENCE),
            ("action_rotation_representation", "rotvec"),
        ):
            if key in metadata and metadata[key] != expected:
                raise ValueError(
                    f"Legacy Flexiv metadata {key}={metadata[key]!r} does not match {expected!r}"
                )
        if "rotation6d_convention" in metadata or "rotation6d_order" in metadata:
            raise ValueError("Legacy Flexiv metadata must not declare a rotation-6D convention/order")
    if source_schema == FLEXIV_RAW_FORCE_STATE_SCHEMA and "dropped_state_names" in metadata:
        if tuple(metadata["dropped_state_names"]) != FLEXIV_RAW_FORCE_DROPPED_STATE_NAMES:
            raise ValueError("LeRobot v3 metadata dropped_state_names/order is not exact")


def detect_flexiv_source_state_contract(
    info: Mapping[str, Any],
    *,
    state_column: str = "observation.state",
    action_column: str = "action",
    allow_legacy_conversion: bool = False,
) -> FlexivSourceStateContract:
    """Detect and validate a LeRobot source state before any row is consumed."""

    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("LeRobot meta/info.json is missing the features mapping")
    state_feature = features.get(state_column)
    action_feature = features.get(action_column)
    state_shape = _source_feature_shape(state_feature, label=state_column)
    action_shape = _source_feature_shape(action_feature, label=action_column)
    state_names = _source_feature_names(state_feature, label=state_column)
    action_names = _source_feature_names(action_feature, label=action_column)
    _validate_source_dtype(state_feature, label=state_column, required=False)
    _validate_source_dtype(action_feature, label=action_column, required=False)
    if action_shape != (FLEXIV_ACTION_DIM,) or action_names != tuple(flexiv_action_names()):
        raise ValueError(
            "LeRobot action schema mismatch: expected exact 14D Flexiv delta-rotvec names/order, "
            f"got shape={action_shape}, names={action_names!r}"
        )

    if state_names == tuple(flexiv_state_names()):
        schema = FLEXIV_STATE_SCHEMA
        state_dim = FLEXIV_STATE_DIM
        transform = "passthrough_v2"
        projection = tuple(range(FLEXIV_STATE_DIM))
        dropped = ()
    elif state_names == FLEXIV_RAW_FORCE_STATE_NAMES:
        schema = FLEXIV_RAW_FORCE_STATE_SCHEMA
        state_dim = FLEXIV_RAW_FORCE_STATE_DIM
        transform = FLEXIV_RAW_FORCE_TO_V2_TRANSFORM
        # The projection is deliberately constructed by exact target names.
        source_index = {name: index for index, name in enumerate(state_names)}
        projection = tuple(source_index[name] for name in flexiv_state_names())
        dropped = FLEXIV_RAW_FORCE_DROPPED_STATE_NAMES
    elif state_names == tuple(flexiv_legacy_state_names()):
        schema = FLEXIV_LEGACY_STATE_SCHEMA
        state_dim = FLEXIV_LEGACY_STATE_DIM
        transform = FLEXIV_LEGACY_TO_V2_TRANSFORM
        projection = None
        dropped = ()
    else:
        raise ValueError(
            "Unknown Flexiv state schema: exact state names/order do not match v2 (34D), "
            "v3 raw-force (48D), or the supported legacy v1 (28D)."
        )
    if state_shape != (state_dim,):
        raise ValueError(
            f"Detected {schema!r} by names but metadata shape is {state_shape}; expected ({state_dim},)"
        )
    if schema == FLEXIV_RAW_FORCE_STATE_SCHEMA:
        _validate_source_dtype(state_feature, label=state_column, required=True)
        metadata_candidates = _source_schema_metadata(info)
        if not metadata_candidates:
            raise ValueError(
                "LeRobot v3 raw-force state requires explicit schema metadata; "
                "dimension and names alone are insufficient"
            )
        if not any(item.get("state_schema") == FLEXIV_RAW_FORCE_STATE_SCHEMA for item in metadata_candidates):
            raise ValueError(
                "LeRobot v3 raw-force state metadata must explicitly declare "
                f"state_schema={FLEXIV_RAW_FORCE_STATE_SCHEMA!r}"
            )
    for metadata in _source_schema_metadata(info):
        _validate_source_metadata(
            metadata,
            source_schema=schema,
            state_names=state_names,
            action_names=action_names,
        )
    if schema == FLEXIV_LEGACY_STATE_SCHEMA and not allow_legacy_conversion:
        raise ValueError(
            "Detected the supported legacy Flexiv 28D absolute-rotvec schema, but legacy "
            "conversion was not explicitly enabled. Re-run with --allow-legacy-state-conversion."
        )
    return FlexivSourceStateContract(
        schema=schema,
        transform=transform,
        state_dim=state_dim,
        state_names=state_names,
        target_projection_indices=projection,
        dropped_state_names=dropped,
    )


def project_flexiv_source_state_to_v2(
    state: Any,
    source_contract: FlexivSourceStateContract,
) -> np.ndarray:
    """Validate one source row and project it to the 34D DP3 target by contract."""

    try:
        values = np.asarray(state, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Source state for {source_contract.schema} cannot be converted to float32"
        ) from exc
    if values.shape != (source_contract.state_dim,):
        raise ValueError(
            f"source state shape {values.shape} != ({source_contract.state_dim},) for "
            f"{source_contract.schema}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Source state for {source_contract.schema} contains NaN or Inf")
    if source_contract.schema == FLEXIV_STATE_SCHEMA:
        validate_flexiv_state_rotation6d(values, context="source v2 state")
        return values.copy()
    if source_contract.schema == FLEXIV_RAW_FORCE_STATE_SCHEMA:
        indices = source_contract.target_projection_indices
        if indices is None or tuple(indices) != tuple(
            {name: index for index, name in enumerate(source_contract.state_names)}[name]
            for name in flexiv_state_names()
        ):
            raise ValueError("v3 source contract does not contain the exact v3-to-v2 name projection")
        projected = values[np.asarray(indices, dtype=np.int64)]
        validate_flexiv_state_rotation6d(projected, context="projected v3 state")
        return projected.astype(np.float32, copy=False)
    if source_contract.schema == FLEXIV_LEGACY_STATE_SCHEMA:
        if source_contract.transform != FLEXIV_LEGACY_TO_V2_TRANSFORM:
            raise ValueError(f"Unsupported legacy state transform {source_contract.transform!r}")
        return convert_legacy_abs_rotvec_state(values)
    raise ValueError(f"Unsupported source state schema {source_contract.schema!r}")


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
