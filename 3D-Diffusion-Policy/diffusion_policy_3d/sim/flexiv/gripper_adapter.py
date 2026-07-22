"""URDF-derived normalized position mapping for Flexiv GN01."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class MimicRelation:
    parent: str
    multiplier: float
    offset: float


@dataclass(frozen=True)
class GripperMapping:
    base_joint: str
    closed_position: float
    open_position: float
    mimic_joints: dict[str, MimicRelation]

    @classmethod
    def from_manifest(cls, payload: dict[str, Any]) -> "GripperMapping":
        return cls(
            base_joint=str(payload["base_joint"]),
            closed_position=float(payload["closed_position"]),
            open_position=float(payload["open_position"]),
            mimic_joints={
                name: MimicRelation(
                    parent=str(item["joint"]),
                    multiplier=float(item["multiplier"]),
                    offset=float(item["offset"]),
                )
                for name, item in payload.get("mimic_joints", {}).items()
            },
        )

    @classmethod
    def from_urdf(cls, urdf_path: str | Path, *, base_suffix: str = "finger_width_joint") -> "GripperMapping":
        root = ET.parse(urdf_path).getroot()
        base = None
        relations: dict[str, MimicRelation] = {}
        for joint in root.findall("joint"):
            name = joint.get("name", "")
            if name.endswith(base_suffix):
                base = joint
            mimic = joint.find("mimic")
            if mimic is not None:
                relations[name] = MimicRelation(
                    parent=mimic.get("joint", ""),
                    multiplier=float(mimic.get("multiplier", "1")),
                    offset=float(mimic.get("offset", "0")),
                )
        if base is None:
            raise ValueError(f"GN01 base joint {base_suffix!r} not found in {urdf_path}")
        limit = base.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"GN01 base joint has no lower/upper limits: {base.get('name')}")
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        if lower == upper:
            raise ValueError("GN01 open and closed positions must differ")
        return cls(base.get("name", ""), min(lower, upper), max(lower, upper), relations)

    def _validate_normalized(self, value: float, *, clip: bool) -> float:
        if not np.isfinite(value):
            raise ValueError("GN01 command contains NaN/Inf")
        if clip:
            return float(np.clip(value, 0.0, 1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"GN01 command must be in [0,1], got {value}")
        return float(value)

    def normalized_to_base(self, value: float, *, clip: bool = False) -> float:
        value = self._validate_normalized(float(value), clip=clip)
        return self.closed_position + value * (self.open_position - self.closed_position)

    def base_to_normalized(self, position: float, *, clip: bool = True) -> float:
        denominator = self.open_position - self.closed_position
        value = (float(position) - self.closed_position) / denominator
        if clip:
            value = float(np.clip(value, 0.0, 1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"GN01 joint position is outside its limits: {position}")
        return float(value)

    def normalized_to_joint_targets(self, value: float, *, clip: bool = False) -> dict[str, float]:
        base_position = self.normalized_to_base(value, clip=clip)
        targets = {self.base_joint: base_position}
        pending = dict(self.mimic_joints)
        while pending:
            progressed = False
            for name, relation in list(pending.items()):
                if relation.parent in targets:
                    targets[name] = relation.multiplier * targets[relation.parent] + relation.offset
                    del pending[name]
                    progressed = True
            if not progressed:
                raise ValueError(f"GN01 mimic graph is unresolved: {sorted(pending)}")
        return targets
