#!/usr/bin/env python3
"""Print the exact state and action field order used by Stage 2."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))

from diffusion_policy_3d.common.flexiv_state_contract import flexiv_action_names, flexiv_state_names


def main() -> int:
    print("state_schema=flexiv_abs_rot6d_v2")
    print("state_dim=34")
    for index, name in enumerate(flexiv_state_names()):
        print(f"{index:02d} {name}")
    print("action_dim=14")
    for index, name in enumerate(flexiv_action_names()):
        print(f"{index:02d} {name}")
    print("rotation_composition=R_target = Exp(delta_rotvec) @ R_current")
    print("translation_frame=each arm base frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
