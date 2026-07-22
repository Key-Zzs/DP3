import os
import sys
from pathlib import Path

import numpy as np
from flexiv_sapien_test_utils import open_flexiv_env

ROOT = Path(__file__).resolve().parents[1]
RMBENCH = ROOT / "third_party/sim/RMBench"
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))
sys.path.insert(0, str(RMBENCH))
os.chdir(RMBENCH)

from envs.flexiv_embodiment_smoke import FlexivEmbodimentSmoke


def test_only_fixed_head_camera_has_finite_rgbd():
    env = open_flexiv_env(FlexivEmbodimentSmoke)
    try:
        rgb, depth = env.capture_head()
        assert env.static_camera_names == ["head_camera"]
        assert rgb.ndim == 3 and rgb.shape[-1] == 3
        assert depth.shape == rgb.shape[:2]
        assert np.isfinite(depth).all()
    finally:
        env.close()
