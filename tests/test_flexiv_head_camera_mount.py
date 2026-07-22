import os
import sys
from pathlib import Path

import numpy as np
import yaml
from flexiv_sapien_test_utils import open_flexiv_env

ROOT = Path(__file__).resolve().parents[1]
RMBENCH = ROOT / "third_party/sim/RMBench"
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))
sys.path.insert(0, str(RMBENCH))
os.chdir(RMBENCH)

from envs.flexiv_embodiment_smoke import FlexivEmbodimentSmoke


def test_home_pose_matches_real_runtime_example():
    real = yaml.safe_load(
        (ROOT / "third_party/real/dual_flexiv_rizon4s/configs/flexiv_runtime.example.yaml").read_text()
    )["robot"]
    home = yaml.safe_load((ROOT / "sim_assets/flexiv_rizon4s_dual_gn01/home_pose.yaml").read_text())
    assert np.allclose(home["left"]["qpos"], real["left_home_joints"], atol=1e-8)
    assert np.allclose(home["right"]["qpos"], real["right_home_joints"], atol=1e-8)


def test_user_specified_station_geometry_contract():
    geometry = yaml.safe_load(
        (ROOT / "sim_assets/flexiv_rizon4s_dual_gn01/installation_geometry.yaml").read_text()
    )
    camera = yaml.safe_load(
        (ROOT / "sim_assets/flexiv_rizon4s_dual_gn01/camera_mount.yaml").read_text()
    )
    left = np.asarray(geometry["left_base_pose_world"]["position"], dtype=float)
    right = np.asarray(geometry["right_base_pose_world"]["position"], dtype=float)
    assert np.isclose(np.linalg.norm(left - right), 0.30, atol=1e-8)
    assert np.isclose(geometry["left_base_pose_world"]["rpy_rad"][0], -np.pi / 4, atol=1e-8)
    assert np.isclose(geometry["right_base_pose_world"]["rpy_rad"][0], np.pi / 4, atol=1e-8)
    assert np.isclose(
        geometry["raised_base_rack"]["platform_top_height_m"] - geometry["table_height_m"],
        0.20,
        atol=1e-8,
    )
    assert (ROOT / "sim_assets/flexiv_rizon4s_dual_gn01" / geometry["raised_base_rack"]["urdf"]).is_file()
    assert np.allclose(
        np.asarray(camera["position_world"], dtype=float),
        np.asarray(camera["base_midpoint_world"], dtype=float)
        + np.asarray(camera["offset_from_base_midpoint_m"], dtype=float),
        atol=1e-8,
    )


def test_only_fixed_head_camera_has_finite_rgbd():
    env = open_flexiv_env(FlexivEmbodimentSmoke)
    try:
        rgb, depth = env.capture_head()
        assert env.static_camera_names == ["head_camera"]
        assert rgb.ndim == 3 and rgb.shape[-1] == 3
        assert depth.shape == rgb.shape[:2]
        assert np.isfinite(depth).all()
        assert np.ptp(rgb) > 5
        assert np.mean(depth > 0) > 0.01
    finally:
        env.close()


def test_bounded_acceptance_views_contain_rendered_geometry():
    env = open_flexiv_env(FlexivEmbodimentSmoke)
    try:
        for name in ("front", "side", "top"):
            rgb, depth = env.capture_view(name)
            assert np.ptp(rgb) > 5, name
            assert np.mean(depth > 0) > 0.01, name
    finally:
        env.close()
