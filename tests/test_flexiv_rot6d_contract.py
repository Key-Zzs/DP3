import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))
from diffusion_policy_3d.common.flexiv_state_contract import rotation_matrix_to_rot6d


def test_rotation6d_is_first_two_matrix_columns_not_reshape():
    matrix = Rotation.from_euler("xyz", [0.3, -0.2, 0.8]).as_matrix()
    actual = rotation_matrix_to_rot6d(matrix)
    expected = np.concatenate((matrix[:, 0], matrix[:, 1])).astype(np.float32)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
