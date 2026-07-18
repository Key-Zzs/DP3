from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "export_lerobot_to_dp3_zarr.py"
SPEC = importlib.util.spec_from_file_location("export_lerobot_to_dp3_zarr", MODULE_PATH)
exporter = importlib.util.module_from_spec(SPEC)
sys.modules["export_lerobot_to_dp3_zarr"] = exporter
assert SPEC.loader is not None
SPEC.loader.exec_module(exporter)


def _calibration() -> dict:
    camera = {
        "depth_scale_m_per_unit": 0.001,
        "streams": {
            "depth": {
                "intrinsics": {
                    "width": 640,
                    "height": 480,
                    "fx": 392.5,
                    "fy": 392.5,
                    "cx": 316.8,
                    "cy": 235.8,
                }
            },
            "color": {
                "intrinsics": {
                    "width": 640,
                    "height": 480,
                    "fx": 606.1,
                    "fy": 606.6,
                    "cx": 320.8,
                    "cy": 256.1,
                }
            },
        },
        "extrinsics": {
            "depth_to_color": {
                "rotation_matrix_row_major": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "translation_m": [0.015, 0.0, 0.0],
            }
        },
    }
    return {
        "cameras": {
            "head_rgb": camera,
            "left_wrist_rgb": camera,
            "right_wrist_rgb": camera,
        }
    }


def _v2_state(frames: int) -> np.ndarray:
    state = np.zeros((frames, exporter.STATE_DIM), dtype=np.float32)
    state[:, :7] = np.arange(frames * 7, dtype=np.float32).reshape(frames, 7)
    state[:, 17:24] = np.arange(frames * 7, dtype=np.float32).reshape(frames, 7)
    state[:, 10:16] = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    state[:, 27:33] = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    state[:, 16] = 0.5
    state[:, 33] = 0.5
    return state


def test_build_pointcloud_builder_config_xyz() -> None:
    config = exporter.build_pointcloud_builder_config(
        _calibration(),
        camera="head",
        pointcloud_mode="xyz",
        num_points=1024,
    )
    assert config["camera"]["aligned_depth_to_color"] is False
    assert config["camera"]["depth_scale"] == 0.001
    assert config["pointcloud"]["use_rgb"] is False
    assert config["pointcloud"]["output_format"] == "xyz"
    assert config["sampling"]["num_points"] == 1024


def test_build_pointcloud_builder_config_xyzrgb() -> None:
    config = exporter.build_pointcloud_builder_config(
        _calibration(),
        camera="head",
        pointcloud_mode="xyzrgb",
        num_points=512,
    )
    assert config["pointcloud"]["use_rgb"] is True
    assert config["pointcloud"]["rgb_mapping"] == "project_depth_to_color"
    assert config["pointcloud"]["rgb_sampling"] == "nearest"
    assert config["camera"]["depth_to_color_extrinsics"]["translation"] == [0.015, 0.0, 0.0]
    assert config["sampling"]["num_points"] == 512


def test_rotation6d_identity_and_known_column_convention() -> None:
    np.testing.assert_array_equal(
        exporter.rotation_matrix_to_rot6d(np.eye(3)),
        np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )
    rotation = Rotation.from_euler("zyx", [0.4, -0.2, 0.7])
    np.testing.assert_allclose(
        exporter.rotation_matrix_to_rot6d(rotation.as_matrix()),
        np.concatenate((rotation.as_matrix()[:, 0], rotation.as_matrix()[:, 1])),
        atol=1e-7,
    )


def test_rotation6d_x_y_z_known_rotations_use_matrix_columns() -> None:
    expected = {
        "x": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "y": [0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        "z": [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
    }
    for axis, values in expected.items():
        np.testing.assert_allclose(
            exporter.rotation_matrix_to_rot6d(
                Rotation.from_euler(axis, np.pi / 2.0).as_matrix()
            ),
            np.asarray(values, dtype=np.float32),
            atol=1e-7,
        )


def test_rotation6d_quaternion_sign_and_pi_crossing_are_continuous() -> None:
    rotation = Rotation.from_euler("xyz", [0.3, -0.7, 1.2])
    quat = rotation.as_quat()
    matrix_a = Rotation.from_quat(quat).as_matrix()
    matrix_b = Rotation.from_quat(-quat).as_matrix()
    np.testing.assert_allclose(
        exporter.rotation_matrix_to_rot6d(matrix_a),
        exporter.rotation_matrix_to_rot6d(matrix_b),
        atol=1e-7,
    )
    axis = np.asarray([0.3, -0.4, 0.5], dtype=float)
    axis /= np.linalg.norm(axis)
    before = Rotation.from_rotvec(axis * (np.pi - 1e-7)).as_matrix()
    after = Rotation.from_rotvec(axis * (np.pi + 1e-7)).as_matrix()
    assert np.linalg.norm(
        exporter.rotation_matrix_to_rot6d(before)
        - exporter.rotation_matrix_to_rot6d(after)
    ) < 1e-5


def _schema_info(state_names: list[str], state_dim: int) -> dict:
    return {
        "features": {
            exporter.STATE_COLUMN: {
                "dtype": "float32",
                "shape": [state_dim],
                "names": state_names,
            },
            exporter.ACTION_COLUMN: {
                "dtype": "float32",
                "shape": [exporter.ACTION_DIM],
                "names": list(exporter.ACTION_FIELD_NAMES),
            },
        }
    }


def _v3_schema_info() -> dict:
    info = _schema_info(list(exporter.RAW_FORCE_STATE_FIELD_NAMES), exporter.FLEXIV_RAW_FORCE_STATE_DIM)
    info["robot_state_schema"] = exporter.build_flexiv_raw_force_state_schema()
    return info


def test_legacy_abs_rotvec_to_v2_conversion_is_explicit_and_exact() -> None:
    legacy = np.zeros(exporter.FLEXIV_LEGACY_STATE_DIM, dtype=np.float32)
    legacy[:7] = np.arange(7, dtype=np.float32)
    legacy[7:10] = [0.1, 0.2, 0.3]
    legacy[10:13] = [0.0, 0.0, np.pi - 1e-7]
    legacy[13] = 0.25
    legacy[14:21] = np.arange(7, dtype=np.float32) + 10
    legacy[21:24] = [0.4, 0.5, 0.6]
    legacy[24:27] = [0.0, 0.0, -np.pi + 1e-7]
    legacy[27] = 0.75
    converted = exporter.convert_legacy_abs_rotvec_to_v2(legacy)
    assert converted.shape == (exporter.STATE_DIM,)
    np.testing.assert_allclose(converted[:7], legacy[:7])
    np.testing.assert_allclose(converted[7:10], legacy[7:10])
    np.testing.assert_allclose(converted[16], legacy[13])
    np.testing.assert_allclose(converted[17:24], legacy[14:21])
    np.testing.assert_allclose(converted[24:27], legacy[21:24])
    np.testing.assert_allclose(converted[33], legacy[27])
    expected_left = Rotation.from_rotvec(legacy[10:13]).as_matrix()
    expected_right = Rotation.from_rotvec(legacy[24:27]).as_matrix()
    np.testing.assert_allclose(
        converted[10:16], np.concatenate((expected_left[:, 0], expected_left[:, 1])), atol=1e-6
    )
    np.testing.assert_allclose(
        converted[27:33], np.concatenate((expected_right[:, 0], expected_right[:, 1])), atol=1e-6
    )


def test_source_schema_detection_rejects_unknown_28d_and_requires_legacy_flag() -> None:
    unknown = _schema_info([f"unknown_{i}" for i in range(28)], 28)
    with pytest.raises(ValueError, match="duplicate|Unknown Flexiv state schema"):
        exporter.detect_source_state_contract(unknown)

    legacy = _schema_info(list(exporter.LEGACY_STATE_FIELD_NAMES), 28)
    with pytest.raises(ValueError, match="legacy.*conversion.*explicitly enabled"):
        exporter.detect_source_state_contract(legacy)
    detected = exporter.detect_source_state_contract(
        legacy,
        allow_legacy_conversion=True,
    )
    assert detected.schema == exporter.FLEXIV_LEGACY_STATE_SCHEMA
    assert detected.transform == exporter.LEGACY_CONVERTER_NAME


def test_source_schema_detection_accepts_exact_v2_names() -> None:
    info = _schema_info(list(exporter.STATE_FIELD_NAMES), exporter.STATE_DIM)
    detected = exporter.detect_source_state_contract(info)
    assert detected.schema == exporter.TARGET_STATE_SCHEMA
    assert detected.transform == "passthrough_v2"
    source = _v2_state(1)[0]
    np.testing.assert_array_equal(exporter.convert_source_state(source, detected), source)


def test_v3_source_projection_is_exactly_name_indexed_and_drops_only_force_fields() -> None:
    info = _v3_schema_info()
    detected = exporter.detect_source_state_contract(info)
    assert detected.schema == exporter.FLEXIV_RAW_FORCE_STATE_SCHEMA
    assert detected.state_dim == exporter.FLEXIV_RAW_FORCE_STATE_DIM
    assert detected.transform == exporter.RAW_FORCE_CONVERTER_NAME
    assert detected.dropped_state_names == exporter.FLEXIV_RAW_FORCE_DROPPED_STATE_NAMES

    source = np.arange(exporter.FLEXIV_RAW_FORCE_STATE_DIM, dtype=np.float32)
    source[10:16] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    source[27:33] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    projected = exporter.convert_source_state(source, detected)
    expected = source[np.asarray(detected.target_projection_indices, dtype=np.int64)]
    np.testing.assert_array_equal(projected, expected)
    np.testing.assert_array_equal(
        projected,
        np.asarray([source[detected.state_names.index(name)] for name in exporter.STATE_FIELD_NAMES]),
    )


def test_v3_force_values_do_not_change_projected_dp3_state() -> None:
    detected = exporter.detect_source_state_contract(_v3_schema_info())
    source = np.arange(exporter.FLEXIV_RAW_FORCE_STATE_DIM, dtype=np.float32)
    source[10:16] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    source[27:33] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    baseline = exporter.convert_source_state(source, detected)
    source_changed = source.copy()
    source_indices = {name: index for index, name in enumerate(detected.state_names)}
    source_changed[
        np.asarray(
            [source_indices[name] for name in detected.dropped_state_names],
            dtype=np.int64,
        )
    ] = np.asarray(
        [1e30, -1e30, 7.5, -8.5, 1e-30, -1e-30, 123456.0, -654321.0, 0.0, 1.0, -2.0, 3.0, -4.0, 5.0],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(exporter.convert_source_state(source_changed, detected), baseline)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda names: names.__setitem__(35, names[34]),
        lambda names: names.__setitem__(34, "unknown_force_field"),
        lambda names: names.__setitem__(34, names[35]),
    ],
)
def test_v3_source_names_must_be_complete_unique_and_exactly_ordered(mutator) -> None:
    names = list(exporter.RAW_FORCE_STATE_FIELD_NAMES)
    mutator(names)
    info = _schema_info(names, exporter.FLEXIV_RAW_FORCE_STATE_DIM)
    info["robot_state_schema"] = exporter.build_flexiv_raw_force_state_schema()
    with pytest.raises(ValueError, match="duplicate|Unknown Flexiv state schema"):
        exporter.detect_source_state_contract(info)


@pytest.mark.parametrize(
    ("names", "state_dim"),
    [
        (list(exporter.RAW_FORCE_STATE_FIELD_NAMES[:-1]), exporter.FLEXIV_RAW_FORCE_STATE_DIM - 1),
        (
            [*exporter.RAW_FORCE_STATE_FIELD_NAMES, "unknown_extra_force_field"],
            exporter.FLEXIV_RAW_FORCE_STATE_DIM + 1,
        ),
    ],
)
def test_v3_missing_or_extra_unknown_field_fails_fast(names, state_dim) -> None:
    info = _schema_info(names, state_dim)
    info["robot_state_schema"] = exporter.build_flexiv_raw_force_state_schema()
    with pytest.raises(ValueError, match="Unknown Flexiv state schema"):
        exporter.detect_source_state_contract(info)


def test_v3_source_metadata_mismatch_fails_before_rows_are_read() -> None:
    info = _v3_schema_info()
    info["robot_state_schema"]["state_dim"] = 34
    with pytest.raises(ValueError, match="metadata state_dim"):
        exporter.detect_source_state_contract(info)

    info = _v3_schema_info()
    info["features"][exporter.STATE_COLUMN]["shape"] = [34]
    with pytest.raises(ValueError, match="metadata shape"):
        exporter.detect_source_state_contract(info)


def test_unknown_source_schema_is_rejected_before_parquet_rows_are_read(tmp_path, monkeypatch) -> None:
    root = tmp_path / "unknown_source"
    (root / "meta").mkdir(parents=True)
    info = _schema_info([f"unknown_{index}" for index in range(48)], 48)
    info["robot_state_schema"] = {"state_schema": "unknown_48d_schema", "state_dim": 48}
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    builder_config = tmp_path / "builder.yaml"
    builder_config.write_text(
        yaml.safe_dump(
            {
                "camera": {"name": "head"},
                "pointcloud": {"output_format": "xyz", "use_rgb": False},
                "sampling": {"num_points": 8},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        exporter,
        "_data_parquet_paths",
        lambda _root: pytest.fail("unknown source schema reached parquet row discovery"),
    )
    args = argparse.Namespace(
        lerobot_path=str(root.resolve()),
        output_zarr=str(tmp_path / "never_created.zarr"),
        builder_config=str(builder_config),
        target_state_schema=exporter.TARGET_STATE_SCHEMA,
        allow_legacy_state_conversion=False,
        camera="head",
        pointcloud_mode="xyz",
        num_points=8,
        overwrite=False,
        max_frames=None,
        save_img=False,
        verbose=False,
    )
    with pytest.raises(ValueError, match="Unknown Flexiv state schema"):
        exporter.export_lerobot_to_dp3_zarr(args)


def test_compute_episode_ends_from_episode_rows_and_max_frames() -> None:
    rows = [
        {"episode_index": 0, "length": 4, "dataset_from_index": 0, "dataset_to_index": 4},
        {"episode_index": 1, "length": 5, "dataset_from_index": 4, "dataset_to_index": 9},
    ]
    assert exporter.compute_episode_ends(episode_rows=rows, total_frames=9).tolist() == [4, 9]
    assert exporter.compute_episode_ends(
        episode_rows=rows,
        total_frames=9,
        max_frames=6,
    ).tolist() == [4, 6]


def test_compute_episode_ends_from_indices() -> None:
    indices = np.asarray([0, 0, 0, 1, 1, 2])
    ends = exporter.compute_episode_ends(episode_indices=indices, total_frames=6)
    assert ends.tolist() == [3, 5, 6]


def test_default_output_zarr_path_uses_sanitized_repo_id(tmp_path) -> None:
    output = exporter.default_output_zarr_path(
        tmp_path / "dataset",
        {"repo_id": "owner/my dataset"},
        camera="head",
        pointcloud_mode="xyzrgb",
        output_root=tmp_path,
    )
    assert output == tmp_path / "owner_my_dataset_head_xyzrgb_state_abs_rot6d_v2.zarr"


def test_default_output_zarr_path_falls_back_to_lerobot_cache_relative(tmp_path) -> None:
    lerobot_path = Path.home() / ".cache" / "huggingface" / "lerobot" / "org" / "dataset"
    output = exporter.default_output_zarr_path(
        lerobot_path,
        {},
        camera="left_wrist",
        pointcloud_mode="xyz",
        output_root=tmp_path,
    )
    assert output == tmp_path / "org_dataset_left_wrist_xyz_state_abs_rot6d_v2.zarr"


def test_write_dp3_zarr_xyz_and_overwrite(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    assert zarr is not None
    output = tmp_path / "mock_xyz.zarr"
    state = _v2_state(2)
    action = np.zeros((2, 14), dtype=np.float32)
    point_cloud = np.zeros((2, 8, 3), dtype=np.float32)
    exporter.write_dp3_zarr(
        output,
        state=state,
        action=action,
        point_cloud=point_cloud,
        episode_ends=np.asarray([2], dtype=np.int64),
        attrs={"source_lerobot_path": "/tmp/mock", "pointcloud_mode": "xyz"},
        overwrite=False,
    )
    root = zarr.open(str(output), mode="r")
    assert root["data"]["state"].shape == (2, exporter.STATE_DIM)
    assert root["data"]["action"].shape == (2, exporter.ACTION_DIM)
    assert root["data"]["point_cloud"].shape == (2, 8, 3)
    assert root.attrs["export_status"] == "complete"
    assert root.attrs["state_schema"] == exporter.TARGET_STATE_SCHEMA
    assert root.attrs["rotation6d_convention"] == "matrix_columns_0_1"
    assert root.attrs["action_rotation_representation"] == "rotvec"
    assert root.attrs["state_dim"] == 34
    assert root.attrs["action_dim"] == 14
    assert root.attrs["state_names"] == list(exporter.STATE_FIELD_NAMES)
    assert root.attrs["action_names"] == list(exporter.ACTION_FIELD_NAMES)
    assert root.attrs["state_rotation_representation"] == "rotation_6d"
    assert root.attrs["state_rotation_reference"] == "absolute_rdk_world_base"
    assert root.attrs["rotation6d_order"] == ["c0x", "c0y", "c0z", "c1x", "c1y", "c1z"]
    assert root.attrs["source_state_schema"] == exporter.TARGET_STATE_SCHEMA
    assert root.attrs["source_state_names"] == list(exporter.STATE_FIELD_NAMES)
    assert root.attrs["state_transform"] == "passthrough_v2"
    assert "source_fps" in root.attrs
    assert len(root.attrs["raw_source_state_sha256"]) == 64
    assert len(root.attrs["derived_state_sha256"]) == 64
    assert root.attrs["integrity"]["raw_source_state"] == root.attrs["raw_source_state_sha256"]
    assert root.attrs["integrity"]["derived_state"] == root.attrs["derived_state_sha256"]
    assert root.attrs["converted_frames"] == 2
    assert exporter.verify_dp3_zarr(output)["state"] == root.attrs["integrity"]["state"]
    with pytest.raises(FileExistsError):
        exporter.write_dp3_zarr(
            output,
            state=state,
            action=action,
            point_cloud=point_cloud,
            episode_ends=np.asarray([2], dtype=np.int64),
            attrs={},
            overwrite=False,
        )
    exporter.write_dp3_zarr(
        output,
        state=state,
        action=action,
        point_cloud=point_cloud,
        episode_ends=np.asarray([2], dtype=np.int64),
        attrs={},
        overwrite=True,
    )


def test_write_dp3_zarr_xyzrgb(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    assert zarr is not None
    output = tmp_path / "mock_xyzrgb.zarr"
    exporter.write_dp3_zarr(
        output,
        state=_v2_state(3),
        action=np.zeros((3, 14), dtype=np.float32),
        point_cloud=np.zeros((3, 4, 6), dtype=np.float32),
        episode_ends=np.asarray([1, 3], dtype=np.int64),
        attrs={"pointcloud_mode": "xyzrgb"},
        overwrite=True,
    )
    root = zarr.open(str(output), mode="r")
    assert root["data"]["point_cloud"].shape == (3, 4, 6)
    assert root["meta"]["episode_ends"][:].tolist() == [1, 3]
    assert exporter.verify_dp3_zarr(output) == root.attrs["integrity"]


def test_write_dp3_zarr_v3_provenance_keeps_raw_and_derived_hashes_distinct(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    output = tmp_path / "v3_output.zarr"
    target_state = _v2_state(2)
    raw_state = np.concatenate(
        [target_state, np.asarray([[1.0] * 14, [-2.0] * 14], dtype=np.float32)],
        axis=1,
    )
    raw_hash = exporter._numpy_sha256(raw_state, dtype=np.float32)
    derived_hash = exporter._numpy_sha256(target_state, dtype=np.float32)
    exporter.write_dp3_zarr(
        output,
        state=target_state,
        action=np.zeros((2, 14), dtype=np.float32),
        point_cloud=np.zeros((2, 8, 3), dtype=np.float32),
        episode_ends=np.asarray([2], dtype=np.int64),
        attrs={
            "source_state_schema": exporter.FLEXIV_RAW_FORCE_STATE_SCHEMA,
            "source_state_dim": exporter.FLEXIV_RAW_FORCE_STATE_DIM,
            "source_state_names": list(exporter.RAW_FORCE_STATE_FIELD_NAMES),
            "state_transform": exporter.RAW_FORCE_CONVERTER_NAME,
            "dropped_state_names": list(exporter.FLEXIV_RAW_FORCE_DROPPED_STATE_NAMES),
            "raw_source_state_sha256": raw_hash,
        },
    )
    root = zarr.open(str(output), mode="r")
    assert root.attrs["state_schema"] == exporter.TARGET_STATE_SCHEMA
    assert root.attrs["state_dim"] == 34
    assert root.attrs["source_state_schema"] == exporter.FLEXIV_RAW_FORCE_STATE_SCHEMA
    assert root.attrs["source_state_dim"] == 48
    assert root.attrs["source_state_names"] == list(exporter.RAW_FORCE_STATE_FIELD_NAMES)
    assert root.attrs["state_transform"] == exporter.RAW_FORCE_CONVERTER_NAME
    assert root.attrs["dropped_state_names"] == list(exporter.FLEXIV_RAW_FORCE_DROPPED_STATE_NAMES)
    assert root.attrs["raw_source_state_sha256"] == raw_hash
    assert root.attrs["derived_state_sha256"] == derived_hash
    assert raw_hash != derived_hash
    assert root["data/state"].shape == (2, 34)
    assert root["data/action"].shape == (2, 14)
    exporter.verify_dp3_zarr(output)


def test_write_dp3_zarr_rejects_conflicting_source_metadata(tmp_path) -> None:
    output = tmp_path / "conflicting_source.zarr"
    with pytest.raises(ValueError, match="source metadata state_transform"):
        exporter.write_dp3_zarr(
            output,
            state=_v2_state(1),
            action=np.zeros((1, 14), dtype=np.float32),
            point_cloud=np.zeros((1, 8, 3), dtype=np.float32),
            episode_ends=np.asarray([1], dtype=np.int64),
            attrs={
                "source_state_schema": exporter.FLEXIV_LEGACY_STATE_SCHEMA,
                "state_transform": "passthrough_v2",
            },
        )
    assert not output.exists()


def test_verify_rejects_incomplete_export(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    output = tmp_path / "incomplete.zarr"
    root = zarr.group(str(output))
    root.attrs["export_status"] = "in_progress"

    with pytest.raises(ValueError, match="export is not complete"):
        exporter.verify_dp3_zarr(output)


def test_verify_rejects_modified_array(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    output = tmp_path / "modified.zarr"
    exporter.write_dp3_zarr(
        output,
        state=_v2_state(2),
        action=np.zeros((2, 14), dtype=np.float32),
        point_cloud=np.zeros((2, 4, 3), dtype=np.float32),
        episode_ends=np.asarray([2], dtype=np.int64),
        attrs={},
    )
    root = zarr.open(str(output), mode="a")
    root["data"]["action"][1, 0] = 1.0

    with pytest.raises(ValueError, match="checksum mismatch"):
        exporter.verify_dp3_zarr(output)
