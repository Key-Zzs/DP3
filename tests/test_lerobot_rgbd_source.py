from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPORTER_PATH = ROOT / "tools" / "export_lerobot_to_dp3_zarr.py"
EXPORTER_SPEC = importlib.util.spec_from_file_location("export_lerobot_to_dp3_zarr", EXPORTER_PATH)
exporter = importlib.util.module_from_spec(EXPORTER_SPEC)
sys.modules["export_lerobot_to_dp3_zarr"] = exporter
assert EXPORTER_SPEC.loader is not None
EXPORTER_SPEC.loader.exec_module(exporter)
source_api = exporter.rgbd_source

CAMERAS = tuple(exporter.CAMERA_SPECS)
T = 4
H = 2
W = 3
EPISODE_ENDS = np.asarray([2, 4], dtype=np.int64)


@dataclass
class SyntheticDataset:
    root: Path
    layout: str
    info: dict[str, Any]
    calibration: dict[str, Any]
    state: np.ndarray
    action: np.ndarray
    depth: dict[str, np.ndarray]
    left_ir: dict[str, np.ndarray]
    right_ir: dict[str, np.ndarray]
    rgb: np.ndarray
    parquet_path: Path
    manifest_path: Path | None
    zarr_path: Path | None


def _calibration() -> dict[str, Any]:
    camera = {
        "depth_scale_m_per_unit": 0.001,
        "streams": {
            name: {
                "width": W,
                "height": H,
                "intrinsics": {
                    "width": W,
                    "height": H,
                    "fx": 2.0,
                    "fy": 2.0,
                    "cx": 1.0,
                    "cy": 0.5,
                },
            }
            for name in ("depth", "color", "infrared1", "infrared2")
        },
        "extrinsics": {
            "depth_to_color": {
                "rotation_matrix_row_major": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "translation_m": [0.0, 0.0, 0.0],
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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _make_dataset(tmp_path: Path, *, layout: str, name: str = "dataset") -> SyntheticDataset:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    zarr = pytest.importorskip("zarr")
    assert layout in {"zarr", "parquet"}

    root = tmp_path / name
    calibration = _calibration()
    calibration_path = root / "meta/realsense_calibration.json"
    _write_json(calibration_path, calibration)
    info = {
        "repo_id": f"synthetic/{name}",
        "fps": 30,
        "total_frames": T,
        "total_episodes": 2,
        "features": {
            spec["video_key"]: {"shape": [H, W, 3]}
            for spec in exporter.CAMERA_SPECS.values()
        },
        "robot_state_schema": exporter.build_flexiv_state_schema(),
    }
    info["features"][exporter.STATE_COLUMN] = {
        "dtype": "float32",
        "shape": [exporter.STATE_DIM],
        "names": list(exporter.STATE_FIELD_NAMES),
    }
    info["features"][exporter.ACTION_COLUMN] = {
        "dtype": "float32",
        "shape": [exporter.ACTION_DIM],
        "names": list(exporter.ACTION_FIELD_NAMES),
    }
    _write_json(root / "meta/info.json", info)

    state = np.zeros((T, exporter.STATE_DIM), dtype=np.float32)
    state[:, :7] = np.arange(T * 7, dtype=np.float32).reshape(T, 7)
    state[:, 17:24] = np.arange(T * 7, dtype=np.float32).reshape(T, 7)
    state[:, 10:16] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    state[:, 27:33] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    state[:, 16] = 0.5
    state[:, 33] = 0.5
    action = np.arange(T * exporter.ACTION_DIM, dtype=np.float32).reshape(T, exporter.ACTION_DIM)
    index = np.arange(T, dtype=np.int64)
    episode_index = np.asarray([0, 0, 1, 1], dtype=np.int64)
    frame_index = np.asarray([0, 1, 0, 1], dtype=np.int64)
    global_frame_index = np.asarray([10, 11, 20, 21], dtype=np.int64)
    robot_timestamp = np.asarray([100.0, 100.1, 101.0, 101.1], dtype=np.float64)
    depth = {
        camera: (
            np.arange(T * H * W, dtype=np.uint16).reshape(T, H, W)
            + 100
            + camera_index * 100
        )
        for camera_index, camera in enumerate(CAMERAS)
    }
    left_ir = {
        camera: (
            np.arange(T * H * W, dtype=np.uint8).reshape(T, H, W)
            + camera_index * 30
        )
        for camera_index, camera in enumerate(CAMERAS)
    }
    right_ir = {
        camera: (values + 7).astype(np.uint8)
        for camera, values in left_ir.items()
    }
    camera_timestamps = {
        camera: robot_timestamp + 0.001 * (camera_index + 1)
        for camera_index, camera in enumerate(CAMERAS)
    }
    camera_reused = {
        camera: np.asarray([False, False, True, False], dtype=np.bool_)
        for camera in CAMERAS
    }
    rgb = np.stack(
        [np.full((H, W, 3), 30 + frame * 20, dtype=np.uint8) for frame in range(T)],
        axis=0,
    )

    columns: dict[str, Any] = {
        exporter.STATE_COLUMN: state.tolist(),
        exporter.ACTION_COLUMN: action.tolist(),
        "index": index.tolist(),
        "episode_index": episode_index.tolist(),
        "frame_index": frame_index.tolist(),
        "global_frame_index": global_frame_index.tolist(),
        "robot_timestamp": robot_timestamp.tolist(),
    }
    for camera in CAMERAS:
        spec = exporter.CAMERA_SPECS[camera]
        columns[spec["timestamp_column"]] = camera_timestamps[camera].tolist()
        columns[spec["reused_column"]] = camera_reused[camera].tolist()
        if layout == "parquet":
            image_type_u16 = pa.list_(pa.list_(pa.uint16()))
            image_type_u8 = pa.list_(pa.list_(pa.uint8()))
            columns[spec["depth_column"]] = pa.array(depth[camera].tolist(), type=image_type_u16)
            columns[spec["left_ir_column"]] = pa.array(
                left_ir[camera].tolist(), type=image_type_u8
            )
            columns[spec["right_ir_column"]] = pa.array(
                right_ir[camera].tolist(), type=image_type_u8
            )
    parquet_path = root / "data/chunk-000/file-000.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), parquet_path, row_group_size=2)

    episode_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1],
                "length": [2, 2],
                "dataset_from_index": [0, 2],
                "dataset_to_index": [2, 4],
            }
        ),
        episode_path,
    )

    manifest_path = None
    zarr_path = None
    if layout == "zarr":
        zarr_path = root / source_api.RAW_ZARR_RELATIVE_PATH
        compressor = zarr.Blosc(cname="zstd", clevel=1, shuffle=1)
        zroot = zarr.group(str(zarr_path))
        zroot.attrs.update(
            {
                "schema_name": source_api.RAW_SCHEMA_NAME,
                "schema_version": source_api.RAW_SCHEMA_VERSION,
                "storage": source_api.RAW_STORAGE,
                "relative_path": source_api.RAW_ZARR_RELATIVE_PATH.as_posix(),
            }
        )
        arrays: dict[str, Any] = {}

        def create(path: str, data: np.ndarray, chunks: tuple[int, ...]) -> None:
            group_path, name_part = path.rsplit("/", 1)
            group = zroot.require_group(group_path)
            array = group.create_dataset(
                name_part,
                data=data,
                chunks=chunks,
                dtype=data.dtype,
                compressor=compressor,
            )
            entry: dict[str, Any] = {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "chunks": list(array.chunks),
                "compressor": array.compressor.get_config(),
            }
            arrays[f"/{path}"] = entry

        create("meta/index", index, (2,))
        create("meta/episode_index", episode_index, (2,))
        create("meta/frame_index", frame_index, (2,))
        create("meta/global_frame_index", global_frame_index, (2,))
        create("meta/robot_timestamp", robot_timestamp, (2,))
        create("meta/episode_ends", EPISODE_ENDS, (2,))
        for camera in CAMERAS:
            create(f"data/{camera}/depth", depth[camera], (2, H, W))
            create(f"data/{camera}/left_ir", left_ir[camera], (2, H, W))
            create(f"data/{camera}/right_ir", right_ir[camera], (2, H, W))
            create(f"data/{camera}/rgbd_timestamp", camera_timestamps[camera], (2,))
            create(f"data/{camera}/rgbd_reused", camera_reused[camera], (2,))

        calibration_sha256 = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        manifest = {
            "schema_name": source_api.RAW_SCHEMA_NAME,
            "schema_version": source_api.RAW_SCHEMA_VERSION,
            "storage": source_api.RAW_STORAGE,
            "relative_path": source_api.RAW_ZARR_RELATIVE_PATH.as_posix(),
            "status": "complete",
            "committed_frames": T,
            "committed_episodes": 2,
            "cameras": list(CAMERAS),
            "modalities": ["depth", "left_ir", "right_ir"],
            "frame_shape": {"height": H, "width": W},
            "row_semantics": (
                "Zarr row ordinal i joins LeRobot Parquet index == i; readers must also "
                "verify episode_index, frame_index, global_frame_index, robot/per-camera "
                "timestamps, and reused."
            ),
            "commit_semantics": "Only the manifest committed prefix is readable.",
            "depth_units": (
                "Native RealSense uint16 depth units; values are not multiplied by "
                "depth_scale_m_per_unit."
            ),
            "calibration": {
                "relative_path": "meta/realsense_calibration.json",
                "sha256": calibration_sha256,
            },
            "arrays": arrays,
        }
        manifest_path = root / source_api.RAW_MANIFEST_RELATIVE_PATH
        _write_json(manifest_path, manifest)

    return SyntheticDataset(
        root=root,
        layout=layout,
        info=info,
        calibration=calibration,
        state=state,
        action=action,
        depth=depth,
        left_ir=left_ir,
        right_ir=right_ir,
        rgb=rgb,
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        zarr_path=zarr_path,
    )


def _open(dataset: SyntheticDataset, selection: str = "auto") -> Any:
    return source_api.open_rgbd_sidecar_source(
        dataset.root,
        source=selection,
        info=dataset.info,
        parquet_row_count=T,
        total_episodes=2,
    )


def _builder_config(dataset: SyntheticDataset, path: Path, *, mode: str) -> Path:
    config = exporter.build_pointcloud_builder_config(
        dataset.calibration,
        camera="head",
        pointcloud_mode=mode,
        num_points=H * W,
    )
    config["device"] = "cpu"
    config["sampling"] = {
        "enabled": True,
        "mode": "stride",
        "stride": 1,
        "num_points": H * W,
        "pad_mode": "repeat",
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _export_args(
    dataset: SyntheticDataset,
    output: Path,
    builder_config: Path,
    *,
    source: str = "auto",
    mode: str = "xyz",
    max_frames: int | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        lerobot_path=str(dataset.root.resolve()),
        output_zarr=str(output),
        camera="head",
        rgbd_sidecar_source=source,
        pointcloud_mode=mode,
        num_points=H * W,
        builder_config=str(builder_config),
        overwrite=False,
        max_frames=max_frames,
        save_img=False,
        verbose=False,
    )


def test_valid_zarr_manifest_store_and_three_camera_ir_pair(tmp_path) -> None:
    dataset = _make_dataset(tmp_path, layout="zarr")
    source = _open(dataset)
    assert isinstance(source, source_api.ZarrRGBDSource)
    assert source.storage == "zarr_v2"
    source.validate_join([dataset.parquet_path], camera="head", batch_size=2)

    for camera in CAMERAS:
        depth = source.get_depth(camera, 2)
        pair = source.get_ir_pair(camera, 2)
        assert depth.dtype == np.uint16 and depth.shape == (H, W)
        assert pair.left_ir.dtype == np.uint8 and pair.left_ir.shape == (H, W)
        assert pair.right_ir.dtype == np.uint8 and pair.right_ir.shape == (H, W)
        np.testing.assert_array_equal(depth, dataset.depth[camera][2])
        np.testing.assert_array_equal(pair.left_ir, dataset.left_ir[camera][2])
        np.testing.assert_array_equal(pair.right_ir, dataset.right_ir[camera][2])
        assert pair.calibration_path == dataset.root / "meta/realsense_calibration.json"
        assert len(pair.calibration_sha256) == 64

    frames = list(
        source.iter_frames(
            [dataset.parquet_path],
            camera="left_wrist",
            columns=[
                "index",
                "episode_index",
                "frame_index",
                "global_frame_index",
                "left_wrist_rgbd_timestamp",
                "left_wrist_rgbd_reused",
            ],
            max_frames=T,
            include_ir=True,
            batch_size=2,
        )
    )
    assert [frame.row_index for frame in frames] == [0, 1, 2, 3]
    np.testing.assert_array_equal(frames[3].depth, dataset.depth["left_wrist"][3])
    assert frames[3].ir_pair is not None
    np.testing.assert_array_equal(frames[3].ir_pair["left_ir"], dataset.left_ir["left_wrist"][3])


def test_legacy_parquet_auto_and_ir_compatibility(tmp_path) -> None:
    dataset = _make_dataset(tmp_path, layout="parquet")
    source = _open(dataset)
    assert isinstance(source, source_api.LegacyParquetRGBDSource)
    source.validate_join([dataset.parquet_path], camera="right_wrist", batch_size=2)
    frame = source.read_frame_at(
        [dataset.parquet_path],
        camera="right_wrist",
        row_index=1,
        columns=[
            "index",
            "episode_index",
            "frame_index",
            "global_frame_index",
            "right_wrist_rgbd_timestamp",
            "right_wrist_rgbd_reused",
        ],
        include_ir=True,
    )
    np.testing.assert_array_equal(frame.depth, dataset.depth["right_wrist"][1])
    assert frame.ir_pair is not None
    np.testing.assert_array_equal(frame.ir_pair.right_ir, dataset.right_ir["right_wrist"][1])


@pytest.mark.parametrize("selection", ["auto", "zarr"])
def test_zarr_source_selection(tmp_path, selection) -> None:
    dataset = _make_dataset(tmp_path, layout="zarr")
    assert _open(dataset, selection).storage == "zarr_v2"


def test_explicit_source_layout_mismatch(tmp_path) -> None:
    zarr_dataset = _make_dataset(tmp_path, layout="zarr", name="zarr_dataset")
    parquet_dataset = _make_dataset(tmp_path, layout="parquet", name="parquet_dataset")
    with pytest.raises(ValueError, match="conflicts with the authoritative"):
        _open(zarr_dataset, "parquet")
    with pytest.raises(FileNotFoundError, match="requires manifest"):
        _open(parquet_dataset, "zarr")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("status", "incomplete", "status"),
        ("schema_name", "unknown", "schema_name"),
        ("schema_version", 2, "schema_version"),
        ("storage", "zarr_v3", "storage"),
    ],
)
def test_manifest_identity_fail_fast(tmp_path, field, value, match) -> None:
    dataset = _make_dataset(tmp_path, layout="zarr")
    assert dataset.manifest_path is not None
    manifest = json.loads(dataset.manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    _write_json(dataset.manifest_path, manifest)
    with pytest.raises(ValueError, match=match):
        _open(dataset)


def test_calibration_hash_mismatch(tmp_path) -> None:
    dataset = _make_dataset(tmp_path, layout="zarr")
    assert dataset.manifest_path is not None
    manifest = json.loads(dataset.manifest_path.read_text(encoding="utf-8"))
    manifest["calibration"]["sha256"] = "0" * 64
    _write_json(dataset.manifest_path, manifest)
    with pytest.raises(ValueError, match="Calibration SHA-256 mismatch"):
        _open(dataset)


@pytest.mark.parametrize("count_source", ["info", "manifest"])
def test_committed_scalar_count_mismatch(tmp_path, count_source) -> None:
    dataset = _make_dataset(tmp_path, layout="zarr")
    if count_source == "info":
        dataset.info["total_frames"] = T - 1
        with pytest.raises(ValueError, match="Parquet row count"):
            _open(dataset)
    else:
        assert dataset.manifest_path is not None
        manifest = json.loads(dataset.manifest_path.read_text(encoding="utf-8"))
        manifest["committed_frames"] = T - 1
        _write_json(dataset.manifest_path, manifest)
        with pytest.raises(ValueError, match="committed_frames"):
            _open(dataset)


def test_array_length_mismatch(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    dataset = _make_dataset(tmp_path, layout="zarr")
    assert dataset.zarr_path is not None
    root = zarr.open(str(dataset.zarr_path), mode="a")
    root["data/head/depth"].resize((T - 1, H, W))
    with pytest.raises(ValueError, match="axis-0 length"):
        _open(dataset)


def test_missing_camera_modality(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    dataset = _make_dataset(tmp_path, layout="zarr")
    assert dataset.zarr_path is not None
    root = zarr.open(str(dataset.zarr_path), mode="a")
    del root["data/right_wrist/right_ir"]
    with pytest.raises(KeyError, match="right_wrist/right_ir"):
        _open(dataset)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("meta/index", 99, "index ordinal"),
        ("meta/episode_index", 5, "episode_index"),
        ("meta/frame_index", 5, "frame_index"),
        ("meta/global_frame_index", 10, "global_frame_index"),
    ],
)
def test_index_order_mismatch(tmp_path, path, value, match) -> None:
    zarr = pytest.importorskip("zarr")
    dataset = _make_dataset(tmp_path, layout="zarr")
    assert dataset.zarr_path is not None
    root = zarr.open(str(dataset.zarr_path), mode="a")
    root[path][1] = value
    with pytest.raises(ValueError, match=match):
        _open(dataset)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("meta/robot_timestamp", 999.0, "robot_timestamp"),
        ("data/head/rgbd_timestamp", 100.0005, "head_rgbd_timestamp"),
        ("data/head/rgbd_reused", True, "head_rgbd_reused"),
    ],
)
def test_timestamp_and_reused_join_mismatch(tmp_path, path, value, match) -> None:
    zarr = pytest.importorskip("zarr")
    dataset = _make_dataset(tmp_path, layout="zarr")
    assert dataset.zarr_path is not None
    root = zarr.open(str(dataset.zarr_path), mode="a")
    root[path][0] = value
    source = _open(dataset)
    with pytest.raises(ValueError, match=match):
        source.validate_join([dataset.parquet_path], camera="head", batch_size=2)


@pytest.mark.parametrize("episode_ends", [[2, 3], [4, 4], [4]])
def test_malformed_episode_ends(tmp_path, episode_ends) -> None:
    zarr = pytest.importorskip("zarr")
    dataset = _make_dataset(tmp_path, layout="zarr")
    assert dataset.zarr_path is not None
    root = zarr.open(str(dataset.zarr_path), mode="a")
    array = root["meta/episode_ends"]
    values = np.asarray(episode_ends, dtype=np.int64)
    array.resize(values.shape)
    array[:] = values
    with pytest.raises(ValueError, match="episode_ends"):
        _open(dataset)


def test_zarr_and_parquet_export_equivalence_and_max_frames(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    zarr_dataset = _make_dataset(tmp_path, layout="zarr", name="zarr_input")
    parquet_dataset = _make_dataset(tmp_path, layout="parquet", name="parquet_input")
    zarr_config = _builder_config(zarr_dataset, tmp_path / "zarr_builder.yaml", mode="xyz")
    parquet_config = _builder_config(
        parquet_dataset, tmp_path / "parquet_builder.yaml", mode="xyz"
    )
    zarr_output = tmp_path / "zarr_output.zarr"
    parquet_output = tmp_path / "parquet_output.zarr"

    exporter.export_lerobot_to_dp3_zarr(
        _export_args(zarr_dataset, zarr_output, zarr_config, max_frames=3)
    )
    exporter.export_lerobot_to_dp3_zarr(
        _export_args(parquet_dataset, parquet_output, parquet_config, max_frames=3)
    )

    zarr_root = zarr.open(str(zarr_output), mode="r")
    parquet_root = zarr.open(str(parquet_output), mode="r")
    for key in ("state", "action", "point_cloud"):
        np.testing.assert_allclose(zarr_root[f"data/{key}"][:], parquet_root[f"data/{key}"][:])
    assert zarr_root["meta/episode_ends"][:].tolist() == [2, 3]
    assert parquet_root["meta/episode_ends"][:].tolist() == [2, 3]
    assert zarr_root.attrs["source_sidecar_storage"] == "zarr_v2"
    assert parquet_root.attrs["source_sidecar_storage"] == "parquet"
    assert zarr_root.attrs["source_sidecar_schema_name"] == source_api.RAW_SCHEMA_NAME
    assert len(zarr_root.attrs["source_sidecar_manifest_sha256"]) == 64
    assert zarr_root.attrs["raw_depth_scale_m_per_unit"] == 0.001
    assert zarr_root.attrs["state_schema"] == "flexiv_abs_rot6d_v2"
    assert zarr_root.attrs["state_dim"] == 34
    assert zarr_root.attrs["action_dim"] == 14
    assert zarr_root.attrs["rotation6d_convention"] == "matrix_columns_0_1"
    assert zarr_root.attrs["rotation6d_order"] == ["c0x", "c0y", "c0z", "c1x", "c1y", "c1z"]
    assert zarr_root.attrs["action_rotation_representation"] == "rotvec"
    assert zarr_root.attrs["source_state_schema"] == "flexiv_abs_rot6d_v2"
    assert zarr_root.attrs["state_transform"] == "passthrough_v2"
    assert len(zarr_root.attrs["raw_source_state_sha256"]) == 64
    assert len(zarr_root.attrs["derived_state_sha256"]) == 64
    assert "left_ir" not in zarr_root["data"] and "right_ir" not in zarr_root["data"]
    assert exporter.verify_dp3_zarr(zarr_output) == zarr_root.attrs["integrity"]


def test_zarr_native_depth_xyzrgb_export(tmp_path, monkeypatch) -> None:
    zarr = pytest.importorskip("zarr")
    dataset = _make_dataset(tmp_path, layout="zarr")
    config = _builder_config(dataset, tmp_path / "xyzrgb_builder.yaml", mode="xyzrgb")
    output = tmp_path / "xyzrgb_output.zarr"
    monkeypatch.setattr(exporter, "_video_paths", lambda *_args, **_kwargs: [tmp_path / "rgb.mp4"])
    monkeypatch.setattr(exporter, "iter_video_frames", lambda _paths: iter(dataset.rgb))

    exporter.export_lerobot_to_dp3_zarr(
        _export_args(dataset, output, config, source="zarr", mode="xyzrgb")
    )
    root = zarr.open(str(output), mode="r")
    assert root["data/point_cloud"].shape == (T, H * W, 6)
    colors = root["data/point_cloud"][0, :, 3:]
    assert np.all(colors >= 0.0) and np.all(colors <= 1.0)
    assert root.attrs["depth_source"] == "native_depth"


def test_inspector_validates_and_reports_source_provenance(tmp_path) -> None:
    dataset = _make_dataset(tmp_path, layout="zarr")
    config = _builder_config(dataset, tmp_path / "inspect_builder.yaml", mode="xyz")
    output = tmp_path / "inspect_output.zarr"
    exporter.export_lerobot_to_dp3_zarr(_export_args(dataset, output, config))

    inspect_path = ROOT / "tools/inspect_dp3_zarr.py"
    spec = importlib.util.spec_from_file_location("inspect_dp3_zarr", inspect_path)
    inspector = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(inspector)
    summary = inspector.inspect_dp3_zarr(output)
    assert summary["source_provenance"]["source_sidecar_storage"] == "zarr_v2"
    assert summary["source_provenance"]["depth_source"] == "native_depth"


def test_invalid_join_leaves_no_partial_output(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    dataset = _make_dataset(tmp_path, layout="zarr")
    assert dataset.zarr_path is not None
    source_root = zarr.open(str(dataset.zarr_path), mode="a")
    source_root["data/head/rgbd_timestamp"][0] = 100.0005
    config = _builder_config(dataset, tmp_path / "builder.yaml", mode="xyz")
    output = tmp_path / "must_not_exist.zarr"

    with pytest.raises(ValueError, match="head_rgbd_timestamp"):
        exporter.export_lerobot_to_dp3_zarr(_export_args(dataset, output, config))
    assert not output.exists()
    assert not list(tmp_path.glob(".must_not_exist.zarr.incomplete-*"))


def test_debug_reads_zarr_frame_through_unified_source(tmp_path) -> None:
    debug_path = ROOT / "tools/debug_lerobot_pointcloud_stages.py"
    spec = importlib.util.spec_from_file_location("debug_lerobot_pointcloud_stages", debug_path)
    debug = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(debug)

    dataset = _make_dataset(tmp_path, layout="zarr")
    args = debug.DebugInputs()
    args.lerobot_path = dataset.root.resolve()
    args.frame_index = 2
    args.camera = "head"
    args.rgbd_sidecar_source = "auto"
    args.pointcloud_mode = "xyz"
    args.num_points = H * W
    args.builder_config = None
    args.dp3_zarr = None
    args.temp_config_path = tmp_path / "debug_builder.yaml"

    stages, _meta, row, _config_path = debug._build_frame_stages(args)
    assert row["_source_sidecar_storage"] == "zarr_v2"
    assert row["_source_manifest_path"] == str(dataset.manifest_path.resolve())
    assert tuple(stages["sampled"].shape) == (H * W, 3)
