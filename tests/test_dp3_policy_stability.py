from __future__ import annotations

import json
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))
sys.path.insert(0, str(ROOT / "tools"))

import analyze_dp3_policy_stability as stability  # noqa: E402


def _synthetic_dataset() -> stability.ZarrDataset:
    state = np.zeros((12, 34), dtype=np.float32)
    for base in (10, 27):
        state[:, base] = 1.0
        state[:, base + 4] = 1.0
    point_cloud = np.zeros((12, 2048, 3), dtype=np.float32)
    action = np.zeros((12, 14), dtype=np.float32)
    return stability.ZarrDataset(
        path=Path("synthetic.zarr"),
        point_cloud=point_cloud,
        state=state,
        action=action,
        episode_ends=np.asarray([6, 12], dtype=np.int64),
        attrs={},
    )


def test_static_window_history_does_not_cross_episode() -> None:
    dataset = _synthetic_dataset()
    metrics = stability.compute_motion_metrics(dataset.state, dataset.action)
    windows, _ = stability.find_static_windows(dataset, metrics, joint_change_threshold=0.01)
    assert windows[0].start >= 1
    assert windows[1].start >= 7
    selected, count = stability.select_static_window(windows, requested_samples=100, minimum_samples=2)
    assert selected.episode_index == 0
    assert count == 5
    plan = stability.build_experiment_plan(selected, sample_count=count, seed_base=0)
    for case in plan:
        assert stability.validate_observation_indices(case.obs_frame_indices, episode_ends=dataset.episode_ends) == case.episode_index


def test_grasp_motion_window_is_anchored_on_right_gripper_close() -> None:
    dataset = _synthetic_dataset()
    dataset.action[:, 13] = 1.0
    dataset.action[3:6, 13] = np.asarray([0.8, 0.2, 0.0], dtype=np.float32)
    dataset.action[9:12, 13] = np.asarray([0.9, 0.7, 0.6], dtype=np.float32)
    dataset.action[1:6, 6] = 0.002
    metrics = stability.compute_motion_metrics(dataset.state, dataset.action)
    windows, metadata = stability.find_grasp_motion_windows(
        dataset,
        metrics,
        requested_samples=5,
    )
    assert metadata["mode"] == "grasp_motion"
    assert windows[0].anchor_frame == 4
    assert windows[0].start >= 1
    assert windows[0].end <= 6
    selected, count = stability.select_grasp_motion_window(windows, minimum_samples=5)
    assert selected.episode_index == 0
    assert count == 5


class _RandomPolicy:
    def predict_action(self, observation):
        del observation
        return {
            "action": torch.randn(1, 4, 14),
            "action_pred": torch.randn(1, 8, 14),
        }


def test_fixed_seed_repeats_without_global_rng_pollution() -> None:
    policy = _RandomPolicy()
    observation = {"point_cloud": torch.zeros(1, 2, 2048, 3), "agent_pos": torch.zeros(1, 2, 34)}
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    first = stability.run_policy_once(
        policy, observation, seed=7, device=torch.device("cpu"), action_steps=4, action_dim=14, horizon=8
    )
    after = torch.random.get_rng_state()
    second = stability.run_policy_once(
        policy, observation, seed=7, device=torch.device("cpu"), action_steps=4, action_dim=14, horizon=8
    )
    different = stability.run_policy_once(
        policy, observation, seed=8, device=torch.device("cpu"), action_steps=4, action_dim=14, horizon=8
    )
    assert torch.equal(before, after)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[1], different[1])


def test_four_groups_pair_seed_and_observation_correctly() -> None:
    window = stability.StaticWindow(episode_index=2, start=10, end=14, eligible_count=4)
    plan = stability.build_experiment_plan(window, sample_count=4, seed_base=100)
    by_group = {group: [case for case in plan if case.group == group] for group in stability.GROUPS}
    assert [case.seed for case in by_group["A"]] == [100, 101, 102, 103]
    assert [case.seed for case in by_group["B"]] == [100] * 4
    assert [case.seed for case in by_group["C"]] == [100] * 4
    assert [case.seed for case in by_group["D"]] == [100, 101, 102, 103]
    assert len({case.obs_frame_indices for case in by_group["A"]}) == 1
    assert len({case.obs_frame_indices for case in by_group["B"]}) == 1
    assert len({case.obs_frame_indices for case in by_group["C"]}) == 4
    assert len({case.obs_frame_indices for case in by_group["D"]}) == 4


def test_policy_result_shape_validation() -> None:
    result = {"action": torch.zeros(1, 4, 14), "action_pred": torch.zeros(1, 8, 14)}
    stability.validate_policy_result(result, action_steps=4, action_dim=14, horizon=8)
    with pytest.raises(ValueError, match="action_pred.*!="):
        stability.validate_policy_result(
            {"action": result["action"], "action_pred": torch.zeros(1, 7, 14)},
            action_steps=4,
            action_dim=14,
            horizon=8,
        )


def _deployment_records(group: str, count: int = 9) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for sample_index in range(count):
        prediction = np.zeros((8, 14), dtype=np.float32)
        prediction[:, 6] = sample_index + np.arange(8, dtype=np.float32)
        records.append(
            {
                "group": group,
                "sample_index": sample_index,
                "obs_frame_indices": [sample_index, sample_index + 1],
                "action": prediction[1:5].copy(),
                "action_pred": prediction,
            }
        )
    return records


def test_chunk_seams_use_full_action_stride_for_varying_observations() -> None:
    records: list[dict[str, object]] = []
    for group in stability.GROUPS:
        for sample_index in range(9):
            chunk = np.zeros((4, 14), dtype=np.float32)
            chunk[:, 6] = float(sample_index)
            records.append(
                {
                    "group": group,
                    "sample_index": sample_index,
                    "action": chunk,
                }
            )
    summary = stability.summarize_chunk_seams(records, action_steps=4)
    assert summary["A"]["pairing"]["boundary_stride_samples"] == 1
    assert summary["C"]["pairing"]["boundary_stride_samples"] == 4
    assert summary["C"]["pairing"]["boundary_pair_count"] == 5
    assert summary["C"]["right_xyz"]["boundary"]["p50"] == pytest.approx(4.0)


def test_temporal_alignment_matches_old_tail_to_new_head() -> None:
    records = _deployment_records("C") + _deployment_records("D")
    summary = stability.summarize_temporal_alignment(
        records,
        n_obs_steps=2,
        action_steps=4,
        horizon=8,
    )
    assert summary["C"]["contract"]["overlap_steps"] == 3
    assert summary["C"]["num_chunk_pairs"] == 5
    assert summary["C"]["aligned_future_delta_disagreement"]["right_xyz"]["p95"] == pytest.approx(0.0)
    assert summary["C"]["unaligned_same_relative_step_disagreement"]["right_xyz"]["p95"] == pytest.approx(4.0)


def test_temporal_ensemble_blends_only_aligned_overlap() -> None:
    records: list[dict[str, object]] = []
    for group in ("C", "D"):
        for sample_index in range(5):
            prediction = np.zeros((8, 14), dtype=np.float32)
            if sample_index == 4:
                prediction[1:5, 6] = 2.0
                prediction[1:5, 13] = 1.0
            records.append(
                {
                    "group": group,
                    "sample_index": sample_index,
                    "obs_frame_indices": [sample_index, sample_index + 1],
                    "action": prediction[1:5].copy(),
                    "action_pred": prediction,
                }
            )
    summary = stability.summarize_temporal_ensemble(
        records,
        dataset_action=np.zeros((20, 14), dtype=np.float32),
        episode_ends=np.asarray([20], dtype=np.int64),
        n_obs_steps=2,
        action_steps=4,
        horizon=8,
        new_prediction_weights=(0.5,),
    )
    baseline = summary["C"]["baseline_new_prediction"]
    candidate = summary["C"]["candidates"]["new_weight_0.5"]
    pose_only = summary["C"]["candidates"]["pose_only_new_weight_0.5"]
    ramp = summary["C"]["candidates"]["pose_only_ramp_new_weight_0.5"]
    assert baseline["boundary_jump"]["right_xyz"]["p95"] == pytest.approx(2.0)
    assert candidate["boundary_jump"]["right_xyz"]["p95"] == pytest.approx(1.0)
    assert pose_only["boundary_jump"]["right_xyz"]["p95"] == pytest.approx(1.0)
    assert pose_only["boundary_jump"]["right_gripper"]["p95"] == pytest.approx(
        baseline["boundary_jump"]["right_gripper"]["p95"]
    )
    assert ramp["new_prediction_weights"] == pytest.approx([0.5, 0.75, 1.0])
    assert ramp["old_prediction_weights"] == pytest.approx([0.5, 0.25, 0.0])
    assert ramp["boundary_jump"]["right_xyz"]["p95"] == pytest.approx(1.0)
    assert ramp["boundary_jump"]["right_gripper"]["p95"] == pytest.approx(
        baseline["boundary_jump"]["right_gripper"]["p95"]
    )


def test_output_schema_contains_outputs_and_provenance(tmp_path: Path) -> None:
    records = [
        {
            "group": "A",
            "sample_index": 0,
            "obs_frame_indices": [1, 2],
            "episode_index": 0,
            "seed": 0,
            "policy_latency_sec": 0.01,
            "action": np.zeros((4, 14), dtype=np.float32),
            "action_pred": np.zeros((8, 14), dtype=np.float32),
            "pointcloud_summary": np.zeros((2, 12), dtype=np.float32),
            "state_summary": np.zeros((2, 136), dtype=np.float32),
        }
    ]
    provenance = {"test": True}
    summary = {"test": True}
    stability.save_outputs(tmp_path, records, summary, provenance)
    arrays = np.load(tmp_path / "samples.npz")
    assert arrays["action"].shape == (1, 4, 14)
    assert arrays["action_pred"].shape == (1, 8, 14)
    line = json.loads((tmp_path / "samples.jsonl").read_text().splitlines()[0])
    assert line["action_shape"] == [4, 14]
    assert json.loads((tmp_path / "provenance.json").read_text())["test"] is True


def test_empty_static_interval_fails_fast() -> None:
    window = stability.StaticWindow(episode_index=0, start=0, end=4, eligible_count=4)
    with pytest.raises(RuntimeError, match="at least 20"):
        stability.select_static_window([window], requested_samples=100, minimum_samples=20)


def test_tool_has_no_flexiv_rdk_or_hardware_calls() -> None:
    source = (ROOT / "tools/analyze_dp3_policy_stability.py").read_text()
    tree = ast.parse(source)
    imported_modules = {
        node.names[0].name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and node.names
    }
    imported_modules.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert "flexivrdk" not in imported_modules
    assert "robot.connect(" not in source
    assert "camera.connect(" not in source
    assert "send_action(" not in source
