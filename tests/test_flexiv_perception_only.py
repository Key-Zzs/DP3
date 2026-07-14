from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_flexiv_dp3_perception_only import (  # noqa: E402
    Runtime,
    _depth_stats,
    _summarize_records,
)


def _runtime(**overrides):
    values = {
        "camera_serial": "1234",
        "stability_window": 3,
        "min_valid_depth_ratio": 0.75,
        "max_valid_depth_ratio_range": 0.08,
    }
    values.update(overrides)
    return Runtime(**values)


def _record(ratio: float, *, owns_data: bool = True, padded: bool = False):
    return {
        "depth": {"valid_ratio": ratio},
        "buffer": {"depth_owns_data": owns_data, "depth_c_contiguous": True},
        "point_cloud": {
            "num_raw_points": int(307200 * ratio),
            "num_cropped_points": int(120000 * ratio),
            "padded": padded,
        },
    }


def test_depth_stats_counts_only_finite_positive_values() -> None:
    depth = np.array([[0, 1000], [2000, 65535]], dtype=np.uint16)

    stats = _depth_stats(depth, depth_scale=0.001)

    assert stats["valid_count"] == 3
    assert stats["invalid_count"] == 1
    assert stats["valid_ratio"] == 0.75
    assert stats["min_m"] == 1.0
    assert stats["max_m"] == 65.535


def test_quality_gate_passes_stable_owned_depth() -> None:
    summary = _summarize_records(
        [_record(0.88), _record(0.90), _record(0.89)],
        _runtime(),
        interrupted=False,
    )

    assert summary["quality_gate"]["passed"]
    assert summary["quality_gate"]["failures"] == []


def test_quality_gate_rejects_low_ratio_and_non_owned_depth() -> None:
    summary = _summarize_records(
        [_record(0.61), _record(0.60, owns_data=False), _record(0.62)],
        _runtime(),
        interrupted=False,
    )

    assert not summary["quality_gate"]["passed"]
    failures = " ".join(summary["quality_gate"]["failures"])
    assert "depth valid ratio median" in failures
    assert "did not own its memory" in failures
