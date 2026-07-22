from __future__ import annotations

import pytest


def open_flexiv_env(FlexivEmbodimentSmoke):
    """Open the visual SAPIEN env or report an honest host-level skip."""

    try:
        return FlexivEmbodimentSmoke(gui=False, seed=0)
    except RuntimeError as exc:
        detail = str(exc)
        if "vk::PhysicalDevice" in detail or "Vulkan" in detail:
            pytest.skip(f"SAPIEN Vulkan device unavailable: {detail}")
        raise
