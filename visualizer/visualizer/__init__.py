"""Optional visualizer package with a dependency-light monitor subpackage."""

__all__ = ["Visualizer", "visualize_pointcloud"]


def __getattr__(name: str):
    if name in __all__:
        from .pointcloud import Visualizer, visualize_pointcloud

        return {"Visualizer": Visualizer, "visualize_pointcloud": visualize_pointcloud}[name]
    raise AttributeError(name)
