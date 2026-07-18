from setuptools import find_packages, setup

setup(
    name="visualizer",
    version="0.1.0",
    packages=find_packages(),
    extras_require={
        "monitor": [
            "rerun-sdk==0.34.1",
            "psutil>=5.9",
        ],
    },
)
