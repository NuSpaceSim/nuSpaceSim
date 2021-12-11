"""
nuspacesim
----------

A Simulator for Space-based Neutrino Detections.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

zsteps = Pybind11Extension(
    "simulation.eas_optical.zsteps",
    ["src/nuspacesim/simulation/eas_optical/src/zsteps.cpp"],
)

setup(cmdclass={"build_ext": build_ext}, ext_modules=[zsteps])
