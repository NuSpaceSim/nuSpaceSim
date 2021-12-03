"""
nuspacesim
----------

A Simulator for Space-based Neutrino Detections.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# nssgeometry = Pybind11Extension(
#     "simulation.geometry.nssgeometry",
#     ["test/simulation/geometry/src/nssgeometry.cpp"],
# )

zsteps = Pybind11Extension(
    "simulation.eas_optical.zsteps",
    ["src/nuspacesim/simulation/eas_optical/src/zsteps.cpp"],
)

# setup(cmdclass={"build_ext": build_ext}, ext_modules=[nssgeometry, zsteps])
setup(cmdclass={"build_ext": build_ext}, ext_modules=[zsteps])
