"""
nuspacesim
----------

A Simulator for Space-based Neutrino Detections.
"""

import setuptools  # noqa: F401
from numpy.distutils.core import setup  # , Extension
from pybind11.setup_helpers import Pybind11Extension

nssgeometry = Pybind11Extension(
    "simulation.geometry.nssgeometry",
    ["src/nuspacesim/simulation/geometry/src/nssgeometry.cpp"],
    cxx_std=17,
)

zsteps = Pybind11Extension(
    "simulation.eas_optical.zsteps",
    ["src/nuspacesim/simulation/eas_optical/src/zsteps.cpp"],
    cxx_std=17,
)

# EXT1 = Extension(
#     name="simulation.eas_optical.subcphotang",
#     sources=["src/nuspacesim/simulation/eas_optical/src/subCphotAng.f"],
#     extra_f90_compile_args=["-fopenmp"],
#     extra_link_args=["-fopenmp"],
# )

setup(ext_modules=[nssgeometry, zsteps])
