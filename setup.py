"""
nuspacesim
----------

A Simulator for Space-based Neutrino Detections.
"""
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

nssgeometry = Pybind11Extension(
    "nssgeometry.nssgeometry", ["src/nuspacesim/nssgeometry/src/nssgeometry.cpp"]
)
zsteps = Pybind11Extension(
    "EAScherGen.zsteps", ["src/nuspacesim/EAScherGen/src/zsteps.cpp"]
)

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=[nssgeometry, zsteps],
)
