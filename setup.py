"""
nuspacesim
----------

A Simulator for Space-based Neutrino Detections.
"""
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

nssgeometry = Pybind11Extension(
    "modules.geometry.nssgeometry",
    ["src/nuspacesim/modules/geometry/src/nssgeometry.cpp"],
)
zsteps = Pybind11Extension(
    "modules.eas_optical.zsteps", ["src/nuspacesim/modules/eas_optical/src/zsteps.cpp"]
)

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=[nssgeometry, zsteps],
)
