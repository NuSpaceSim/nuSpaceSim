"""
nuSpaceSim
----------

A Simulator for Space-based Neutrino Detections.
"""
import ast
import re
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

_version_re = re.compile(r"__version__\s+=\s+(.*)")

with open("nuSpaceSim/__init__.py", "rb") as f:
    version = str(
        ast.literal_eval(_version_re.search(f.read().decode("utf-8")).group(1))
    )


nssgeometry = Pybind11Extension(
    "nuSpaceSim.nssgeometry", ["nuSpaceSim/nssgeometry/src/nssgeometry.cpp"]
)
zsteps = Pybind11Extension(
    "nuSpaceSim.EAScherGen.zsteps", ["nuSpaceSim/EAScherGen/src/zsteps.cpp"]
)

setup(
    version=version,
    cmdclass={"build_ext": build_ext},
    ext_modules=[nssgeometry, zsteps],
)
