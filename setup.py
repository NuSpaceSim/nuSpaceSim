"""
nuSpaceSim
----------

A Simulator for Space-based Nutrino Detections.
"""
import ast
import os
import platform
import re
import setuptools
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension as pyExtension
from pybind11.setup_helpers import build_ext as py_build_ext
import subprocess
import sys

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('nuSpaceSim/__init__.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))


nssgeometry = pyExtension("nssgeometry",
                          ["nuSpaceSim/nssgeometry/src/nssgeometry.cpp"])

setup(
    version=version,
    cmdclass={"build_ext": py_build_ext},
    ext_modules=[nssgeometry, ],
)
