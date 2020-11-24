"""
nuSpaceSim
----------

A Simulator for Space-based Nutrino Detections.
"""

import ast
import os
import platform
import re
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('nuSpaceSim/__init__.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='nuSpaceSim/nssgeometry'):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """
    Extend setuptools' build_ext command for building compiled code.
    """

    def run(self):
        """
        Build the compiled code modules with CMake.
        """
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(
                    e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        """
        Build 1 compiled code module with CMake.
        """

        extdir = os.path.abspath(
            os.path.dirname(
                self.get_ext_fullpath(
                    ext.name)))

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DPYBIND11_PYTHON_VERSION=' + '3.6']

        cfg = 'Debug' if self.debug else 'Release'

        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        # build_args += ['--', '-j4']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] +
                              cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] +
                              build_args, cwd=self.build_temp)


setup(
    version=version,
    ext_modules=[CMakeExtension('nssgeometry')],
    cmdclass=dict(build_ext=CMakeBuild),
)
