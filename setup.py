"""
nuspacesim
----------

A Simulator for Space-based Neutrino Detections.
"""

import os
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class ZigBuilder(build_ext):
    def build_extension(self, ext):
        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)
        self.spawn(
            [
                sys.executable,
                "-m",
                "ziglang",
                "build-lib",
                "-O",
                "ReleaseFast",
                "-lc",
                *([]),
                f"-femit-bin={ext_path}",
                "-fallow-shlib-undefined",
                # "-dynamic",
                *[f"-I{d}" for d in self.include_dirs],
                *(),
                ext.sources[0],
            ]
        )


cphotang = Extension(
    "simulation.eas_optical.cphotang",
    sources=["src/nuspacesim/simulation/eas_optical/cphotang.zig"],
)

setup(cmdclass={"build_ext": ZigBuilder}, ext_modules=[cphotang])
