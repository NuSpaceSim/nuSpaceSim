# The Clear BSD License
#
# Copyright (c) 2023 Alexander Reustle and the NuSpaceSim Team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#      * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from dataclasses import dataclass

import numpy as np

__all__ = ["MonoCloud", "NoCloud"]


@dataclass
class NoCloud:
    pass


@dataclass
class MonoCloud:
    altitude: float = -np.inf
    """Altitude of monoheight cloud."""


@dataclass
class PressureMapCloud:
    month: str = "01"
    """Cloud Map Month in 2-digit MM format."""

    version: str = "0"
    """Cloud Map File Version."""


# @dataclass
# class BetaPressureMap:
#     map_file: str | None = None


# @dataclass
# class BetaPressureCloudMap:
#     map_file: str | None = None
#
#     def __init__(self, cloud_map_filename="Default"):
#         self.map_file = cloud_map_filename
#         if self.map_file == "Default" or "" or None:
#             with as_file(
#                 files("nuspacesim.data.cloud_maps")
#                 / "nss_rmap_CloudTopPressure_01_2011_2020_9E6D7805.fits"
#             ) as file:
#                 pass
#                 # self.pexit_grid = NssGrid.read(file, path="/", format="hdf5")
#
#         self.map = None  # hdu[0].data
#
#     def __call__(self, lat, long) -> float:
#         return self.map[lat, long]
