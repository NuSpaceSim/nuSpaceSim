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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Union

try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files

import numpy as np
import numpy.typing as npt
from astropy.io import fits

from ...config import NssConfig
from ...types import cloud_types
from ..eas_optical import atmospheric_models as atm


class CloudTopHeight:
    def __init__(self, config: NssConfig):
        self.cloud_model = config.simulation.cloud_model

        if self.cloud_model is None:
            self.altitude = mono_altitude()

        elif isinstance(self.cloud_model, cloud_types.NoCloud):
            self.altitude = mono_altitude()

        elif isinstance(self.cloud_model, cloud_types.MonoCloud):
            self.altitude = mono_altitude(self.cloud_model.altitude)

        elif isinstance(self.cloud_model, cloud_types.PressureMapCloud):
            self.map = extract_fits_cloud_pressure_map_v0(self.cloud_model)
            self.altitude = altitude_from_pressure_map_v0(self.map)

        else:
            RuntimeError(f"Unrecognized Cloud Model Type: {type(self.cloud_model)}!")

    def __call__(self, *args, **kwargs) -> npt.single:
        return self.altitude(*args, **kwargs)


def mono_altitude(altitude=0.0):
    def f(*args, **kwargs) -> np.single:
        return altitude

    return f


def altitude_from_pressure_map_v0(map: npt.ArrayLike):
    latitudes = np.linspace(-90, 90, map.shape[0])
    longitudes = np.linspace(-180, 180, map.shape[1])

    def f(lat: float, long: float, *args, **kwargs) -> np.single:
        i = np.searchsorted(latitudes, lat)
        j = np.searchsorted(longitudes, lat)
        pressure: np.single = map[i, j]
        return atm.us_std_atm_altitude_from_pressure(pressure)

    return f


def extract_fits_cloud_pressure_map_v0(cloud_model: cloud_types.PressureMapCloud):
    month = cloud_model.month
    version = cloud_model.version
    with as_file(
        files("nuspacesim.data.cloud_maps")
        / f"nss_map_CloudTopPressure_{month:02d}.v{version}.fits"
    ) as file:
        hdul = fits.open(file)
        map = np.copy(hdul[0].data)
    return map


# def validate_fits_cloud_pressure_map() -> bool:
#     pass
