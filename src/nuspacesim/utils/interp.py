# The Clear BSD License
#
# Copyright (c) 2021 Alexander Reustle and the NuSpaceSim Team
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

""" Generalized interpolation functions. """

import numpy as np

from typing import Callable
from ..utils.misc import cartesian_product

__all__ = [
    "grid_interpolator",
    "grid_RegularGridInterpolator",
    # "grid_RBFInterpolator",
    # "legacy_RBFInterpolator",
]


def grid_interpolator(grid, interpolator=None, **kwargs) -> Callable:
    """Factory function to return and interpolation function from a grid."""

    if interpolator is None:
        interpolator = grid_RegularGridInterpolator

    interpf = interpolator(grid, **kwargs)

    def interpolate(xi, *args, use_grid=False, **kwargs):
        xi = cartesian_product(*xi) if use_grid else xi
        return interpf(xi, *args, **kwargs)

    return interpolate


def grid_RegularGridInterpolator(grid, **kwargs):
    from scipy.interpolate import RegularGridInterpolator

    if "bounds_error" not in kwargs:
        kwargs["bounds_error"] = False
    if "fill_value" not in kwargs:
        kwargs["fill_value"] = None

    return RegularGridInterpolator(grid.axes, grid.data, **kwargs)


# def grid_RBFInterpolator(grid, **kwargs) -> Callable:
#     from scipy.interpolate import RBFInterpolator

#     return RBFInterpolator(cartesian_product(*grid.axes), grid.data, **kwargs)


def legacy_RBFInterpolator(grid, **kwargs) -> Callable:
    from scipy.interpolate import Rbf

    print(grid)

    peb_rbf = np.tile(grid.axes[1], grid.axes[0].shape)
    pelne_rbf = np.repeat(grid.axes[0], grid.axes[1].shape, 0)
    return Rbf(peb_rbf, pelne_rbf, grid.data, **kwargs)

