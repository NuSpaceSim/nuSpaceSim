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

""" Special interpolation functions.

.. autosummary::
   :toctree:
   :recursive:

   grid_interpolator
   grid_slice_interp
   grid_RegularGridInterpolator
   vec_1d_interp

"""

from typing import Any, Callable

import numpy as np
from scipy.interpolate import interp1d

from nuspacesim.utils.grid import NssGrid

__all__ = [
    "grid_interpolator",
    "grid_slice_interp",
    "grid_RegularGridInterpolator",
    "vec_1d_interp",
]


def grid_slice_interp(grid: NssGrid, value: float, axis: Any) -> NssGrid:
    r"""Continuous grid slice using interpolation.

    Slice the N-Dimensional NssGrid along an axis value that may not exist in the grid.
    Linearly interpolate other values as necessary, returning an N-1 D NssGrid.

    Parameters
    ----------
    grid: NssGrid
        The grid to be sliced
    value: float
        The value at which to slice the grid.
    axis: int, string
        The axis along which to slice.

    Returns
    -------
    NssGrid
        The N-1 resulting Dimensional grid.

    """

    axis = grid.axis_names.index(axis) if isinstance(axis, str) else axis

    new_data = interp1d(grid.axes[axis], grid.data, axis=axis)(value)
    new_axes = [ax for i, ax in enumerate(grid.axes) if i != axis]
    new_names = [n for i, n in enumerate(grid.axis_names) if i != axis]
    return NssGrid(new_data, new_axes, new_names)


def grid_interpolator(grid, interpolator=None, **kwargs) -> Callable:
    """Factory function to return and interpolation function from a grid."""

    if interpolator is None:
        interpolator = grid_RegularGridInterpolator

    return interpolator(grid, **kwargs)


def grid_RegularGridInterpolator(grid, **kwargs):
    from scipy.interpolate import RegularGridInterpolator

    if "bounds_error" not in kwargs:
        kwargs["bounds_error"] = False
    if "fill_value" not in kwargs:
        kwargs["fill_value"] = None

    return RegularGridInterpolator(grid.axes, grid.data, **kwargs)


def left_shift(arr):
    result = np.empty_like(arr)
    result[:, -1:] = True
    result[:, :-1] = arr[:, 1:]
    return result


def right_shift(arr):
    result = np.empty_like(arr)
    result[:, :1] = True
    result[:, 1:] = arr[:, :-1]
    return result


def vec_1d_interp(xs, ys, x):
    # mask and index for upper bound
    hi_msk = xs >= x[:, None]
    shf_hi = left_shift(hi_msk)
    hi_m = np.logical_xor(hi_msk, shf_hi)
    hi = np.where(hi_m)[1]

    # mask and index for lower bound
    lo_msk = xs < x[:, None]
    shf_lo = right_shift(lo_msk)
    lo_m = np.logical_xor(lo_msk, shf_lo)
    lo = np.where(lo_m)[1]

    y0 = ys[lo]
    x0 = xs[lo_m]
    y1 = ys[hi]
    x1 = xs[hi_m]

    y = y0 + (x - x0) * ((y1 - y0) / (x1 - x0))

    return y
