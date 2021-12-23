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

"""CDF Sampling Utilities.

.. autosummary::
   :toctree:
   :recursive:

   grid_cdf_sampler
   invert_cdf_grid
   grid_inverse_sampler
   nearest_cdf_sampler
   lerp_cdf_sampler

"""

from typing import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d, interpn

from nuspacesim.utils.grid import NssGrid
from nuspacesim.utils.interp import grid_slice_interp, vec_1d_interp

__all__ = [
    "grid_inverse_sampler",
    "nearest_cdf_sampler",
    "lerp_cdf_sampler",
    "grid_cdf_sampler",
]


def invert_cdf_grid(grid: NssGrid) -> NssGrid:
    r"""Invert a Taus CDF grid for easier z(E_tau / E_nu) sampling.

    Use the CDF values of the given NssGrid to interpolate every measured z point,
    Then invert the grid such that (beta_rad, z) bins giving CDF values become
    (beta_rad, cdf) bins giving z values.

    Parameters
    ----------
    grid: NssGrid
        A 2D CDF grid with (beta_rad, z) bins and CDF values.

    Returns
    -------
    NssGrid
        The inverted grid.

    Examples
    --------

    >>> grid = NssGrid.read(file, format="hdf5")
    >>> igrid = invert_cdf_grid(grid)

    """

    if grid.ndim != 2:
        raise ValueError("Dimensions of grid are invalid for inversion")

    cdf_vals = np.ravel(grid.data)
    cdf_vals[cdf_vals > 1.0] = 1.0
    cdf_vals[cdf_vals < 0.0] = 0.0
    cdf_vals = np.unique(np.sort(cdf_vals))
    new_shape = list(grid.data.shape)
    new_shape[-1] = cdf_vals.size
    new_data = np.empty(new_shape, dtype=grid.data.dtype)

    for i, _ in enumerate(grid["beta_rad"]):

        s = np.s_[i, :]

        local_cdf = grid[s].data
        mask = local_cdf < (1.0 - 8 * np.finfo(grid.dtype).eps)
        first_one = np.argmin(mask)
        mask[first_one] = True

        cdf = np.asarray(local_cdf[mask])
        etf = np.asarray(grid["e_tau_frac"][mask])
        cdf[-1] = 1.0

        f = interp1d(cdf, etf)
        new_data[s] = f(cdf_vals)
        # , bounds_error=False, fill_value=(0.0, 1.0)

    new_axes = [grid["beta_rad"], cdf_vals]
    new_names = ["beta_rad", "cdf"]

    return NssGrid(new_data, new_axes, new_names)


def grid_inverse_sampler(grid: NssGrid, log_e_nu: float, sliced=False) -> Callable:
    r"""Sample Tau Energies by inverting the tau cdf grid.

    Given a log_e_nu value, slice an interpolated 2D tau cdf grid for that log_e_nu.
    Then invert the grid and sample z values directly from beta angles and the grid.
    Multiply by 10^log_e_nu to determine the tau energy.

    Parameters
    ----------
    grid: NssGrid
        The full, 3D tau_cdf grid object.
    log_e_nu: float
        The user defined log of the neutrino energy.

    Returns
    -------
    sample : Callable
        A function that will take an array of beta angles in radians and return an
        array of sampled tau energies.

    Examples
    --------

    >>> grid = NssGrid.read(file, format="hdf5")
    >>> sampler = grid_inverse_sampler(grid, config.simulation.log_nu_tau_energy)
    >>> samples = sampler(betas)

    """

    sliced_grid = grid if sliced else grid_slice_interp(grid, log_e_nu, "log_e_nu")

    inv_grid = invert_cdf_grid(sliced_grid)

    interpolate = RegularGridInterpolator(inv_grid.axes, inv_grid.data)

    def sample(x, u=None):
        r"""Sampling function for grid_inverse_sampler.

        Interpolate z (E_tau / E_nu) values from the inverse tau_cdf grid.

        Parameters
        ----------
        x: ArrayLike
            The beta angles (in radians) to sample z values from.
        u: ArrayLike, Optional
            Random numbers for CDF interpolation. If 'None' values will be generated.

        Returns
        -------
        result : ArrayLike
            Array of sampled z values parameterized by the interpolated CDFs.

        Examples
        --------

        >>> sample = sampler(np.random.randn(np.radians(1), np.radians(42)))

        """
        it = np.nditer(
            [x, u, None],
            flags=["external_loop", "buffered"],
            op_flags=[
                ["readonly"],
                ["readwrite", "virtual"] if u is None else ["readonly"],
                ["writeonly", "allocate", "no_broadcast"],
            ],
        )
        with it:
            for xi, ui, zi in it:
                ui = np.random.uniform(0.0, 1.0, size=xi.size) if u is None else ui
                pts = np.column_stack([xi, ui])
                zi[...] = interpolate(pts)
            return it.operands[2]

    return sample


def nearest_cdf_sampler(grid: NssGrid, log_e_nu: float) -> Callable:
    r"""Sample Tau Energies by slicing and interpolating the tau_cdf grid.

    Given a log_e_nu value, slice an interpolated 2D tau cdf grid for that log_e_nu.
    Then sample z values by interpolating cdfs from the grid for given beta angles.
    Multiply by 10^log_e_nu to determine the tau energy.

    Parameters
    ----------
    grid: NssGrid
        The full, 3D tau_cdf grid object.
    log_e_nu: float
        The user defined base 10 log of the neutrino energy.

    Returns
    -------
    sample : Callable
        A function that will take an array of beta angles in radians and return an
        array of sampled tau energies.
    """

    enu_idx = grid.axis_names.index("log_e_nu")

    sliced_grid = grid_slice_interp(grid, log_e_nu, enu_idx)

    bx = sliced_grid.axis_names.index("beta_rad")
    ex = sliced_grid.axis_names.index("e_tau_frac")

    interp_cdfs = [
        interp1d(
            sliced_grid.data[np.s_[i, :] if bx == 0 else np.s_[:, i]],
            sliced_grid.axes[ex],
        )
        for i in range(sliced_grid.data.shape[bx])
    ]

    teBetaLowBnds = (sliced_grid.axes[bx][1:] + sliced_grid.axes[bx][:-1]) / 2
    teBetaUppBnds = np.append(teBetaLowBnds, sliced_grid.axes[bx][-1])

    def sample(x, u=None):
        r"""Sampling function for nearest_cdf_sampler.

        Interpolate cdf values and inverse transfom sample z (E_tau / E_nu) values from
        the tau_cdf grid.

        Parameters
        ----------
        x: ArrayLike
            The beta angles (in radians) to sample z values from.
        u: ArrayLike, Optional
            Random numbers for CDF interpolation. If 'None' values will be generated.

        Returns
        -------
        result : ArrayLike
            Array of sampled z values parameterized by the interpolated CDFs.

        Examples
        --------

        >>> sample = sampler(np.random.randn(np.radians(1), np.radians(42)))

        """
        it = np.nditer(
            [x, u, None],
            flags=["external_loop", "buffered"],
            op_flags=[
                ["readonly"],
                ["readwrite", "virtual"] if u is None else ["readonly"],
                ["writeonly", "allocate", "no_broadcast"],
            ],
        )

        with it:
            for xi, ui, zi in it:
                betaIdxs = np.searchsorted(teBetaUppBnds, xi)
                ui = np.random.uniform(0.0, 1.0, size=xi.size) if u is None else ui
                for i, interp in enumerate(interp_cdfs):
                    mask = betaIdxs == i
                    zi[...][mask] = interp(ui[mask])
            return it.operands[2]

    return sample


def lerp_cdf_sampler(grid: NssGrid, log_e_nu: float) -> Callable:
    r"""Sample Tau Energies by interpolating the tau_cdf grid.

    Given a log_e_nu value, slice an interpolated 2D tau cdf grid for that log_e_nu.
    Then sample z values by interpolating cdfs from the grid for given beta angles.
    Multiply by 10^log_e_nu to determine the tau energy.

    Parameters
    ----------
    grid: NssGrid
        The full, 3D tau_cdf grid object.
    log_e_nu: float
        The user defined base 10 log of the neutrino energy.

    Returns
    -------
    sample : Callable
        A function that will take an array of beta angles in radians and return an
        array of sampled tau energies.
    """

    enu_idx = grid.axis_names.index("log_e_nu")

    sliced = grid_slice_interp(grid, log_e_nu, enu_idx)

    bx = sliced.axis_names.index("beta_rad")
    ex = sliced.axis_names.index("e_tau_frac")

    interp_cdfs = interp1d(sliced.axes[bx], sliced.data, axis=bx)

    def sample(beta, u=None):
        r"""Sampling function for nearest_cdf_sampler.

        Interpolate cdf values and inverse transfom sample z (E_tau / E_nu) values from
        the tau_cdf grid.

        Parameters
        ----------
        x: ArrayLike
            The beta angles (in radians) to sample z values from.
        u: ArrayLike, Optional
            Random numbers for CDF interpolation. If 'None' values will be generated.

        Returns
        -------
        result : ArrayLike
            Array of sampled z values parameterized by the interpolated CDFs.

        Examples
        --------

        >>> sample = sampler(np.random.randn(np.radians(1), np.radians(42)))

        """
        it = np.nditer(
            [beta, u, None],
            flags=["external_loop", "buffered"],
            op_flags=[
                ["readonly"],
                ["readwrite", "virtual"] if u is None else ["readonly"],
                ["writeonly", "allocate", "no_broadcast"],
            ],
        )

        with it:
            for bi, ui, zi in it:
                cdfs = interp_cdfs(bi)
                ui = np.random.uniform(0.0, 1.0, size=bi.size) if u is None else ui
                zi[...] = vec_1d_interp(cdfs, sliced.axes[ex], ui)
            return it.operands[2]

    return sample


def grid_cdf_sampler(grid: NssGrid) -> Callable:
    r"""Sample Tau Energies by interpolating the tau_cdf grid.

    Parameters
    ----------
    grid: NssGrid
        The full, 3D tau_cdf grid object.

    Returns
    -------
    sample : Callable
        A function that will take an array of beta angles in radians and return an
        array of sampled tau energies.
    """

    def sample(log_e_nu, beta, u=None):
        r"""Sampling function for nearest_cdf_sampler.

        Interpolate cdf values and inverse transfom sample z (E_tau / E_nu) values from
        the tau_cdf grid.

        Parameters
        ----------
        log_e_nu: ArrayLike
            The log energies to sample z values from.
        beta: ArrayLike
            The beta angles (in radians) to sample z values from.
        u: ArrayLike, Optional
            Random numbers for CDF interpolation. If 'None', values will be generated.

        Returns
        -------
        result : ArrayLike
            Array of sampled z values parameterized by the interpolated CDFs.

        """
        it = np.nditer(
            [log_e_nu, beta, u, None],
            flags=["external_loop", "buffered"],
            op_flags=[
                ["readonly"],
                ["readonly"],
                ["readwrite", "virtual"] if u is None else ["readonly"],
                ["writeonly", "allocate", "no_broadcast"],
            ],
        )

        with it:
            for li, bi, ui, zi in it:
                cdfs = interpn(
                    (grid["log_e_nu"], grid["beta_rad"]), grid.data, (li, bi)
                )
                ui = np.random.uniform(0.0, 1.0, size=bi.size) if u is None else ui
                zi[...] = vec_1d_interp(cdfs, grid["e_tau_frac"], ui)
            return it.operands[3]

    return sample
