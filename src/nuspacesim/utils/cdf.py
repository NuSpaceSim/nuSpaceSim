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

"""CDF Sampling Utilities."""

from typing import Callable

from scipy.interpolate import interp1d, RegularGridInterpolator
import numpy as np

from nuspacesim.utils.grid import NssGrid
from nuspacesim.utils.interp import grid_slice_interp, vec_1d_interp

__all__ = ["grid_inverse_sampler", "nearest_cdf_sampler", "lerp_cdf_sampler"]


def invert_cdf_grid(grid: NssGrid, cdf_axis: int = 1) -> NssGrid:
    r"""Invert a Taus CDF grid for easier z(E_tau / E_nu) sampling.

    Use the CDF values of the given NssGrid to interpolate every measured z point,
    Then invert the grid such that (beta_rad, z) bins giving CDF values become
    (beta_rad, cdf) bins giving z values.

    Parameters
    ----------
    grid: NssGrid
        A 2D CDF grid with (beta_rad, z) bins and CDF values.
    cdf_axis: int
        The target axis to store the CDF bins in the result grid.

    Returns
    -------
    NssGrid
        The inverted grid.
    """

    assert grid.ndim == 2
    assert cdf_axis <= 2

    brad_axis = 0 if cdf_axis == 1 else 1

    cdf_vals = np.unique(np.sort(np.ravel(grid.data)))
    new_shape = list(grid.data.shape)
    new_shape[cdf_axis] = cdf_vals.size
    new_data = np.empty(new_shape, dtype=grid.data.dtype)

    for i in range(grid.shape[brad_axis]):

        s = np.s_[i, :] if cdf_axis == 1 else np.s_[:, i]

        new_data[s] = interp1d(
            grid.data[s], grid.axes[cdf_axis], bounds_error=False, fill_value=(0.0, 1.0)
        )(cdf_vals)

    new_axes = [grid.axes[brad_axis], cdf_vals]
    new_names = [grid.axis_names[brad_axis], "cdf"]

    return NssGrid(new_data, new_axes, new_names)


def grid_inverse_sampler(grid: NssGrid, log_e_nu: float) -> Callable:
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
    """

    enu_idx = grid.axis_names.index("log_e_nu")

    sliced_grid = grid_slice_interp(grid, log_e_nu, enu_idx)

    z_idx = sliced_grid.axis_names.index("e_tau_frac")

    inv_grid = invert_cdf_grid(sliced_grid, z_idx)

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


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    g = NssGrid.read("data/nupyprop_tables/nu2tau_cdf.hdf5")
    print(g.axis_names)
    gsampler = grid_inverse_sampler(g, 7.890123456)
    # sampler = nearest_cdf_sampler(g, 7.890123456)
    # lsampler = lerp_cdf_sampler(g, 7.890123456)
    N = 1e8

    x = np.random.uniform(np.radians(1.0), np.radians(42.0), size=int(N))
    u = np.random.uniform(0.0, 1.0, size=int(N))

    # z = sampler(x)
    # print(z)
    # z = sampler(x, u)
    # print(z)
    z = gsampler(x, u)
    print(z)
    # z = lsampler(x, u)
    # print(z)
