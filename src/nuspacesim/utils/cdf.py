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

from scipy.interpolate import interp1d
import numpy as np

from ..utils.grid import NssGrid
# from nuspacesim.utils.interp import grid_interpolator


# def cdf_sample_factory(grid: NssGrid, cdf_ax: int, interpolation_class=None, **kwargs):
#     """CDF sampling function from given grid object."""

#     if interpolation_class is None:
#         interpolation_class = grid_interpolator

#     interpolator = interpolation_class(grid, **kwargs)

#     def sample(*xi, u=None):

#         if len(xi) != grid.ndim - 1:
#             raise ValueError(f"Must provide {grid.ndim - 1} axes")

#         u = np.random.uniform(size=xi[cdf_ax].shape[0]) if u is None else u

#         for a in xi:
#             print(xi, a)
#             if u.shape != a.shape:
#                 raise ValueError(f"All axes must have same shape {u.shape}, {a.shape}")

#         points = list([*xi])
#         points.insert(cdf_ax, u)
#         points = np.column_stack(points)
#         print(points)

#         return interpolator(points)

#     return sample

def cdf_sampler(grid):

    f = interp1d(grid.axes[1], grid.data, axis=1)

    def sample(xs, u=None):

        u = np.random.uniform(size=xs) if u is None else u

        samples = np.empty_like(xs)

        for i, x in enumerate(xs):
            cdf = f(x)
            samples[i] = interp1d(cdf, grid.axes[0])(u[i])

        return samples

    return sample




def legacy_cdf_sample(grid) -> Callable:
    from scipy.interpolate import interp1d

    teBetaLowBnds = (grid.axes[1][1:] + grid.axes[1][:-1]) / 2
    teBetaUppBnds = np.append(teBetaLowBnds, grid.axes[1][-1])
    teBetaLowBnds = np.insert(teBetaLowBnds, 0, 0.0, axis=0)

    # Array of tecdf interpolation functions
    interps = [
        interp1d(grid.data[:, i], grid.axes[0]) for i in range(grid.axes[1].shape[0])
    ]

    # print("upper:", teBetaLowBnds)
    # print("lower:", teBetaUppBnds)

    def interpolate(xi, u=None):
        u = np.random.uniform(size=xi.shape[0]) if u is None else u

        # fast interpolation selection with masking
        betaIdxs = np.searchsorted(teBetaUppBnds, xi)
        # print("index:", betaIdxs)
        tauEF = np.empty_like(xi)

        for i in range(grid.axes[1].shape[0]):
            mask = betaIdxs == i
            tauEF[mask] = interps[i](u[mask])

        return tauEF

    return interpolate


if __name__ == "__main__":
    g = NssGrid.read("src/nuspacesim/data/RenoNu2TauTables/nu2tau_cdf.hdf5")
    g.index_name("log_nu_e")
    g.index_where_eq("log_nu_e", 7.5)
    sample = cdf_sample_factory(g[g.index_where_eq("log_nu_e", 7.5), :, :], 0)
    n = int(1e8)
    print(sample(np.random.uniform(1, 25, n)))
