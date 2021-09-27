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

from nuspacesim.utils.grid import NssGrid
from nuspacesim.utils.interp import grid_interpolator
import numpy as np


def cdf_sample_factory(grid: NssGrid, cdf_ax: int, interpolation_class=None, **kwargs):
    """CDF sampling function from given grid object."""

    if interpolation_class is None:
        interpolation_class = grid_interpolator(grid, **kwargs)

    interpolator = interpolation_class(grid, **kwargs)

    def sample(*axes, u=None):

        if len(axes) != grid.ndim - 1:
            raise ValueError(f"Must provide {grid.ndim - 1} axes")

        u = np.random.uniform(size=axes[0].shape[0]) if u is None else u

        for a in axes:
            if u.shape != a.shape:
                raise ValueError(f"axes must have same shape")

        full_axes = list([*axes])
        full_axes.insert(cdf_ax, u)

        v = np.column_stack(full_axes)
        return interpolator(v)

    return sample


if __name__ == "__main__":
    g = NssGrid.read("src/nuspacesim/data/RenoNu2TauTables/nu2tau_cdf.hdf5")
    g.index_name("log_nu_e")
    g.index_where_eq("log_nu_e", 7.5)
    sample = cdf_sample_factory(g[g.index_where_eq("log_nu_e", 7.5), :, :], 0)
    n = int(1e8)
    print(sample(np.random.uniform(1, 25, n)))
