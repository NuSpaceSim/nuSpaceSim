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

"""Wrapper class for interacting with RenoNuTau table data and nupyprop tables."""

import numpy as np

from astropy.table import Table as AstropyTable
from ..utils.grid import NssGrid
from ..utils.interp import grid_interpolator

# from typing import Iterable, Union


def nu2tau_pexit_from_ascii():
    pexit_file = AstropyTable.read(
        "src/nuspacesim/data/RenoNu2TauTables/multi-efix.26", format="ascii"
    )

    beta_rad = np.unique(np.radians(pexit_file.columns[0].data))
    log_nu_e = np.unique(np.log10(pexit_file.columns[1].data))
    pexit_table = np.reshape(
        np.log10(pexit_file.columns[-1].data), (len(log_nu_e), len(beta_rad))
    )

    del pexit_file

    return NssGrid(pexit_table, [log_nu_e, beta_rad], ["log_nu_e", "beta_rad"])


def make_nu2tau_pexit_from_ascii():
    r"""Make a nu2taudata output file with data tables."""

    pexit_grid = nu2tau_pexit_from_ascii()

    fname = "src/nuspacesim/data/RenoNu2TauTables/nu2tau_pexit.fits"
    hname = "src/nuspacesim/data/RenoNu2TauTables/nu2tau_pexit.hdf5"
    pexit_grid.write(fname, overwrite=True, format="fits")
    pexit_grid.write(hname, overwrite=True, format="hdf5")
    grd2 = NssGrid.read(fname, format="fits")
    grd3 = NssGrid.read(hname, format="hdf5")
    print(pexit_grid == grd2)
    print(pexit_grid == grd3)


def nu2tau_tauEDistCDF_from_ascii():

    bdeg = np.array([1.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0])
    brad = np.radians(bdeg)

    grids = []
    for lognuenergy in np.arange(7.0, 11.0, 0.25):
        textf = "src/nuspacesim/data/RenoNu2TauTables/nu2tau-angleC-e{:02.0f}-{:02.0f}smx.dat".format(
            np.floor(lognuenergy), (lognuenergy - np.floor(lognuenergy)) * 100
        )
        tau_e_file = np.loadtxt(textf, dtype=np.float64)
        tauEfrac = tau_e_file[:, 0]
        tauCDF = tau_e_file[:, 1:]
        grids.append(
            (
                lognuenergy,
                NssGrid(tauCDF, [tauEfrac, brad], axis_names=["tauEfrac", "beta_rad"]),
            )
        )
    return grids

    # full_grids = []
    # for g in grids:
    #     interp = grid_interpolator(g)
    #     full_grids.append(
    #         interp(grids[-1].axes, use_grid=True).reshape(grids[-1].shape)
    #     )

    # return NssGrid(
    #     np.stack(full_grids),
    #     [np.arange(7.0, 11.0, 0.25), *grids[-1].axes],
    #     ["log_nu_e", *grids[-1].axis_names],
    # )


def make_nu2tau_cdf_from_ascii():

    tau_cdf_grids = nu2tau_tauEDistCDF_from_ascii()

    # fname = "src/nuspacesim/data/RenoNu2TauTables/nu2tau_cdf.fits"
    hname = "src/nuspacesim/data/RenoNu2TauTables/nu2tau_cdf.hdf5"
    for log_nu_e, g in tau_cdf_grids:
        # g.write(fname, overwrite=True, format="fits")
        g.write(hname, path=f"/log_nu_e_{log_nu_e}", format="hdf5")
        # grd2 = NssGrid.read(fname, format="fits")
        grd3 = NssGrid.read(hname, format="hdf5", path=f"/log_nu_e_{log_nu_e}")
        # print(tau_cdf_grid == grd2)
        print(g == grd3)


if __name__ == "__main__":
    make_nu2tau_pexit_from_ascii()
    make_nu2tau_cdf_from_ascii()
