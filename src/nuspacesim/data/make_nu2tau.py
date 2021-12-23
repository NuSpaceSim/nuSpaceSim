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

import h5py
import numpy as np
from astropy.table import Table as AstropyTable

from nuspacesim.utils.grid import NssGrid

np.set_printoptions(linewidth=np.inf)


def nu2tau_pexit_from_ascii():
    pexit_file = AstropyTable.read(
        "src/nuspacesim/data/RenoNu2TauTables/multi-efix-v5-sm35-5r-2020.26",
        format="ascii",
    )

    beta_rad = np.unique(np.radians(pexit_file.columns[0].data))
    log_e_nu = np.unique(np.log10(pexit_file.columns[1].data))
    pexit_table = np.reshape(
        np.log10(pexit_file.columns[-1].data), (len(log_e_nu), len(beta_rad))
    )

    # del pexit_file

    return NssGrid(pexit_table, [log_e_nu, beta_rad], ["log_e_nu", "beta_rad"])


def nu2tau_pexit_from_nupyprop_v1(filename: str):

    with h5py.File(filename, "r") as f:

        pexit_ds = f["Exit_Probability"]

        logEnu = np.asarray(list(pexit_ds.keys()), dtype=np.float64)
        logEnu = np.sort(logEnu)

        bdeg = np.asarray([d for d, _, _ in pexit_ds[str(logEnu[0])]], dtype=np.float64)
        bdeg = np.sort(bdeg)

        pexit_no_regen = np.empty((logEnu.size, bdeg.size), dtype=np.float64)
        pexit_regen = np.empty((logEnu.size, bdeg.size), dtype=np.float64)

        for li, l in enumerate(logEnu):
            for d, n, r in pexit_ds[str(l)]:
                d = float(d)
                if d not in bdeg:
                    raise ValueError(f"Got {d} not in {bdeg}")
                bi = np.where(bdeg == d)[0][0]
                pexit_no_regen[li, bi] = n
                pexit_regen[li, bi] = r

    beta_rad = np.radians(bdeg)
    return (
        NssGrid(pexit_no_regen, [logEnu, beta_rad], ["log_e_nu", "beta_rad"]),
        NssGrid(pexit_regen, [logEnu, beta_rad], ["log_e_nu", "beta_rad"]),
    )


def nu2tau_cdf_from_nupyprop_v1(filename: str):

    with h5py.File(filename, "r") as f:

        cdf_grp = f["CLep_out_cdf"]

        log_e_nu = np.asarray(list(cdf_grp.keys()), dtype=np.float64)
        log_e_nu = np.sort(log_e_nu)

        zs = np.asarray(cdf_grp[str(log_e_nu[0])]["z"])

        bdeg = np.arange(1.0, 43.0, 1.0, dtype=np.float64)

        tau_cdf_data = np.empty((log_e_nu.size, bdeg.size, zs.size), dtype=np.float64)

        for li, l in enumerate(log_e_nu):
            for bi, b in enumerate(bdeg):
                tau_cdf_data[li, bi, :] = cdf_grp[str(l)][str(b)]

    beta_rad = np.radians(bdeg)
    return NssGrid(
        tau_cdf_data, [log_e_nu, beta_rad, zs], ["log_e_nu", "beta_rad", "e_tau_frac"]
    )


def make_nu2tau_pexit_from_ascii():
    r"""Make a nu2taudata output file with data tables."""

    pexit_grid = nu2tau_pexit_from_ascii()

    hname = "src/nuspacesim/data/RenoNu2TauTables/reno_nu2tau_pexit.hdf5"
    pexit_grid.write(hname, path="/", overwrite=True, format="hdf5")
    grd3 = NssGrid.read(hname, path="/", format="hdf5")
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
                NssGrid(
                    tauCDF.T, [brad, tauEfrac], axis_names=["beta_rad", "e_tau_frac"]
                ),
            )
        )
    return grids


def make_nu2tau_cdf_from_ascii():

    tau_cdf_grids = nu2tau_tauEDistCDF_from_ascii()

    hname = "src/nuspacesim/data/RenoNu2TauTables/nu2tau_cdf.hdf5"
    for log_e_nu, g in tau_cdf_grids:
        g.write(hname, path=f"/log_nu_e_{log_e_nu}", format="hdf5")
        grd3 = NssGrid.read(hname, format="hdf5", path=f"/log_nu_e_{log_e_nu}")
        print(g == grd3)


def make_nu2tau_pexit_from_nupyprop_v1():
    r"""Make a nu2taudata output file with data tables."""

    pexit_no_regen, pexit_regen = nu2tau_pexit_from_nupyprop_v1(
        "output_nu_tau_4km_ct18nlo_allm_stochastic_1e8.h5"
    )

    hname = "nupyprop_tables/nu2tau_pexit.hdf5"
    pexit_no_regen.write(hname, path="/pexit_no_regen", overwrite=True, format="hdf5")
    pexit_regen.write(hname, path="/pexit_regen", overwrite=True, format="hdf5")
    grd2 = NssGrid.read(hname, path="/pexit_no_regen", format="hdf5")
    grd3 = NssGrid.read(hname, path="/pexit_regen", format="hdf5")
    print(pexit_no_regen == grd2)
    print(pexit_regen == grd3)

    frname = "nupyprop_tables/nu2tau_pexit_regen.fits"
    fnname = "nupyprop_tables/nu2tau_pexit_no_regen.fits"
    pexit_no_regen.write(fnname, overwrite=True, format="fits")
    pexit_regen.write(frname, overwrite=True, format="fits")
    grd2 = NssGrid.read(fnname, format="fits")
    grd3 = NssGrid.read(frname, format="fits")
    print(pexit_no_regen == grd2)
    print(pexit_regen == grd3)


def make_nu2tau_cdf_from_nupyprop_v1():
    r"""Make a nu2taudata output file with data tables."""

    tau_cdf_grid = nu2tau_cdf_from_nupyprop_v1(
        "output_nu_tau_4km_ct18nlo_allm_stochastic_1e8.h5"
    )

    hname = "nupyprop_tables/nu2tau_cdf.hdf5"
    tau_cdf_grid.write(hname, path="/", overwrite=True, format="hdf5")
    grd2 = NssGrid.read(hname, path="/", format="hdf5")
    print(tau_cdf_grid == grd2)

    fnname = "nupyprop_tables/nu2tau_cdf.fits"
    tau_cdf_grid.write(fnname, overwrite=True, format="fits")
    grd2 = NssGrid.read(fnname, format="fits")
    print(tau_cdf_grid == grd2)


if __name__ == "__main__":
    # make_nu2tau_pexit_from_ascii()
    # make_nu2tau_pexit_from_nupyprop_v1()
    # make_nu2tau_cdf_from_nupyprop_v1()
    make_nu2tau_cdf_from_ascii()
