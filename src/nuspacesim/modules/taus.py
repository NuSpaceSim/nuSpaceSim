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

"""tau propagation module"""

import importlib_resources
import h5py
import numpy as np
from scipy import interpolate
from nuspacesim.core import NssConfig


def extract_nutau_data(filename: str, log_nu_energy: float) -> tuple:
    r"""Extract RenoNuTau tables into Tau params.

    Parameters
    ----------
    filename: str
        Name of RenoNuTau tables file.
    log_nu_energy: float
        log of neutrino energy.

    Returns
    -------
    tuple
        Tau parameter data object.
    """

    f = h5py.File(filename, "r")
    pegrp = f["pexitdata"]
    pegrppedset = pegrp["logPexit"]
    pegrpbdset = pegrp["BetaRad"]
    pegrplnedset = pegrp["logNuEnergy"]

    pegrppearr = np.array(pegrppedset)
    pegrpbarr = np.array(pegrpbdset)
    pegrplnearr = np.array(pegrplnedset)

    # lognuebin = float(closestNumber(np.rint(lognuenergy*100),25))/100.
    # If we want to do closest bin rather than histogram bins
    q = int((log_nu_energy * 100) / 25.0)
    lognuebin = float((q * 25) / 100.0)

    testring = "TauEdist_grp_e{:02.0f}_{:02.0f}".format(
        np.floor(lognuebin), (lognuebin - np.floor(lognuebin)) * 100
    )

    tegrp = f[testring]
    tegrpcdfdset = tegrp["TauEDistCDF"]
    tegrpbdset = tegrp["BetaRad"]
    tegrptauedset = tegrp["TauEFrac"]

    tegrpcdfarr = np.array(tegrpcdfdset)
    tegrpbarr = np.array(tegrpbdset)

    teBetaLowBnds = (tegrpbarr[1:] + tegrpbarr[:-1]) / 2
    teBetaUppBnds = np.append(teBetaLowBnds, tegrpbarr[-1])
    teBetaLowBnds = np.insert(teBetaLowBnds, 0, 0.0, axis=0)

    tegrptauearr = np.array(tegrptauedset)
    f.close()

    return (
        pegrppearr,
        pegrpbarr,
        pegrplnearr,
        tegrpcdfarr,
        tegrpbarr,
        teBetaLowBnds,
        teBetaUppBnds,
        tegrptauearr,
    )


class Taus(object):
    """
    Describe Tau Module HERE!
    """

    def __init__(self, config: NssConfig):
        """
        Intialize the Taus object.
        """
        self.config = config

        ref = (
            importlib_resources.files("nuspacesim.data.RenoNu2TauTables")
            / "nu2taudata.hdf5"
        )

        with importlib_resources.as_file(ref) as path:
            (
                self.pearr,
                self.pebarr,
                self.pelnearr,
                self.tecdfarr,
                self.tebarr,
                self.betaLowBnds,
                self.betaUppBnds,
                self.tetauefracarr,
            ) = extract_nutau_data(path, config.simulation.log_nu_tau_energy)

        self.peb_rbf = np.tile(self.pebarr, self.pelnearr.shape)
        self.pelne_rbf = np.repeat(self.pelnearr, self.pebarr.shape, 0)

        # Interpolating function to be used to find P_exit
        self.pexitfunc = interpolate.Rbf(
            self.peb_rbf, self.pelne_rbf, self.pearr, function="cubic", smooth=0
        )

        # Array of tecdf interpolation functions
        self.tauEFracInterps = [
            interpolate.interp1d(self.tecdfarr[:, betaind], self.tetauefracarr)
            for betaind in range(self.tecdfarr.shape[1])
        ]

        self.nuTauEnergy = self.config.simulation.nu_tau_energy

    def tau_exit_prob(self, betaArr):
        """
        Tau Exit Probability
        """
        brad = np.radians(betaArr)

        logtauexitprob = self.pexitfunc(
            brad, np.full(brad.shape, self.config.simulation.log_nu_tau_energy)
        )

        tauexitprob = 10 ** logtauexitprob

        return tauexitprob

    def tau_energy(self, betaArr, u=None):
        """
        Tau energies interpolated from teCDF for given beta index.
        """
        u = np.random.rand(betaArr.shape[0]) if u is None else u

        # fast interpolation selection with masking
        betaIdxs = np.searchsorted(np.degrees(self.betaUppBnds), betaArr)

        tauEF = np.empty_like(betaArr)

        for i in range(self.tecdfarr.shape[1]):
            mask = betaIdxs == i
            tauEF[mask] = self.tauEFracInterps[i](u[mask])

        return tauEF * self.nuTauEnergy

    def __call__(self, betaArr, store=None):
        """
        Perform main operation for Taus module.

        Returns:

        """

        tauExitProb = self.tau_exit_prob(betaArr)
        tauEnergy = self.tau_energy(betaArr)

        # in units of 100 PeV
        showerEnergy = self.config.simulation.e_shower_frac * tauEnergy / 1.0e8

        tauLorentz = tauEnergy / self.config.constants.massTau

        tauBeta = np.sqrt(1.0 - np.reciprocal(tauLorentz ** 2))

        if store is not None:
            store(
                ["tauBeta", "tauLorentz", "showerEnergy", "tauExitProb"],
                [tauBeta, tauLorentz, showerEnergy, tauExitProb],
            )

        return tauBeta, tauLorentz, showerEnergy, tauExitProb
