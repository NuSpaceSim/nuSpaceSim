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
import numpy as np
from nuspacesim.core import NssConfig
from nuspacesim.utils.grid import NssGrid
from nuspacesim.utils.cdf import cdf_sample_factory


class Taus(object):
    """
    Describe Tau Module HERE!
    """

    def __init__(self, config: NssConfig):
        """
        Intialize the Taus object.
        """
        self.config = config

        # grid of pexit table
        with importlib_resources.as_file(
            importlib_resources.files("nuspacesim.data.RenoNu2TauTables")
            / "nu2tau_pexit.hdf5"
        ) as file:
            self.pexit_grid = NssGrid.read(file, format="hdf5").slc(
                "log_nu_e", config.simulation.log_nu_tau_energy, 0
            )

        # grid of tau_cdf tables
        with importlib_resources.as_file(
            importlib_resources.files("nuspacesim.data.RenoNu2TauTables")
            / "nu2tau_cdf.hdf5"
        ) as file:
            self.tau_cdf_grid = NssGrid.read(file, format="hdf5").slc(
                "log_nu_e", config.simulation.log_nu_tau_energy, 0
            )

        self.tau_cdf_sample = cdf_sample_factory(self.tau_cdf_grid, 0)
        self.nu_tau_energy = self.config.simulation.nu_tau_energy

    def tau_exit_prob(self, betas):
        """
        Tau Exit Probability
        """
        logtauexitprob = self.pexit_grid.interpolate(betas)

        tauexitprob = 10 ** logtauexitprob

        return tauexitprob

    def tau_energy(self, betas, u=None):
        """
        Tau energies interpolated from tau_cdf_sampler for given beta index.
        """

        tauEF = self.tau_cdf_sample(betas, u)
        return tauEF * self.nu_tau_energy

    def __call__(self, betas, store=None):
        """
        Perform main operation for Taus module.

        Returns:

        """

        tauExitProb = self.tau_exit_prob(betas)
        tauEnergy = self.tau_energy(betas)

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
