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

r"""Tau propagation module. A class for sampling tau attributes from beta angles."""

import numpy as np
from scipy.interpolate import interp1d

from ... import NssConfig
from ...utils import decorators
from ...utils.grid import NssGrid
from ...utils.cdf import grid_inverse_sampler
from ...utils.interp import grid_slice_interp
from .local_plots import taus_scatter, taus_histogram, taus_pexit

try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files


class Taus(object):
    r"""Tau attributes from beta angles via sampling CDF tables provided by nupyprop.

    Attributes
    ----------
    config: NssConfig
        Configuration object
    pexit_interp: Callable
        Interpolation function for tau exit probability.
    tau_cdf_sample: Callable
        Interpolative sampler of tau_cdf grid, returning random tau energies
        distributed by the cdf corresponding to a given beta angle.

    """

    def __init__(self, config: NssConfig):
        r"""Intialize the Taus object.

        Read local nu2tau table files into NssGrid objects, and produce sampling and
        interpolation functions for local methods.

        Parameters
        ----------
        config: NssConfig
            Configuration object.
        """
        self.config = config

        # grid of pexit table
        with as_file(
            files("nuspacesim.data.nupyprop_tables") / "nu2tau_pexit.hdf5"
        ) as file:
            g = NssGrid.read(file, path="pexit_regen", format="hdf5")
            self.sliced_tau_pexit_grid = grid_slice_interp(
                g, config.simulation.log_nu_tau_energy, "log_e_nu"
            )
            self.pexit_interp = interp1d(
                self.sliced_tau_pexit_grid.axes[0],
                np.log10(self.sliced_tau_pexit_grid.data),
            )

        # grid of tau_cdf tables
        with as_file(
            files("nuspacesim.data.nupyprop_tables") / "nu2tau_cdf.hdf5"
        ) as file:
            self.tau_cdf_grid = NssGrid.read(file, format="hdf5")
            self.tau_cdf_sample = grid_inverse_sampler(
                self.tau_cdf_grid, config.simulation.log_nu_tau_energy
            )

    def tau_exit_prob(self, betas):
        """
        Tau Exit Probability
        """
        mask = betas >= np.radians(1.0)
        tau_exit_prob = np.full(
            betas.shape, 10 ** self.pexit_interp(np.array([np.radians(1.0)]))
        )
        tau_exit_prob[mask] = 10 ** self.pexit_interp(betas[mask])
        return tau_exit_prob

    def tau_energy(self, betas):
        """
        Tau energies interpolated from tau_cdf_sampler for given beta index.
        """
        mask = betas >= np.radians(1.0)
        # E_tau = np.full(betas.shape, self.tau_cdf_sample(np.array([np.radians(1.0)])))
        E_tau = np.empty_like(betas)
        E_tau[mask] = self.tau_cdf_sample(betas[mask])
        E_tau[~mask] = self.tau_cdf_sample(
            np.full(betas[~mask].shape, self.tau_cdf_grid["beta_rad"][0])
        )
        return E_tau * self.config.simulation.nu_tau_energy

    @decorators.nss_result_plot(taus_scatter, taus_histogram, taus_pexit)
    @decorators.nss_result_store("tauBeta", "tauLorentz", "showerEnergy", "tauExitProb")
    def __call__(self, betas):
        r"""Perform main operation for Taus module.

        Parameters
        ----------
        betas: array_like
            beta_tr array of simulated neutrinos.

        Returns
        -------
        tauBeta: array_like
            Tau beta angles.
        tauLorentz: array_like
            Non-deterministically sampled tau lorentz factors. tau energy / tau mass.
        showerEnergy: array_like
            Non-deterministically sampled shower energies in 100 PeV.
        tauExitProb: array_like
            Non-deterministically sampled tau exit probability.
        """

        tauExitProb = self.tau_exit_prob(betas)
        tauEnergy = self.tau_energy(betas)

        # in units of 100 PeV
        showerEnergy = self.config.simulation.e_shower_frac * tauEnergy / 1e8

        tauLorentz = tauEnergy / self.config.constants.massTau

        tauBeta = np.sqrt(1.0 - np.reciprocal(tauLorentz ** 2))

        return tauBeta, tauLorentz, showerEnergy, tauExitProb
