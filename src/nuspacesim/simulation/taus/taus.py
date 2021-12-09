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
from scipy.interpolate import RegularGridInterpolator, interp1d

from ...config import NssConfig
from ...utils import decorators
from ...utils.cdf import grid_cdf_sampler, grid_inverse_sampler
from ...utils.grid import NssGrid
from ...utils.interp import grid_slice_interp
from .local_plots import taus_histogram, taus_overview, taus_pexit, taus_scatter

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
            self.pexit_grid = NssGrid.read(file, path="pexit_regen", format="hdf5")

        # grid of tau_cdf tables
        with as_file(
            files("nuspacesim.data.nupyprop_tables") / "nu2tau_cdf.hdf5"
        ) as file:
            self.tau_cdf_grid = NssGrid.read(file, format="hdf5")

    def tau_exit_prob(self, betas, log_e_nu):
        """
        Tau Exit Probability
        """
        mask = betas >= np.radians(1.0)

        if isinstance(log_e_nu, (int, float)):
            sliced_tau_pexit_grid = grid_slice_interp(
                self.pexit_grid, log_e_nu, "log_e_nu"
            )
            pexit_interp = interp1d(
                sliced_tau_pexit_grid.axes[0],
                np.log10(sliced_tau_pexit_grid.data),
            )
            Pexit = np.full(
                betas.shape, 10 ** pexit_interp(np.array([np.radians(1.0)]))
            )
            Pexit[mask] = 10 ** pexit_interp(betas[mask])
        elif hasattr(log_e_nu, "__getitem__"):
            pexit_interp = RegularGridInterpolator(
                self.pexit_grid.axes, np.log10(self.pexit_grid.data)
            )
            Pexit = np.empty_like(betas)
            Pexit[mask] = 10 ** pexit_interp((log_e_nu[mask], betas[mask]))
            Pexit[~mask] = 10 ** pexit_interp((log_e_nu[~mask], np.radians(1.0)))
        else:
            raise RuntimeError(f"log_e_nu type not recognized: {type(log_e_nu)}")

        return Pexit

    def tau_energy(self, betas, log_e_nu):
        """
        Tau energies interpolated from tau_cdf_sampler for given beta index.
        """

        mask = betas >= np.radians(1.0)
        E_tau = np.empty_like(betas)

        if isinstance(log_e_nu, (int, float)):
            tau_cdf_sample = grid_inverse_sampler(self.tau_cdf_grid, log_e_nu)
            E_tau[mask] = tau_cdf_sample(betas[mask])
            E_tau[~mask] = tau_cdf_sample(
                np.full(betas[~mask].shape, self.tau_cdf_grid["beta_rad"][0])
            )
        elif hasattr(log_e_nu, "__getitem__"):
            tau_cdf_sample = grid_cdf_sampler(self.tau_cdf_grid)
            E_tau[mask] = tau_cdf_sample(log_e_nu[mask], betas[mask])
            E_tau[~mask] = tau_cdf_sample(
                log_e_nu[~mask],
                np.full(betas[~mask].shape, self.tau_cdf_grid["beta_rad"][0]),
            )
        else:
            raise RuntimeError(f"log_e_nu type not recognized: {type(log_e_nu)}")

        return E_tau * 10 ** log_e_nu

    @decorators.nss_result_plot(taus_scatter, taus_histogram, taus_pexit, taus_overview)
    @decorators.nss_result_store(
        "tauBeta", "tauLorentz", "tauEnergy", "showerEnergy", "tauExitProb"
    )
    def __call__(self, betas, log_e_nu, *args, **kwargs):
        r"""Perform main operation for Taus module.

        Parameters
        ----------
        betas: array_like
            beta_tr array of simulated neutrinos.

        Returns
        -------
        tauBeta: array_like
            Tau beta factor calculated from tauLorentz.
        tauLorentz: array_like
            Non-deterministically sampled tau lorentz factors. tau energy / tau mass.
        showerEnergy: array_like
            Non-deterministically sampled shower energies in 100 PeV.
        tauExitProb: array_like
            Non-deterministically sampled tau exit probability.
        """

        tauExitProb = self.tau_exit_prob(betas, log_e_nu)
        tauEnergy = self.tau_energy(betas, log_e_nu)

        # in units of 100 PeV
        showerEnergy = self.config.simulation.e_shower_frac * tauEnergy / 1e8

        tauLorentz = tauEnergy / self.config.constants.massTau

        tauBeta = np.sqrt(1.0 - np.reciprocal(tauLorentz ** 2))

        return tauBeta, tauLorentz, tauEnergy, showerEnergy, tauExitProb
