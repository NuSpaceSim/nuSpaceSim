# The Clear BSD License
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

import numpy as np
from matplotlib import pyplot as plt

from ..simulation import eas_optical, taus
from . import decorators
from .plots import hist2d

__all__ = ["dashboard", "show_plot"]


def dashboard(sim, _, fig, ax, *args, **kwargs):
    """Full dashboard of plots"""

    ax.remove()
    fig.set_size_inches(15, 8)
    ax = fig.subplots(nrows=3, ncols=4)
    fig.suptitle("Nuspacesim Results Dashboard")

    tau_input = None, sim["beta_rad"], sim["log_e_nu"]
    tau_results = (
        sim["tauBeta"],
        sim["tauLorentz"],
        sim["tauEnergy"],
        sim["showerEnergy"],
        sim["tauExitProb"],
    )

    eas_input = None, sim["beta_rad"], sim["altDec"], sim["showerEnergy"]
    eas_results = sim["numPEs"], sim["costhetaChEff"]

    taus.local_plots.energy_hists(
        tau_input, tau_results, fig, ax[0, 0], *args, **kwargs
    )
    taus.local_plots.tau_exit_prob_hist(
        tau_input, tau_results, fig, ax[1, 0], *args, **kwargs
    )
    taus.local_plots.tau_lorentz_hex(
        tau_input, tau_results, fig, ax[2, 0], *args, **kwargs
    )

    taus.local_plots.beta_hist(tau_input, tau_results, fig, ax[0, 1], *args, **kwargs)
    taus.local_plots.tau_exit_prob_hex(
        tau_input, tau_results, fig, ax[1, 1], *args, **kwargs
    )
    eas_optical.local_plots.altdec_vs_beta(
        eas_input, eas_results, fig, ax[2, 1], *args, **kwargs
    )

    eas_optical.local_plots.numpes_hist(
        eas_input, eas_results, fig, ax[0, 2], *args, **kwargs
    )
    eas_optical.local_plots.numpes_vs_beta(
        eas_input, eas_results, fig, ax[1, 2], *args, **kwargs
    )
    eas_optical.local_plots.altdec_vs_numpes(
        eas_input, eas_results, fig, ax[2, 2], *args, **kwargs
    )

    # eas_input_density(sim, fig, ax[0, 3])
    eas_optical.local_plots.costhetacheff_hist(
        eas_input, eas_results, fig, ax[0, 3], *args, **kwargs
    )
    eas_optical.local_plots.costhetacheff_vs_numpes(
        eas_input, eas_results, fig, ax[1, 3], *args, **kwargs
    )
    eas_optical.local_plots.costhetacheff_vs_beta(
        eas_input, eas_results, fig, ax[2, 3], *args, **kwargs
    )


@decorators.ensure_plot_registry(dashboard)
def show_plot(sim, plot_wrapper):
    plot_wrapper(sim, None, (dashboard,))
