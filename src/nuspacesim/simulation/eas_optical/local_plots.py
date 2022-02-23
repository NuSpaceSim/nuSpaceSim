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

import numpy as np
from matplotlib import pyplot as plt

from ...utils.plots import hexbin


def eas_optical_density(inputs, results, *args, **kwargs):
    r"""Plot some density plots"""

    _, betas, altDec, showerEnergy = inputs
    numPEs, costhetaChEff = results

    plotting_opts = kwargs.get("kwargs")
    if "default_colormap" in plotting_opts:
        cm = plotting_opts.get("default_colormap")
    else:
        cm = "viridis"

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    hexbin(
        fig,
        ax[0, 0],
        np.degrees(betas),
        numPEs,
        "$\\beta$ / $^{\\circ}$",
        "#PEs",
        cm=cm,
    )
    hexbin(
        fig,
        ax[1, 0],
        np.degrees(betas),
        costhetaChEff,
        "$\\beta$ / $^{\\circ}$",
        "$\\cos(\\theta_\\mathrm{Cherenkov}$ / rad)",
        cm=cm,
    )

    hexbin(
        fig,
        ax[0, 1],
        altDec,
        numPEs,
        "$Decay_\\mathrm{Altitude}$ / km",
        "#PEs",
        cm=cm,
    )

    hexbin(
        fig,
        ax[1, 1],
        altDec,
        costhetaChEff,
        "Decay_\\mathrm{Altitude}$ / km",
        "$\\cos(\\theta_\\mathrm{Cherenkov}$ / rad)",
        cm=cm,
    )

    hexbin(
        fig,
        ax[0, 2],
        showerEnergy,
        numPEs,
        "$E_\\mathrm{shower}$ / 100 PeV",
        "#PEs",
        cm=cm,
    )

    hexbin(
        fig,
        ax[1, 2],
        showerEnergy,
        costhetaChEff,
        "$E_\\mathrm{shower}$ / 100 PeV",
        "$\\cos(\\theta_\\mathrm{Cherenkov}$ / rad)",
        cm=cm,
    )

    fig.suptitle("EAS Optical Cherenkov properties.")

    if plotting_opts.get("pop_up") is True:
        plt.show()
    if plotting_opts.get("save_to_file") is True:
        fig.savefig(
            plotting_opts.get("filename")
            + "_eas_optical_density."
            + plotting_opts.get("save_as")
        )


def eas_optical_histogram(inputs, results, *args, **kwargs):
    r"""Plot some histograms"""

    # eas_self, betas, altDec, showerEnergy = inputs
    plotting_opts = kwargs.get("kwargs")
    if "default_color" in plotting_opts:
        c = "C{}".format(plotting_opts.get("default_color"))
    else:
        c = "C0"
    numPEs, costhetaChEff = results

    fig, ax = plt.subplots(2, 1, constrained_layout=True)

    ax[0].hist(numPEs, 100, log=True, facecolor=c)
    ax[0].set_xlabel("log(#PEs)")
    ax[0].set_ylabel("Counts")

    ax[1].hist(costhetaChEff, 100, log=True, facecolor=c)
    ax[1].set_xlabel("$\\cos(\\theta_\\mathrm{Cherenkov}$ / rad)")
    ax[1].set_ylabel("Counts")

    fig.suptitle("EAS Optical Cherenkov property Histograms")
    if "pop_up" not in plotting_opts:
        plt.show()
    elif "pop_up" in plotting_opts and plotting_opts.get("pop_up") is True:
        plt.show()
    if plotting_opts.get("save_to_file") is True:
        fig.savefig(
            plotting_opts.get("filename")
            + "_eas_optical_histogram."
            + plotting_opts.get("save_as")
        )
