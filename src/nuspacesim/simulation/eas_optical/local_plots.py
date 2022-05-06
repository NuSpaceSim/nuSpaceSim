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

from ...utils.plot_wrapper import PlotWrapper


def numpes_vs_beta(inputs, results, fig, ax, *args, **kwargs):
    _, betas, altDec, showerEnergy = inputs
    numPEs, costhetaChEff = results
    binning_b = np.arange(
        np.min(np.degrees(betas)) - 1, np.max(np.degrees(betas)) + 2, 1
    )
    im = fig.hexbin(
        ax,
        x=np.degrees(betas),
        y=numPEs,
        gs=len(binning_b),
        logx=True,
        logy=True,
    )
    fig.make_labels(
        ax,
        "$\\beta$ / $^{\\circ}$",
        "#PEs",
        clabel="Counts",
        im=im,
        logx=True,
        logy=True,
    )


def altdec_vs_numpes(inputs, results, fig, ax, *args, **kwargs):
    _, betas, altDec, showerEnergy = inputs
    numPEs, costhetaChEff = results

    im = fig.hexbin(
        ax,
        x=altDec,
        y=numPEs,
        gs=25,
        logx=True,
        logy=True,
    )
    fig.make_labels(
        ax,
        "$Decay_\\mathrm{Altitude}$ / km",
        "#PEs",
        clabel="Counts",
        im=im,
        logx=True,
        logy=True,
    )


def altdec_vs_beta(inputs, results, fig, ax, *args, **kwargs):
    _, betas, altDec, showerEnergy = inputs
    numPEs, costhetaChEff = results
    binning_b = np.arange(
        np.min(np.degrees(betas)) - 1, np.max(np.degrees(betas)) + 2, 1
    )
    im = fig.hexbin(
        ax,
        x=np.degrees(betas),
        y=altDec,
        gs=len(binning_b),
        logx=True,
        logy=True,
    )
    fig.make_labels(
        ax,
        "$\\beta$ / $^{\\circ}$",
        "$Decay_\\mathrm{Altitude}$ / km",
        clabel="Counts",
        im=im,
        logx=True,
        logy=True,
    )


def showerenergy_vs_numpes(inputs, results, fig, ax, *args, **kwargs):
    _, betas, altDec, showerEnergy = inputs
    numPEs, costhetaChEff = results

    im = fig.hexbin(
        ax,
        x=showerEnergy,
        y=numPEs,
        gs=25,
        logx=True,
        logy=True,
    )
    fig.make_labels(
        ax,
        "$E_\\mathrm{shower}$ / 100 PeV",
        "#PEs",
        clabel="Counts",
        im=im,
        logx=True,
        logy=True,
    )


def costhetacheff_vs_beta(inputs, results, fig, ax, *args, **kwargs):
    _, betas, altDec, showerEnergy = inputs
    numPEs, costhetaChEff = results
    binning_b = np.arange(
        np.min(np.degrees(betas)) - 1, np.max(np.degrees(betas)) + 2, 1
    )
    im = fig.hexbin(
        ax,
        x=np.degrees(betas),
        y=costhetaChEff,
        gs=len(binning_b),
        logx=True,
        logy=True,
    )
    fig.make_labels(
        ax,
        "$\\beta$ / $^{\\circ}$",
        "$\\cos(\\theta_\\mathrm{Cherenkov}$ / rad)",
        clabel="Counts",
        im=im,
        logx=True,
        logy=True,
    )


def costhetacheff_vs_numpes(inputs, results, fig, ax, *args, **kwargs):
    _, betas, altDec, showerEnergy = inputs
    numPEs, costhetaChEff = results
    im = fig.hexbin(
        ax,
        x=numPEs,
        y=costhetaChEff,
        gs=25,
        logx=True,
        logy=True,
    )
    fig.make_labels(
        ax,
        "#PEs",
        "$\\cos(\\theta_\\mathrm{Cherenkov}$ / rad)",
        clabel="Counts",
        im=im,
        logx=True,
        logy=True,
    )


def altdec_vs_costhetacheff(inputs, results, fig, ax, *args, **kwargs):
    _, betas, altDec, showerEnergy = inputs
    numPEs, costhetaChEff = results
    im = fig.hexbin(
        ax,
        x=altDec,
        y=costhetaChEff,
        gs=25,
        logx=True,
        logy=True,
    )
    fig.make_labels(
        ax,
        "$Decay_\\mathrm{Altitude}$ / km",
        "$\\cos(\\theta_\\mathrm{Cherenkov}$ / rad)",
        clabel="Counts",
        im=im,
        logx=True,
        logy=True,
    )


def showerenergy_vs_costhetacheff(inputs, results, fig, ax, *args, **kwargs):
    _, betas, altDec, showerEnergy = inputs
    numPEs, costhetaChEff = results
    im = fig.hexbin(
        ax,
        x=showerEnergy,
        y=costhetaChEff,
        gs=25,
        logx=True,
        logy=True,
    )
    fig.make_labels(
        ax,
        "$E_\\mathrm{shower}$ / 100 PeV",
        "$\\cos(\\theta_\\mathrm{Cherenkov}$ / rad)",
        clabel="Counts",
        im=im,
        logx=True,
        logy=True,
    )


def numpes_hist(inputs, results, fig, ax, *args, **kwargs):
    _, betas, altDec, showerEnergy = inputs
    numPEs, costhetaChEff = results
    ax.hist(
        x=numPEs,
        bins=np.linspace(min(numPEs), max(numPEs), 100),
        color=fig.params["default_colors"][0],
    )
    fig.make_labels(
        ax,
        "#PEs",
        "Counts",
    )


def costhetacheff_hist(inputs, results, fig, ax, *args, **kwargs):
    _, betas, altDec, showerEnergy = inputs
    numPEs, costhetaChEff = results
    ax.hist(
        x=costhetaChEff,
        bins=np.linspace(min(costhetaChEff), max(costhetaChEff), 100),
        color=fig.params["default_colors"][0],
    )
    fig.make_labels(
        ax,
        "$\\cos(\\theta_\\mathrm{Cherenkov}$ / rad)",
        "Counts",
    )


def eas_optical_density_overview(inputs, results, plot_kwargs, *args, **kwargs):
    r"""Plot some density plots"""

    _, betas, altDec, showerEnergy = inputs
    numPEs, costhetaChEff = results
    fig = PlotWrapper(plot_kwargs, 2, 3, (15, 8), "EAS Optical Cherenkov properties")
    numpes_vs_beta(inputs, results, fig, fig.ax[0, 0])
    altdec_vs_numpes(inputs, results, fig, fig.ax[0, 1])
    showerenergy_vs_numpes(inputs, results, fig, fig.ax[0, 2])
    costhetacheff_vs_beta(inputs, results, fig, fig.ax[1, 0])
    altdec_vs_costhetacheff(inputs, results, fig, fig.ax[1, 1])
    showerenergy_vs_costhetacheff(inputs, results, fig, fig.ax[1, 2])
    fig.close(
        "eas_optical_density_overview", fig.params["save_to_file"], fig.params["pop_up"]
    )


def eas_optical_histogram_overview(inputs, results, plot_kwargs, *args, **kwargs):
    r"""Plot some histograms"""

    eas_self, betas, altDec, showerEnergy = inputs
    numPEs, costhetaChEff = results

    fig = PlotWrapper(
        plot_kwargs, 2, 1, title="EAS Optical Cherenkov property Histograms"
    )
    numpes_hist(inputs, results, fig, fig.ax[0])
    costhetacheff_hist(inputs, results, fig, fig.ax[1])
    fig.close("eas_optical_histogram", fig.params["save_to_file"], fig.params["pop_up"])
