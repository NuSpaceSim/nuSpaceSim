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

from ...utils.plots import hexbin, make_labels


def numpes_vs_beta(inputs, results, fig, ax, *_, **kwargs):
    _, betas, _, _ = inputs
    numPEs, _ = results
    binning_b = np.arange(
        np.min(np.degrees(betas)) - 1, np.max(np.degrees(betas)) + 2, 1
    )
    im = hexbin(
        ax,
        x=np.degrees(betas),
        y=numPEs,
        gs=len(binning_b),
        logx=True,
        logy=True,
        cmap=kwargs["cmap"],
    )
    make_labels(
        fig,
        ax,
        "$\\beta$ / $^{\\circ}$",
        "#PEs",
        clabel="Counts",
        im=im,
        logx=True,
        logy=True,
    )


def altdec_vs_numpes(inputs, results, fig, ax, *_, **kwargs):
    _, _, altDec, _ = inputs
    numPEs, _ = results

    im = hexbin(
        ax,
        x=altDec,
        y=numPEs,
        gs=25,
        logx=True,
        logy=True,
        cmap=kwargs["cmap"],
    )
    make_labels(
        fig,
        ax,
        "$Decay_\\mathrm{Altitude}$ / km",
        "#PEs",
        clabel="Counts",
        im=im,
        logx=True,
        logy=True,
    )


def altdec_vs_beta(inputs, results, fig, ax, *_, **kwargs):
    _, betas, altDec, _ = inputs
    binning_b = np.arange(
        np.min(np.degrees(betas)) - 1, np.max(np.degrees(betas)) + 2, 1
    )
    im = hexbin(
        ax,
        x=np.degrees(betas),
        y=altDec,
        gs=len(binning_b),
        logx=True,
        logy=True,
        cmap=kwargs["cmap"],
    )
    make_labels(
        fig,
        ax,
        "$\\beta$ / $^{\\circ}$",
        "$Decay_\\mathrm{Altitude}$ / km",
        clabel="Counts",
        im=im,
        logx=True,
        logy=True,
    )


def showerenergy_vs_numpes(inputs, results, fig, ax, *args, **kwargs):
    _, _, _, showerEnergy = inputs
    numPEs, _ = results

    im = hexbin(
        ax,
        x=showerEnergy,
        y=numPEs,
        gs=25,
        logx=True,
        logy=True,
        cmap=kwargs["cmap"],
    )
    make_labels(
        fig,
        ax,
        "$E_\\mathrm{shower}$ / 100 PeV",
        "#PEs",
        clabel="Counts",
        im=im,
        logx=True,
        logy=True,
    )


def costhetacheff_vs_beta(inputs, results, fig, ax, *_, **kwargs):
    _, betas, _, _ = inputs
    _, costhetaChEff = results
    binning_b = np.arange(
        np.min(np.degrees(betas)) - 1, np.max(np.degrees(betas)) + 2, 1
    )
    im = hexbin(
        ax,
        x=np.degrees(betas),
        y=costhetaChEff,
        gs=len(binning_b),
        logx=True,
        logy=True,
        cmap=kwargs["cmap"],
    )
    make_labels(
        fig,
        ax,
        "$\\beta$ / $^{\\circ}$",
        "$\\cos(\\theta_\\mathrm{Cherenkov}$ / rad)",
        clabel="Counts",
        im=im,
        logx=True,
        logy=True,
    )


def costhetacheff_vs_numpes(inputs, results, fig, ax, *_, **kwargs):
    numPEs, costhetaChEff = results
    im = hexbin(
        ax,
        x=numPEs,
        y=costhetaChEff,
        gs=25,
        logx=True,
        logy=True,
        cmap=kwargs["cmap"],
    )
    make_labels(
        fig,
        ax,
        "#PEs",
        "$\\cos(\\theta_\\mathrm{Cherenkov}$ / rad)",
        clabel="Counts",
        im=im,
        logx=True,
        logy=True,
    )


def altdec_vs_costhetacheff(inputs, results, fig, ax, *_, **kwargs):
    _, _, altDec, _ = inputs
    _, costhetaChEff = results
    im = hexbin(
        ax,
        x=altDec,
        y=costhetaChEff,
        gs=25,
        logx=True,
        logy=True,
        cmap=kwargs["cmap"],
    )
    make_labels(
        fig,
        ax,
        "$Decay_\\mathrm{Altitude}$ / km",
        "$\\cos(\\theta_\\mathrm{Cherenkov}$ / rad)",
        clabel="Counts",
        im=im,
        logx=True,
        logy=True,
    )


def showerenergy_vs_costhetacheff(inputs, results, fig, ax, *_, **kwargs):
    _, _, _, showerEnergy = inputs
    _, costhetaChEff = results
    im = hexbin(
        ax,
        x=showerEnergy,
        y=costhetaChEff,
        gs=25,
        logx=True,
        logy=True,
        cmap=kwargs["cmap"],
    )
    make_labels(
        fig,
        ax,
        "$E_\\mathrm{shower}$ / 100 PeV",
        "$\\cos(\\theta_\\mathrm{Cherenkov}$ / rad)",
        clabel="Counts",
        im=im,
        logx=True,
        logy=True,
    )


def numpes_hist(inputs, results, fig, ax, *_, **kwargs):
    numPEs, _ = results
    ax.hist(
        x=numPEs,
        bins=np.linspace(min(numPEs), max(numPEs), 100),
        color=kwargs["color"][0],
    )
    make_labels(
        fig,
        ax,
        "#PEs",
        "Counts",
    )


def costhetacheff_hist(inputs, results, fig, ax, *_, **kwargs):
    _, costhetaChEff = results
    ax.hist(
        x=costhetaChEff,
        bins=np.linspace(min(costhetaChEff), max(costhetaChEff), 100),
        color=kwargs["color"][0],
    )
    make_labels(
        fig,
        ax,
        "$\\cos(\\theta_\\mathrm{Cherenkov}$ / rad)",
        "Counts",
    )


def eas_optical_density_overview(inputs, results, fig, ax, *args, **kwargs):
    r"""Plot some density plots"""

    ax.remove()
    fig.set_size_inches(15, 8)
    ax = fig.subplots(nrows=2, ncols=3)
    fig.suptitle("EAS Optical Cherenkov properties")

    numpes_vs_beta(inputs, results, fig, ax[0, 0], *args, **kwargs)
    altdec_vs_numpes(inputs, results, fig, ax[0, 1], *args, **kwargs)
    showerenergy_vs_numpes(inputs, results, fig, ax[0, 2], *args, **kwargs)
    costhetacheff_vs_beta(inputs, results, fig, ax[1, 0], *args, **kwargs)
    altdec_vs_costhetacheff(inputs, results, fig, ax[1, 1], *args, **kwargs)
    showerenergy_vs_costhetacheff(inputs, results, fig, ax[1, 2], *args, **kwargs)

    return "eas_optical_density_overview"


def eas_optical_histogram_overview(inputs, results, fig, ax, *args, **kwargs):
    r"""Plot some histograms"""

    # fig = PlotWrapper(
    #     plot_kwargs, 2, 1, title="EAS Optical Cherenkov property Histograms"
    # )

    ax.remove()
    ax = fig.subplots(nrows=2, ncols=1)
    fig.suptitle("EAS Optical Cherenkov property Histograms")

    numpes_hist(inputs, results, fig, ax[0], *args, **kwargs)
    costhetacheff_hist(inputs, results, fig, ax[1], *args, **kwargs)
    return "eas_optical_histogram_overview"
