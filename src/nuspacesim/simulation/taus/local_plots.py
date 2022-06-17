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

from ...utils.plots import get_profile, hexbin, make_labels

r"""Single plots"""


def energy_hists(inputs, results, fig, ax, *_, **kwargs):
    _, _, log_e_nu = inputs
    _, _, tauEnergy, showerEnergy, _ = results
    log_e_nu = log_e_nu + 9
    tauEnergy = tauEnergy * 1e9
    showerEnergy = showerEnergy * 1e9 * 1e8
    binning_e = np.arange(
        np.round(np.min(np.log10(showerEnergy)), 1) - 0.1,
        np.round(np.max(log_e_nu), 1) + 0.1,
        0.1,
    )
    ax.hist(
        x=log_e_nu,
        bins=binning_e,
        color=kwargs["color"][0],
        alpha=0.6,
        label="$E_{\\nu_\\tau}$",
    )
    ax.hist(
        x=np.log10(tauEnergy),
        bins=binning_e,
        color=kwargs["color"][1],
        alpha=0.6,
        label="$E_\\tau$",
    )
    ax.hist(
        x=np.log10(showerEnergy),
        bins=binning_e,
        color=kwargs["color"][2],
        alpha=0.6,
        label="$E_\\mathrm{shower}$",
    )
    make_labels(
        fig,
        ax,
        "Energy / $\\log_\\mathrm{10}\\left(\\frac{E}{\\mathrm{eV}}\\right)$",
        "Counts",
        logy_scale=True,
    )
    ax.legend(loc="best")


def beta_hist(inputs, _, fig, ax, *__, **kwargs):
    _, betas, _ = inputs
    binning_b = np.arange(
        np.min(np.degrees(betas)) - 1, np.max(np.degrees(betas)) + 2, 1
    )
    ax.hist(
        x=np.degrees(betas),
        bins=binning_b,
        color=kwargs["color"][0],
    )
    make_labels(
        fig,
        ax,
        "Earth emergence angle $\\beta$ / $^{\\circ}$",
        "Counts",
    )


def tau_beta_hist(_, results, fig, ax, *__, **kwargs):
    tauBeta, *_ = results
    ax.hist(
        x=tauBeta,
        bins=np.linspace(min(tauBeta), max(tauBeta), 100),
        color=kwargs["color"][0],
    )
    make_labels(
        fig,
        ax,
        "$\\tau_\\beta$",
        "Counts",
    )


def tau_lorentz_hist(_, results, fig, ax, *__, **kwargs):
    _, tauLorentz, *_ = results
    ax.hist(
        x=tauLorentz,
        bins=np.linspace(min(tauLorentz), max(tauLorentz), 100),
        color=kwargs["color"][0],
    )
    make_labels(
        fig,
        ax,
        "$\\tau_\\mathrm{Lorentz}$",
        "Counts",
    )


def tau_exit_prob_hist(_, results, fig, ax, *__, **kwargs):
    *_, tauExitProb = results
    ax.hist(
        x=tauExitProb,
        bins=np.linspace(min(tauExitProb), max(tauExitProb), 100),
        color=kwargs["color"][0],
    )
    make_labels(
        fig,
        ax,
        "$P_\\mathrm{Exit}(\\tau)$",
        "Counts",
    )


def tau_beta_hex(inputs, results, fig, ax, *args, **kwargs):
    _, betas, log_e_nu = inputs
    tauBeta, tauLorentz, tauEnergy, showerEnergy, tauExitProb = results
    binning_b = np.arange(
        np.min(np.degrees(betas)) - 1, np.max(np.degrees(betas)) + 2, 1
    )
    im = hexbin(
        ax,
        x=np.degrees(betas),
        y=tauBeta,
        gs=len(binning_b),
        logy_scale=True,
        cmap=kwargs["cmap"],
    )
    make_labels(
        fig,
        ax,
        "Earth emergence angle $\\beta$ / $^{\\circ}$",
        "$\\tau_\\beta$",
        clabel="Counts",
        im=im,
        logy_scale=True,
    )


def tau_lorentz_hex(inputs, results, fig, ax, *args, **kwargs):
    _, betas, log_e_nu = inputs
    tauBeta, tauLorentz, tauEnergy, showerEnergy, tauExitProb = results
    binning_b = np.arange(
        np.min(np.degrees(betas)) - 1, np.max(np.degrees(betas)) + 2, 1
    )
    im = hexbin(
        ax,
        x=np.degrees(betas),
        y=tauLorentz,
        gs=len(binning_b),
        logy_scale=True,
        cmap=kwargs["cmap"],
    )
    bincenter, mean, std, binwidth = get_profile(
        np.degrees(betas), tauLorentz, 10, useStd=True
    )
    ax.errorbar(
        bincenter,
        mean,
        xerr=binwidth,
        yerr=std,
        color=kwargs["color"][0],
        fmt=".",
        label="Profile",
    )
    ax.legend(loc="best")
    make_labels(
        fig,
        ax,
        "Earth emergence angle $\\beta$ / $^{\\circ}$",
        "$\\tau_\\mathrm{Lorentz}$",
        clabel="Counts",
        im=im,
        logy_scale=True,
    )


def tau_exit_prob_hex(inputs, results, fig, ax, *args, **kwargs):
    _, betas, log_e_nu = inputs
    *_, tauExitProb = results
    binning_b = np.arange(
        np.min(np.degrees(betas)) - 1, np.max(np.degrees(betas)) + 2, 1
    )
    im = hexbin(
        ax,
        x=np.degrees(betas),
        y=tauExitProb,
        gs=len(binning_b),
        logy_scale=True,
        cmap=kwargs["cmap"],
    )
    make_labels(
        fig,
        ax,
        "Earth emergence angle $\\beta$ / $^{\\circ}$",
        "$P_\\mathrm{Exit}(\\tau)$",
        clabel="Counts",
        im=im,
        logy_scale=True,
    )


r"""Multiplot collections"""


def taus_density_beta_overview(inputs, results, fig, ax, *args, **kwargs):
    r"""Plot some density plots"""

    ax.remove()
    fig.set_size_inches(15, 4)
    ax = fig.subplots(nrows=1, ncols=3)
    fig.suptitle("Tau interaction properties vs. Earth emergence angles $\\beta$")
    tau_beta_hex(inputs, results, fig, ax[0], *args, **kwargs)
    tau_lorentz_hex(inputs, results, fig, ax[1], *args, **kwargs)
    tau_exit_prob_hex(inputs, results, fig, ax[2], *args, **kwargs)

    return "taus_density_beta"


def taus_histograms_overview(inputs, results, fig, ax, *args, **kwargs):
    r"""Plot some histograms"""

    ax.remove()
    fig.set_size_inches(15, 4)
    ax = fig.subplots(nrows=1, ncols=3)
    fig.suptitle("$\\tau$ interaction property Histograms")

    tau_beta_hist(inputs, results, fig, ax[0], *args, **kwargs)
    tau_lorentz_hist(inputs, results, fig, ax[1], *args, **kwargs)
    tau_exit_prob_hist(inputs, results, fig, ax[2], *args, **kwargs)

    return "taus_histogram"


def taus_pexit_overview(inputs, results, fig, ax, *args, **kwargs):

    ax.remove()
    fig.set_size_inches(10, 4)
    ax = fig.subplots(nrows=1, ncols=2)
    fig.suptitle("$\\tau$ exit probability")

    tau_exit_prob_hex(inputs, results, fig, ax[0], *args, **kwargs)
    tau_exit_prob_hist(inputs, results, fig, ax[1], *args, **kwargs)

    return "taus_pexit"


def taus_overview(inputs, results, fig, ax, *args, **kwargs):
    r"""Overview plot for taus"""

    ax.remove()
    fig.set_size_inches(10, 8)
    ax = fig.subplots(nrows=2, ncols=2)
    fig.suptitle("Overview of Tau interaction properties")

    energy_hists(inputs, results, fig, ax[0, 0], *args, **kwargs)
    beta_hist(inputs, results, fig, ax[0, 1], *args, **kwargs)
    tau_lorentz_hex(inputs, results, fig, ax[1, 0], *args, **kwargs)
    tau_exit_prob_hex(inputs, results, fig, ax[1, 1], *args, **kwargs)

    return "taus_overview"
