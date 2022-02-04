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
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_profile(x, y, nbins, useStd=True, *args, **kwargs):
    if sum(np.isnan(y)) > 0:
        # print "Array contains NaN, removing them for profile plot"
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]

    n, _ = np.histogram(x, bins=nbins)
    sy, _ = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y * y)
    mean = sy / n
    std = np.sqrt(sy2 / n - mean * mean)
    if not useStd:
        std /= np.sqrt(n)
    bincenter = (_[1:] + _[:-1]) / 2
    binwidth = bincenter - _[1:]

    return bincenter, mean, std, binwidth


def taus_density_beta(inputs, results, *args, **kwargs):
    r"""Plot some density plots"""

    _, betas, log_e_nu = inputs
    tauBeta, tauLorentz, tauEnergy, showerEnergy, tauExitProb = results
    plotting_opts = kwargs.get("kwargs")
    if "default_colormap" in plotting_opts:
        cm = plotting_opts.get("default_colormap")
    else:
        cm = "viridis"

    fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True, constrained_layout=True)

    im = None

    def hist2d(ax, x, y, xlab, ylab):
        nonlocal im
        _, _, _, im = ax.hist2d(
            x=np.degrees(x), y=np.log10(y), bins=(50, 50), cmin=1, cmap=cm
        )
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f"{xlab} vs {ylab}")

    hist2d(ax[0, 0], betas, tauBeta, "β", "log(τ_β)")
    hist2d(ax[0, 1], betas, tauLorentz, "β", "log(τ_Lorentz)")
    hist2d(ax[1, 0], betas, showerEnergy, "β", "log(showerEnergy)")
    hist2d(ax[1, 1], betas, tauExitProb, "β", "log(Exit Probability (τ))")

    fig.colorbar(im, ax=ax, label="Counts", format="%.0e")

    fig.suptitle("Tau interaction properties vs. β_tr Angles")
    if "pop_up" not in plotting_opts:
        plt.show()
    elif "pop_up" in plotting_opts and plotting_opts.get("pop_up") is True:
        plt.show()
    if plotting_opts.get("save_to_file") is True:
        fig.savefig(
            plotting_opts.get("filename")
            + "_taus_density_beta."
            + plotting_opts.get("save_as")
        )


def taus_histogram(inputs, results, *args, **kwargs):
    r"""Plot some histograms"""

    _, _, _ = inputs
    tauBeta, tauLorentz, tauEnergy, showerEnergy, tauExitProb = results

    plotting_opts = kwargs.get("kwargs")
    if "default_color" in plotting_opts:
        c = "C{}".format(plotting_opts.get("default_color"))
    else:
        c = "C0"

    fig, ax = plt.subplots(2, 2, constrained_layout=True)

    ax[0, 0].hist(tauBeta, 100, log=True, facecolor=c)
    ax[0, 0].set_xlabel("log(τ_β)")
    ax[0, 1].hist(tauLorentz, 100, log=True, facecolor=c)
    ax[0, 1].set_xlabel("log(τ_Lorentz)")
    ax[1, 0].hist(showerEnergy, 100, log=True, facecolor=c)
    ax[1, 0].set_xlabel("log(showerEnergy)")
    ax[1, 1].hist(tauExitProb, 100, log=True, facecolor=c)
    ax[1, 1].set_xlabel("log(PExit(τ))")

    fig.suptitle("Tau interaction property Histograms")
    if "pop_up" not in plotting_opts:
        plt.show()
    elif "pop_up" in plotting_opts and plotting_opts.get("pop_up") is True:
        plt.show()
    if plotting_opts.get("save_to_file") is True:
        fig.savefig(
            plotting_opts.get("filename")
            + "_taus_histogram."
            + plotting_opts.get("save_as")
        )


def taus_pexit(inputs, results, *args, **kwargs):
    _, betas, _ = inputs
    _, _, _, _, tauExitProb = results

    plotting_opts = kwargs.get("kwargs")
    if "default_color" in plotting_opts:
        c = "C{}".format(plotting_opts.get("default_color"))
    else:
        c = "C0"
    alpha = 0.1 / np.log10(betas.size)

    fig, ax = plt.subplots(1, 2, constrained_layout=True)

    ax[0].scatter(
        x=np.degrees(betas),
        y=np.log10(tauExitProb),
        s=1,
        c=c,
        marker=".",
        alpha=alpha,
    )
    ax[0].set_xlabel("β")
    ax[0].set_ylabel("log(PExit(τ))")
    ax[0].set_title("β vs Exit Probability.")

    ax[1].hist(tauExitProb, 100, log=True, facecolor=c)
    ax[1].set_ylabel("log(frequency)")
    ax[1].set_xlabel("PExit(τ)")

    fig.suptitle("Tau Pexit")
    if "pop_up" not in plotting_opts:
        plt.show()
    elif "pop_up" in plotting_opts and plotting_opts.get("pop_up") is True:
        plt.show()
    if plotting_opts.get("save_to_file") is True:
        fig.savefig(
            plotting_opts.get("filename")
            + "_taus_pexit."
            + plotting_opts.get("save_as")
        )


def taus_overview(inputs, results, *args, **kwargs):

    r"""Overview plot for taus"""
    plotting_opts = kwargs.get("kwargs")
    if "default_color" in plotting_opts:
        c1, c2, c3 = (
            "C{}".format(plotting_opts.get("default_color")),
            "C{}".format(plotting_opts.get("default_color") + 1),
            "C{}".format(plotting_opts.get("default_color") + 2),
        )
    else:
        c1, c2, c3 = "C0", "C1", "C2"
    if "default_colormap" in plotting_opts:
        cm = plotting_opts.get("default_colormap")
    else:
        cm = "viridis"
    _, betas, log_e_nu = inputs
    _, tauLorentz, tauEnergy, showerEnergy, tauExitProb = results
    log_e_nu = log_e_nu + 9
    tauEnergy = tauEnergy * 1e9
    showerEnergy = showerEnergy * 1e9 * 1e8

    fig, ax = plt.subplots(2, 2, figsize=kwargs.get("figsize"))
    binning_b = np.arange(
        np.min(np.degrees(betas)) - 1, np.max(np.degrees(betas)) + 2, 1
    )
    binning_e = np.arange(
        np.round(np.min(np.log10(showerEnergy)), 1) - 0.1,
        np.round(np.max(log_e_nu), 1) + 0.1,
        0.1,
    )
    ax[0, 0].hist(
        x=log_e_nu,
        bins=binning_e,
        color=c1,
        alpha=0.6,
        label=r"$E_{\nu_\tau}$",
    )
    ax[0, 0].hist(
        x=np.log10(tauEnergy),
        bins=binning_e,
        color=c2,
        alpha=0.6,
        label=r"$E_\tau$",
    )
    ax[0, 0].hist(
        x=np.log10(showerEnergy),
        bins=binning_e,
        color=c3,
        alpha=0.6,
        label=r"$E_\mathrm{shower}$",
    )
    ax[0, 0].set_yscale("log")
    ax[0, 0].legend(loc="upper left")
    ax[0, 0].set_xlabel(
        r"Energy / $\log_\mathrm{10}\left(\frac{E}{\mathrm{eV}}\right)$"
    )
    ax[0, 0].set_ylabel(r"Counts")

    n, bins, _ = ax[0, 1].hist(
        x=np.degrees(betas),
        bins=binning_b,
        color=c1,
    )
    ax[0, 1].set_xlabel(r"Earth emergence angle $\beta$ / $^{\circ}$")
    ax[0, 1].set_ylabel("Counts")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_xlim(min(binning_b), max(binning_b))

    binning_y = np.logspace(
        np.log10(np.min(tauLorentz)) - 0.2, np.log10(np.max(tauLorentz)) + 0.2, 25
    )
    _, _, _, im = ax[1, 0].hist2d(
        np.degrees(betas), tauLorentz, bins=[binning_b, binning_y], cmin=1, cmap=cm
    )
    bincenter, mean, std, binwidth = get_profile(
        np.degrees(betas), tauLorentz, 10, useStd=True, **kwargs
    )
    ax[1, 0].errorbar(
        bincenter,
        mean,
        yerr=std,
        xerr=binwidth,
        color=c2,
        fmt=".",
        lw=2,
        zorder=5,
        label="Profile",
    )
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.0)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Counts")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlim(min(binning_b), max(binning_b))
    ax[1, 0].legend(loc="upper center")
    ax[1, 0].set_xlabel(r"Earth emergence angle $\beta$ / $^{\circ}$")
    ax[1, 0].set_ylabel(r"$\tau_\mathrm{Lorentz}$")

    binning_y = np.logspace(
        np.log10(np.min(tauExitProb)) - 0.2, np.log10(np.max(tauExitProb)) + 0.2, 25
    )
    _, _, _, im = ax[1, 1].hist2d(
        np.degrees(betas),
        tauExitProb,
        bins=[binning_b, binning_y],
        cmin=1,
        cmap=cm,
    )
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.0)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Counts")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlim(min(binning_b), max(binning_b))
    ax[1, 1].set_xlabel(r"Earth emergence angle $\beta$ / $^{\circ}$")
    ax[1, 1].set_ylabel(r"$P_\mathrm{Exit}(\tau)$")

    fig.suptitle("Overview of Tau interaction properties")
    fig.tight_layout()
    if "pop_up" not in plotting_opts:
        plt.show()
    elif "pop_up" in plotting_opts and plotting_opts.get("pop_up") is True:
        plt.show()
    if plotting_opts.get("save_to_file") is True:
        fig.savefig(
            plotting_opts.get("filename")
            + "_taus_overview."
            + plotting_opts.get("save_as")
        )
