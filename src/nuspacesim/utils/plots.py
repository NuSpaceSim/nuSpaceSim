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

from . import decorators

__all__ = ["dashboard", "energy_histograms", "show_plot"]


def hexbin(fig, ax, x, y, xlab, ylab, cm="viridis", logx=True, logy=True):

    xf = np.log10 if logx else lambda q: q
    yf = np.log10 if logy else lambda q: q

    xl = f"log({xlab})" if logx else xlab
    yl = f"log({ylab})" if logy else ylab

    xmask = x > 0 if logx else np.full(x.shape, True)
    ymask = y > 0 if logy else np.full(y.shape, True)
    m = xmask & ymask

    im = ax.hexbin(
        x=xf(x[m]), y=yf(y[m]), gridsize=25, mincnt=1, cmap=cm, edgecolors="none"
    )

    ax.set_xlabel(xl)
    ax.set_ylabel(yl)

    ax.set_title(f"{yl} vs {xl}")
    cbar = fig.colorbar(im, ax=ax, pad=0.0)
    cbar.set_label("Counts")


def hist2d(fig, ax, x, y, xlab, ylab, cm="viridis", logx=True, logy=True):

    xf = np.log10 if logx else lambda q: q
    yf = np.log10 if logy else lambda q: q

    xl = f"log({xlab})" if logx else xlab
    yl = f"log({ylab})" if logy else ylab

    xmask = x > 0 if logx else np.full(x.shape, True)
    ymask = y > 0 if logy else np.full(y.shape, True)
    m = xmask & ymask

    _, _, _, im = ax.hist2d(x=xf(x[m]), y=yf(y[m]), bins=(50, 50), cmin=1, cmap=cm)

    ax.set_xlabel(xl)
    ax.set_ylabel(yl)

    ax.set_title(f"{yl} vs {xl}")
    cbar = fig.colorbar(im, ax=ax, pad=0.0)
    cbar.set_label("Counts")


def dashboard(sim, plot_kwargs={}):
    """Full dashboard of plots"""
    if plot_kwargs:
        if "default_color" in plot_kwargs:
            c1, c2, c3 = (
                "C{}".format(plot_kwargs.get("default_color")),
                "C{}".format(plot_kwargs.get("default_color") + 1),
                "C{}".format(plot_kwargs.get("default_color") + 2),
            )
        if "default_colormap" in plot_kwargs:
            cm = plot_kwargs.get("default_colormap")
    else:
        c1, c2, c3 = "C0", "C1", "C2"
        cm = "viridis"

    fig, ax = plt.subplots(3, 4, figsize=(14, 8), constrained_layout=True)

    energy_histograms(sim, fig, ax[0, 0], [c1, c2, c3])
    tau_pexit_hist(sim, fig, ax[1, 0], c1)
    tau_lorentz(sim, fig, ax[2, 0], cm)

    betas_histogram(sim, fig, ax[0, 1], c1)
    tau_pexit_density(sim, fig, ax[1, 1], cm)
    decay_altitude(sim, fig, ax[2, 1], cm)

    # decay_altitude_hist(sim, fig, ax[0, 2])
    num_photo_electrons_hist(sim, fig, ax[0, 2], c1)
    num_photo_electrons_density(sim, fig, ax[1, 2], cm)
    num_photo_electrons_altitude(sim, fig, ax[2, 2], cm)

    # eas_input_density(sim, fig, ax[0, 3])
    cherenkov_angle_hist(sim, fig, ax[0, 3], c1)
    eas_results_density(sim, fig, ax[1, 3], cm)
    cherenkov_angle(sim, fig, ax[2, 3], cm)

    # tau_betas(sim, fig, ax[1, 2])

    fig.suptitle("Nuspacesim Results Dashboard", size="x-large")
    if "pop_up" not in plot_kwargs:
        plt.show()
    elif "pop_up" in plot_kwargs and plot_kwargs.get("pop_up") is True:
        plt.show()
    if plot_kwargs.get("save_to_file") is True:
        fig.savefig(
            plot_kwargs.get("filename") + "_dashboard." + plot_kwargs.get("save_as")
        )


def energy_histograms(sim, fig, ax, c):

    energy_bins = np.arange(
        np.round(np.min(np.log10(sim["showerEnergy"]) + 17), 1) - 0.1,
        np.round(np.max(sim["log_e_nu"] + 9), 1) + 0.1,
        0.1,
    )
    ax.hist(
        x=sim["log_e_nu"] + 9,
        bins=energy_bins,
        color=c[0],
        alpha=0.6,
        label=r"$E_{\nu_\tau}$",
    )
    ax.hist(
        x=np.log10(sim["tauEnergy"]) + 9,
        bins=energy_bins,
        color=c[1],
        alpha=0.6,
        label=r"$E_\tau$",
    )
    ax.hist(
        x=np.log10(sim["showerEnergy"]) + 17,
        bins=energy_bins,
        color=c[2],
        alpha=0.6,
        label=r"$E_\mathrm{shower}$",
    )
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.set_xlabel(r"Energy / $\log_\mathrm{10}\left(\frac{E}{\mathrm{eV}}\right)$")
    ax.set_ylabel(r"Counts")


def betas_histogram(sim, fig, ax, c):

    beta_bins = np.arange(
        np.min(np.degrees(sim["beta_rad"])) - 1,
        np.max(np.degrees(sim["beta_rad"])) + 2,
        1,
    )

    ax.hist(x=np.degrees(sim["beta_rad"]), bins=beta_bins, color=c)
    ax.set_xlabel(r"Earth emergence angle $\beta$ / $^{\circ}$")
    ax.set_ylabel("Counts")
    ax.set_yscale("log")
    ax.set_xlim(min(beta_bins), max(beta_bins))


def tau_lorentz(sim, fig, ax, cm):
    hexbin(
        fig,
        ax,
        np.degrees(sim["beta_rad"]),
        sim["tauLorentz"],
        r"$\beta$",
        r"$τ_\mathrm{Lorentz}$",
        cm=cm,
        logx=False,
        logy=True,
    )


def tau_pexit_hist(sim, fig, ax, c):
    ax.hist(sim["tauExitProb"], 100, log=True, color=c)
    ax.set_ylabel("Counts")
    ax.set_xlabel(r"$\log(P_\mathrm{exit}(\tau))$")


def tau_pexit_density(sim, fig, ax, cm):
    hexbin(
        fig,
        ax,
        np.degrees(sim["beta_rad"]),
        sim["tauExitProb"],
        r"$\beta$",
        r"$P_\mathrm{exit}(\tau)$",
        cm=cm,
        logx=False,
        logy=True,
    )


def tau_betas(sim, fig, ax, cm):
    hexbin(
        fig,
        ax,
        sim["beta_rad"],
        sim["tauBeta"],
        r"$\beta$",
        r"$τ_β$",
        cm=cm,
        logx=False,
        logy=True,
    )


def decay_altitude_hist(sim, fig, ax, c):
    ax.hist(sim["altDec"], 100, log=True, color=c)
    ax.set_ylabel("Counts")
    ax.set_xlabel("decay_altitude log(km)")


def num_photo_electrons_hist(sim, fig, ax, c):
    m = sim["numPEs"] != 0
    ax.hist(np.log(sim["numPEs"][m]), 100, log=False, color=c)
    ax.set_ylabel("Counts")
    ax.set_xlabel("log(numPEs)")


def num_photo_electrons_density(sim, fig, ax, cm):
    hexbin(
        fig,
        ax,
        sim["numPEs"],
        np.degrees(sim["beta_rad"]),
        "numPEs",
        "β",
        cm=cm,
        logx=True,
        logy=False,
    )


def num_photo_electrons_altitude(sim, fig, ax, cm):
    hexbin(
        fig,
        ax,
        sim["numPEs"],
        sim["altDec"],
        "numPEs",
        "decay_altitude km",
        cm=cm,
        logx=True,
        logy=True,
    )


def decay_altitude(sim, fig, ax, cm):
    hexbin(
        fig,
        ax,
        np.degrees(sim["beta_rad"]),
        sim["altDec"],
        "β",
        "decay_altitude km",
        cm=cm,
        logx=False,
        logy=True,
    )


def cherenkov_angle_hist(sim, fig, ax, c):
    ax.hist(np.degrees(np.arccos(sim["costhetaChEff"])), 100, log=True, color=c)
    ax.set_ylabel("Counts")
    ax.set_xlabel("θ_chEff")


def eas_input_density(sim, fig, ax, cm):
    hexbin(
        fig,
        ax,
        np.degrees(sim["beta_rad"]),
        sim["altDec"],
        "beta_rad",
        "decay alt km",
        cm=cm,
        logx=False,
        logy=True,
    )


def cherenkov_angle(sim, fig, ax, cm):
    hexbin(
        fig,
        ax,
        np.degrees(np.arccos(sim["costhetaChEff"])),
        np.degrees(sim["beta_rad"]),
        "θ_chEff",
        "β",
        cm=cm,
        logx=False,
        logy=False,
    )


def eas_results_density(sim, fig, ax, cm):
    hexbin(
        fig,
        ax,
        np.degrees(np.arccos(sim["costhetaChEff"])),
        sim["numPEs"],
        "θ_chEff",
        "NumPEs",
        cm=cm,
        logx=False,
        logy=False,
    )


@decorators.ensure_plot_registry(dashboard)
def show_plot(sim, plot, plot_kwargs={}):

    if dashboard.__name__ in plot:
        dashboard(sim, plot_kwargs)

    # dashboard(sim, plot)
