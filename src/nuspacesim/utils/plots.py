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

__all__ = ["make_labels", "get_profile", "hexbin", "hist2d"]


def make_labels(
    fig,
    ax,
    xlabel,
    ylabel,
    clabel=None,
    im=None,
    logx=False,
    logy=False,
    logx_scale=False,
    logy_scale=False,
):

    xl = "$\\log_{10}$" + f"({xlabel})" if logx else xlabel
    yl = "$\\log_{10}$" + f"({ylabel})" if logy else ylabel
    xs = "log" if logx_scale else "linear"
    ys = "log" if logy_scale else "linear"

    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_xscale(xs)
    ax.set_yscale(ys)
    if clabel is not None:
        cbar = fig.colorbar(im, ax=ax, pad=0.0)
        cbar.set_label(clabel)


def get_profile(x, y, nbins, useStd=True):

    if sum(np.isnan(y)) > 0:
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


def hexbin(
    ax,
    x,
    y,
    gs=25,
    logx=False,
    logy=False,
    logx_scale=False,
    logy_scale=False,
    cmap=None,
):

    xf = np.log10 if logx else lambda q: q
    yf = np.log10 if logy else lambda q: q

    xs = "log" if logx_scale else "linear"
    ys = "log" if logy_scale else "linear"

    xmask = x > 0 if logx else np.full(x.shape, True)
    ymask = y > 0 if logy else np.full(y.shape, True)
    m = xmask & ymask

    im = ax.hexbin(
        x=xf(x[m]),
        y=yf(y[m]),
        gridsize=gs,
        mincnt=1,
        xscale=xs,
        yscale=ys,
        cmap=cmap,
        edgecolors="none",
    )
    return im


def hist2d(fig, ax, x, y, xlab, ylab, cmap="jet", logx=True, logy=True):

    xf = np.log10 if logx else lambda q: q
    yf = np.log10 if logy else lambda q: q

    xl = f"log({xlab})" if logx else xlab
    yl = f"log({ylab})" if logy else ylab

    xmask = x > 0 if logx else np.full(x.shape, True)
    ymask = y > 0 if logy else np.full(y.shape, True)
    m = xmask & ymask

    _, _, _, im = ax.hist2d(x=xf(x[m]), y=yf(y[m]), bins=(50, 50), cmin=1, cmap=cmap)

    ax.set_xlabel(xl)
    ax.set_ylabel(yl)

    ax.set_title(f"{yl} vs {xl}")
    cbar = fig.colorbar(im, ax=ax, pad=0.0)
    cbar.set_label("Counts")
