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


def geom_beta_tr_hist(inputs, results, fig, ax, *_, **kwargs):
    r"""Plot a histgram of beta trajectories."""

    betas, _, _ = results
    ax.hist(
        x=np.degrees(betas),
        bins=np.linspace(min(np.degrees(betas)), max(np.degrees(betas)), 50),
        color=kwargs["color"][0],
    )
    make_labels(
        fig=fig,
        ax=ax,
        xlabel="Earth emergence angle $\\beta$ / $^{\\circ}$",
        ylabel="Counts",
    )


def path_length_to_detector(inputs, results, fig, ax, *_, **kwargs):
    r"""Plot a histgram of beta trajectories."""

    _ = inputs
    betas, _, path_lens = results
    binning_b = np.arange(
        np.min(np.degrees(betas)) - 1, np.max(np.degrees(betas)) + 2, 1
    )
    im = hexbin(
        ax,
        x=np.degrees(betas),
        y=path_lens,
        gs=len(binning_b),
        cmap=kwargs["cmap"],
    )
    make_labels(
        fig,
        ax,
        "Earth emergence angle $\\beta$ / $^{\\circ}$",
        "Path length to detector / km",
        clabel="Counts",
        im=im,
    )
