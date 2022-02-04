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


def geom_beta_tr_hist(inputs, results, *args, **kwargs):
    r"""Plot a histgram of beta trajectories."""

    _ = inputs
    betas, _, _ = results
    plotting_opts = kwargs.get("kwargs")
    if "default_color" in plotting_opts:
        c = "C{}".format(plotting_opts.get("default_color"))
    else:
        c = "C0"

    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    plt.hist(np.degrees(betas), 50, color=c)
    plt.xlabel("beta_tr (radians)")
    plt.ylabel("frequency (counts)")
    plt.title(f"Histogram of {betas.size} Beta Angles")
    if "pop_up" not in plotting_opts:
        plt.show()
    elif "pop_up" in plotting_opts and plotting_opts.get("pop_up") is True:
        plt.show()
    if plotting_opts.get("save_to_file") is True:
        fig.savefig(
            plotting_opts.get("filename")
            + "_geom_beta_tr_hist."
            + plotting_opts.get("save_as")
        )
