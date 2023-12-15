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

"""Simulation Results class and full simulation main function.

.. autosummary::
   :toctree:
   :recursive:

"""

from __future__ import annotations

import datetime

from astropy.table import Table as AstropyTable

from .config import NssConfig
from .utils.misc import flatten_dict

__all__ = ["init", "output_filename"]


def init(config: NssConfig | None = None):
    r"""Initialize a simulation results table with metadata"""
    if config is None:
        config = NssConfig()

    if isinstance(config, NssConfig):
        now = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
        return AstropyTable(
            meta={
                "simTime": (now, "Start time of Simulation"),
                **flatten_dict(config.model_dump(), "HIERARCH Config", sep=" "),
            }
        )
    elif isinstance(config, AstropyTable):
        return AstropyTable(config)

    else:
        return AstropyTable()


def output_filename(filename: str | None, now: str | None = None) -> str:
    if filename is not None:
        return filename
    now = now if now else f"{datetime.datetime.now():%Y%m%d%H%M%S}"
    return f"nuspacesim_run_{now}.fits"
