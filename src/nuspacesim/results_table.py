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

    ResultsTable

"""

import datetime
from typing import Any, Iterable, Union

from astropy.table import Table as AstropyTable
from numpy.typing import ArrayLike

from .config import NssConfig

__all__ = ["ResultsTable"]


class ResultsTable(AstropyTable):
    r"""Results of NuSpaceSim simulation stages.

    ResultsTable inherits from astropy.table.Table and uses that implementation to manage
    Result data columns. This enables easy serialization of simulation results to
    output file formats.
    """

    def __init__(self, config=None):
        r"""Constructor for ResultsTable class instances.

        Parameters
        ----------
        config: NssConfig
            Configuration object. Used to initialize result metadata.
        """

        if config is None:
            config = NssConfig()

        if isinstance(config, NssConfig):
            now = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
            super().__init__(
                meta={
                    **config.detector(),
                    **config.simulation(),
                    **config.constants(),
                    "simTime": (now, "Start time of Simulation"),
                }
            )
        elif isinstance(config, AstropyTable):
            super().__init__(config)

    def __call__(self, col_names: Iterable[str], columns: Iterable[ArrayLike]) -> None:
        r"""Add named columns to the simulation results.

        Insert data into the result table, with the names corresponding to the
        values in col_names.

        Parameters
        ----------
        col_names: Iterable[str]
            List of column names.
        columns: Iterable[ArrayLike]
            List of column
        """

        self.add_columns(columns, names=col_names)

    def add_meta(self, name: str, value: Any, comment: str) -> None:
        r"""Add metadata attributes to the simulation results.

        Insert a named attribute into the table metadata store, with a descriptive
        comment.

        Parameters
        ----------
        name: str
            Attribute name keyword.
        value: Any
            Attribute scalar value.
        comment: str
            Attribute descriptive comments.
        """

        self.meta[name] = (value, comment)


class ResultsWriter:
    def __init__(self, filename, **kwargs):
        """
        The filename of the output file. If None, return default with timestamp."""

        now = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
        self.filename = f"nuspacesim_run_{now}.fits" if filename is None else filename

        self.kwargs = kwargs
        self.kwargs["format"] = "fits" if "format" not in kwargs else kwargs["format"]

        if self.kwargs["format"] not in ["fits", "hdf5"]:
            raise ValueError(
                f"File output format {self.kwargs['format']} not in {{ fits, hdf5 }}!"
            )

        if self.kwargs["format"] == "hdf5":
            self.kwargs["path"] = "/" if "path" not in kwargs else kwargs["path"]
            self.kwargs["overwrite"] = (
                True if "overwrite" not in kwargs else kwargs["overwrite"]
            )
            self.kwargs["serialize_meta"] = True

    def __call__(self, *tables):
        r"""Write the simulation results to a Fits file.

        Uses the astropy.table.Table write method of the ResultsTable base class to
        populate the contents of a FITS file with 1 or more tables.

        Parameters
        ----------
        tables: {ResultsTable}
        """

        for table in tables:
            table.super().write(self.filename, **self.kwargs)
