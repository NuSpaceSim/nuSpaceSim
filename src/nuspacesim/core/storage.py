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

"""nuspacesim module for storing simulation results in diverse file formats."""

from typing import Union
from nuspacesim.core import Simulation

__all__ = ["write_fits", "write_hdf5"]


def write_fits(simulation: Simulation, filename: Union[str, None] = None) -> None:
    r"""Write the simulation results to a FITS file.

    Uses the astropy.table.Table write method of the Simulation base class to write
    FITS file.

    Parameters
    ----------
    simulation: Simulation
        The simulation results object.
    filename, {str, None}, optional
        The filename of the output file. If None, return default with timestamp.
    """
    simulation.write(filename, format="fits")


def write_hdf5(simulation: Simulation, filename: Union[str, None] = None) -> None:
    r"""Write the simulation results to an HDF5 file.

    Uses the astropy.table.Table write method of the Simulation base class to write
    HDF5 file.

    Parameters
    ----------
    simulation: Simulation
        The simulation results object.
    filename, {str, None}, optional
        The filename of the output file. If None, return default with timestamp.
    """
    simulation.write(filename, format="hdf5")
