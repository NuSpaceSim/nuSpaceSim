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

r"""NuSpaceSim - Simulate upward-going neutrino showers, interactions, and detections.

Python package, library, and data tables.

************************
Command Line Application
************************

Command line client application

.. autosummary::
   :toctree:
   :recursive:

   nuspacesim.apps


******************
Simulation Modules
******************

Scientific modules for performing specific stages of the simulation.

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:

   nuspacesim.simulation


*********************
Configuration Objects
*********************

NuSpaceSim provides a structured means of defining configuration parameters for
your simulations.

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:

   nuspacesim.NssConfig
   nuspacesim.DetectorCharacteristics
   nuspacesim.SimulationParameters
   nuspacesim.xml_config

*****************
Simulate Function
*****************

Perform a full simulation by calling the default nuspacesim.simulate function.

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:

   nuspacesim.compute


**********************
Data Interface Objects
**********************

Classes and Methods for storing and interacting with nuspacesim data.

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:

   nuspacesim.results_table
   nuspacesim.utils.grid


******************
Data lookup tables
******************

Supporting data files such as the nupyprop tau energy CDFs and exit probability tables.

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:

   nuspacesim.data


*****************
Utility Functions
*****************

Supporting classes and functions for gridded data, CDF sampling, Interpolation, etc.

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:

   nuspacesim.utils


"""

from . import constants, data, utils, xml_config
from ._version import version, version_tuple
from .compute import compute
from .config import DetectorCharacteristics, NssConfig, SimulationParameters
from .results_table import ResultsTable
from .simulation import eas_optical, geometry, taus

__all__ = [
    # Core
    "constants",
    "NssConfig",
    "DetectorCharacteristics",
    "SimulationParameters",
    "ResultsTable",
    "compute",
    # modules
    "geometry",
    "eas_optical",
    "taus",
    # other
    "data",
    "utils",
    "xml_config",
    # version
    "version",
    "version_tuple",
]
