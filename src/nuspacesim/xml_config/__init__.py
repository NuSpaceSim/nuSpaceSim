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

r"""XML file utilities for configuration objects

--------------------------------------
NuSpaceSim Configuration XML Interface
--------------------------------------

Configuration object can be serialized to an XML file, in whole or in part. These
configuration files can be read back into nuspacesim with
:func:`config_from_xml<nuspacesim.xml_config.config_from_xml>`. All XML config files
are validated with an XSD Schema to by
is_valid_xml ensure correctness.
:func:`is_valid_xml<nuspacesim.xml_config.is_valid_xml>` ensure correctness.

.. autosummary::
   :toctree:
   :nosignatures:

   create_xml
   config_from_xml
   is_valid_xml
   parseXML
   parse_detector_chars
   parse_simulation_params
   config_xml_schema

"""

__all__ = [
    "is_valid_xml",
    "parse_config",
    "parse_detector_chars",
    "parse_simulation_params",
    "parseXML",
    "config_from_xml",
    "create_xml",
]

from . import parse_config
from .parse_config import (
    config_from_xml,
    create_xml,
    is_valid_xml,
    parse_detector_chars,
    parse_simulation_params,
    parseXML,
)
