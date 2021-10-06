.. _configuration:

********************************
NuSpaceSim Configuration Objects
********************************


NuSpaceSim provides a structured means of defining configuration parameters for
your simulations.

.. autosummary::
   :toctree: generated/
   :recursive:
   :nosignatures:

   nuspacesim.NssConfig
   nuspacesim.DetectorCharacteristics
   nuspacesim.SimulationParameters


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
   :toctree: generated/
   :recursive:
   :nosignatures:

   nuspacesim.xml_config.create_xml
   nuspacesim.xml_config.config_from_xml
   nuspacesim.xml_config.is_valid_xml
   nuspacesim.xml_config.parseXML
   nuspacesim.xml_config.parse_detector_chars
   nuspacesim.xml_config.parse_simulation_params
   nuspacesim.xml_config.config_xml_schema
