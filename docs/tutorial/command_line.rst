.. _command_line:

========================
Command Line Application
========================

nuspacesim is a command line client shipped with the nuspacesim package. It can be run
in application or module mode.

::

  nuspacesim --help


Create an XML configuration file.

::

  nuspacesim create-config --help
  nuspacesim create-config my_config_file.xml


Run a simulation

::

  nuspacesim run --help
  nuspacesim run my_config_file --plotall -w
