.. _command_line:

========================
Command Line Application
========================

nuspacesim is a command line client shipped with the nuspacesim package. It can be run
in application or module mode.

::

  nuspacesim tui


Opens a TUI to run nuspacesim

::

  nuspacesim --help


Create an XML configuration file.

::

  nuspacesim create-config --help
  nuspacesim create-config my_config_file.xml


Run a simulation.

::

  nuspacesim run --help
  nuspacesim run my_config_file --plotall -w


Plot results from a simulation file.

::

  nuspacesim show_plot -p dashboard simulation_file.fits
