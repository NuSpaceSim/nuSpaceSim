.. _quickstart_ref:

==========
Quickstart
==========

------------
Installation
------------

Install nuspacesim from pypi. This will install needed dependencies along with
nuspacesim.  ::

  python3 -m pip install nuspacesim

Alternatively install nuspacesim from our conda channel.::

  conda create -n nuspacesim -c conda-forge -c nuspacesim nuspacesim
  conda activate nuspacesim


----------------------------------
Read the nuspacesim help docstring
----------------------------------
::

  python3 -m nuspacesim --help

----------------------------------
Create the XML configuration file
----------------------------------

Create a configuration file with the ``create-config`` command. This is editable by the
user for defining different simulation parameters.  ::

  nuspacesim create-config --numtrajs 1e6 --monospectrum 10.25 my_config_file.xml

-----------------
Run the simulator
-----------------

Simulate neutrino interactions, and extensive air showers, then save the results to a
FITS file.  ::

  nuspacesim run my_config_file.xml -output my_nss_sim.fits

Optionally, override the configuration file on the command line.  ::

  nuspacesim run my_config_file.xml 1e5 --powerspectrum 2 6 12 -o my_nss_sim.fits


.. raw:: html

    <script src="https://asciinema.org/a/i4gGSkDlQqkYn73AbBxX05lfu.js" id="asciicast-i4gGSkDlQqkYn73AbBxX05lfu" async data-autoplay="true"></script>


--------------------------
Explore simulation results
--------------------------

Quickly overview the generated table with astropy's `showtable`.

::

  showtable my_nss_sim.fits


------------------
Help Documentation
------------------

Use the --help flag for documentation.::


  nuspacesim --help
  Usage: nuspacesim [OPTIONS] COMMAND [ARGS]...

  Options:
    --debug / --no-debug
    --help                Show this message and exit.

  Commands:
    create-config  Generate a configuration file from the given parameters.
    run            Main Simulator for nuspacesim.


Also works for the subcommands.

::

  nuspacesim run --help

---------
Uninstall
---------

``python3 -m pip uninstall nuspacesim``

--------------------------------------
Clone the Repository (for development)
--------------------------------------

1. ``git clone https://github.com/NuSpaceSim/nuSpaceSim.git``
2. ``cd nuSpaceSim``
3. ``python3 -m pip install -e .``
