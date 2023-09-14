![NuSpaceSim logo](https://raw.githubusercontent.com/NuSpaceSim/nuSpaceSim/main/docs/_static/NuSpaceSimLogoBlack.png)

# νSpaceSim

[![PyPI](https://img.shields.io/pypi/v/nuspacesim)](https://pypi.org/project/nuspacesim/)
[![Build](https://github.com/NuSpaceSim/nuSpaceSim/actions/workflows/pypi-build-test-publish.yml/badge.svg)](https://github.com/NuSpaceSim/nuSpaceSim/actions/workflows/pypi-build-test-publish.yml)
[![CICD](https://github.com/NuSpaceSim/nuSpaceSim/actions/workflows/pypi-build-test.yml/badge.svg)](https://github.com/NuSpaceSim/nuSpaceSim/actions/workflows/pypi-build-test.yml)
[![Documentation](https://readthedocs.org/projects/nuspacesim/badge/?version=latest)](https://nuspacesim.readthedocs.io/en/latest/?badge=latest)
![PyPI - Downloads](https://img.shields.io/pypi/dm/nuspacesim)


This is the official release of the *nuspacesim* simulator tool!

This package simulates upward-going extensive air showers caused by neutrino
interactions with the atmosphere. It calculates the tau neutrino acceptance for the
Optical Cherenkov technique. The simulation is parameterized by an input XML
configuration file, with settings for detector characteristics and global parameters.
The package also provides a python3 API for programatic access.

Tau propagation is interpolated using included data tables from [nupyprop](https://github.com/NuSpaceSim/nupyprop).

# Installation

`nuspacesim` is available through [pip](https://pypi.org/project/nuspacesim/).

`python3 -m pip install nuspacesim`

<!-- or `conda create -n nuspacesim -c conda-forge -c nuspacesim nuspacesim`-->

# Usage

![NuSpaceSim Usage](https://raw.githubusercontent.com/NuSpaceSim/nuSpaceSim/main/docs/_static/run.svg)

### *nuspacesim* TUI

![NuSpaceSim tui](https://raw.githubusercontent.com/NuSpaceSim/nuSpaceSim/73-tui-option/docs/_static/nuSpaceSim_TUI_example.png)

`nuspacesim tui`

This opens a Text-based user interface (TUI) allowing to explore all CLI options available to *nuspacesim*. On the right hand side are the different *nuspacesim* commands while the left lets youchose and enter all configuration variables. After selecting all options (and leave othern as defaults) the resulting command is shown at the bottom. To execute *Crtl + r*.

### Create an XML configuration script

The command line simulator uses an XML file to store configuration parameters. To
generate a default configuration file run the following, with your choice of file name.

`nuspacesim create-config my_config_file.xml`

### Run simulator

Simulate neutrino interactions and save the results to a named fits file.

`nuspacesim run my_config_file.xml -o my_nss_sim.fits`

# Documentation

The sphinx documentation is available at [ReadTheDocs](https://nuspacesim.readthedocs.io/en/latest/index.html)


### Help Documentation

Use the `--help` flag for documentation.

```
$ nuspacesim --help
Usage: nuspacesim [OPTIONS] COMMAND [ARGS]...

Options:
  --debug / --no-debug
  --help                Show this message and exit.

Commands:
  create-config  Generate a configuration file from the given parameters.
  run            Main Simulator for nuspacesim.
```

Help documentation is also available for the subcommands.

` $  nuspacesim run --help `

### Uninstall

`python3 -m pip uninstall nuspacesim`

# Download & Build

### Clone the Repository (for development)

1. `git clone https://github.com/NuSpaceSim/nuSpaceSim.git`
2. `cd nuSpaceSim`
3. `python3 -m pip install -e .`



```
      ___           ___
     /\  \         /\  \
     \:\  \        \:\  \
      \:\  \        \:\  \
  _____\:\  \   ___  \:\  \
 /::::::::\__\ /\  \  \:\__\
 \:\~~\~~\/__/ \:\  \ /:/  /
  \:\  \        \:\  /:/  /
   \:\  \        \:\/:/  /
    \:\__\        \::/  /
     \/__/         \/__/
      ___           ___         ___           ___           ___
     /\__\         /\  \       /\  \         /\__\         /\__\
    /:/ _/_       /::\  \     /::\  \       /:/  /        /:/ _/_
   /:/ /\  \     /:/\:\__\   /:/\:\  \     /:/  /        /:/ /\__\
  /:/ /::\  \   /:/ /:/  /  /:/ /::\  \   /:/  /  ___   /:/ /:/ _/_
 /:/_/:/\:\__\ /:/_/:/  /  /:/_/:/\:\__\ /:/__/  /\__\ /:/_/:/ /\__\
 \:\/:/ /:/  / \:\/:/  /   \:\/:/  \/__/ \:\  \ /:/  / \:\/:/ /:/  /
  \::/ /:/  /   \::/__/     \::/__/       \:\  /:/  /   \::/_/:/  /
   \/_/:/  /     \:\  \      \:\  \        \:\/:/  /     \:\/:/  /
     /:/  /       \:\__\      \:\__\        \::/  /       \::/  /
     \/__/         \/__/       \/__/         \/__/         \/__/
      ___                       ___
     /\__\                     /\  \
    /:/ _/_       ___         |::\  \
   /:/ /\  \     /\__\        |:|:\  \
  /:/ /::\  \   /:/__/      __|:|\:\  \
 /:/_/:/\:\__\ /::\  \     /::::|_\:\__\                                  .
 \:\/:/ /:/  / \/\:\  \__  \:\~~\  \/__/
  \::/ /:/  /   ~~\:\/\__\  \:\  \                                      /
   \/_/:/  /       \::/  /   \:\  \                                    /
     /:/  /        /:/  /     \:\__\                                  /
     \/__/         \/__/       \/__/                                 /

```
