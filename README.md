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
 /:/_/:/\:\__\ /::\  \     /::::|_\:\__\
 \:\/:/ /:/  / \/\:\  \__  \:\~~\  \/__/                                 .
  \::/ /:/  /   ~~\:\/\__\  \:\  \                                      /
   \/_/:/  /       \::/  /   \:\  \                                    /
     /:/  /        /:/  /     \:\__\                                  /
     \/__/         \/__/       \/__/                                 /

```

# νSpaceSim

This is the beta release of the *nuSpaceSim* simulator tool!

This package simulates upward-going electromagnetic air showers caused by neutrino
interactions with the atmosphere. It calculates the tau neutrino acceptance for the
Optical Cherenkov technique. The simulation is parameterized by an input XML
configuration file, with settings for detector characteristics and global parameters.
The package also provides a python3 API for programatic access.

Tau propagation is interpolated using included data tables from Reno et at.
2019.

This package incorporates compiled sub-packages such as nssgeometry and
EAScherGen.


# Download & Build

### Requirements

Conda is no longer required to build nuSpaceSim, although it may be desirable
for downloading compilers. All other build and runtime requirements are 
downloaded automatically during setup.

 * python3
 * pip
 * C++11 (or higher) compiler


### Clone the Repository (for development)

1. `git clone https://github.com/NuSpaceSim/nuSpaceSim.git`
2. `cd nuSpaceSim`
3. `python3 -m pip install -e .`

# Usage

### Create an XML configuration script

`python -m nuspacesim create-config my_config_file.xml`

### Run simulator

Simulate neutrino interactions and save the results to a fits file.

`nuspacesim run my_config_file.xml 1000 8.0 -o my_nss_sim.fits`

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

Also works for the subcommands.

` $  nuspacesim run --help `

### Uninstall

`python3 -m pip uninstall nuspacesim`

