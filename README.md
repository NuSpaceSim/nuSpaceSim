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
 \:\/:/ /:/  / \/\:\  \__  \:\~~\  \/__/
  \::/ /:/  /   ~~\:\/\__\  \:\  \
   \/_/:/  /       \::/  /   \:\  \
     /:/  /        /:/  /     \:\__\
     \/__/         \/__/       \/__/

```

# nuSpaceSim

This is the beta release of the *nuSpaceSim* simulator tool!

This application includes the main scheduler code which calculates the tau
nutrino acceptance for the Optical Cherenkov technique. The configuration
settings for which are controlled by an input XML configuration file. Generating
a new file with input parameters is also supported.

Tau propagation is interpolated using included data tables from Reno et at.
2019.

This package incorporates compiled sub-packages such as nssgeometry and
EAScherGen.

## Requirements

Conda is no longer required to build nuSpaceSim, although it may be desirable
for downloading compilers. All other build and runtime requirements are 
downloaded automatically during setup.

 * python3
 * pip
 * C++11 or higher compiler


## Download & Build

### Direct from GitHub

1. `python3 -m pip install -e git+https://github.com/NuSpaceSim/nuSpaceSim.git#egg=nuSpaceSim`

### Or clone the Repository for development

1. `git clone https://github.com/NuSpaceSim/nuSpaceSim.git`
2. `cd nuSpaceSim`
3. `python3 -m pip install -e .`

## Create an XML configuration script

`nuSpaceSim create-config my_config_file.xml`

## Run simulator

`nuSpaceSim run my_config_file.xml 1000000`

## Uninstall

`python3 -m pip uninstall nuSpaceSim`

