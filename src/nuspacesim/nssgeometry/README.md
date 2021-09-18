# NuSpaceSim Geometry Module (nssgeometry)

A python extension module that contains the "Geom_params" class that defines the geometry of the observation region for the detector. The "Geom_params" class includes an "Event" class that contains the geometrical parameters for generated events and a function for generating events and their trajectories.

## Requirements

To build this module you require:

* CMake version >= 3.12
* Conda - python3
* pybind11 -- for right now, install through anaconda using: conda install -c conda-forge pybind11
* HDF5 c and c++ libraries installed -- conda install -c anaconda hdf5

* pytest

## Build

1. git clone https://github.com/NuSpaceSim/nssgeometry.git
2. pip install -e .

## Test

pytest -v tests

## Uninstall

pip uninstall nssgeometry
