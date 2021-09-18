"""
storage.py: nuspacesim module for storing simulation intermediates in HDF5
Files.

    Design requirements:
    1 table in 1 file per simulation.
"""

from typing import Union
from nuspacesim.simulate import Simulation

__all__ = ["write_fits", "write_hdf5"]


def write_fits(simulation: Simulation, filename: Union[str, None] = None) -> None:
    simulation.write(
        f"nuspacesim_run_{simulation.meta['simTime'][0]}.fits"
        if filename is None
        else filename,
        format="fits",
    )


def write_hdf5(simulation: Simulation, filename: Union[str, None] = None) -> None:
    filename = (
        f"nuspacesim_run_{simulation.meta['simTime'][0]}.h5"
        if filename is None
        else filename
    )
    simulation.write(filename, format="hdf5", path=filename, serialize_meta=True)
