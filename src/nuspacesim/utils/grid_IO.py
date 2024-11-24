"""N-Dimensional Gridded Data class with labeled Axes.

.. autosummary::
   :toctree:
   :recursive:

    Grid
    GridRead
    GridWrite
    grid_concatenate
    fits_grid_reader
    fits_grid_writer
    hdf5_grid_reader
    hdf5_grid_writer

"""

from typing import Any, Iterable, Union

import astropy.units as u
import numpy as np
from astropy.io import fits, registry
from astropy.io.misc import hdf5
from astropy.nddata import NDDataArray
from astropy.table import Table as AstropyTable
from astropy.time import Time

__all__ = [
    "Grid",
    "GridRead",
    "GridWrite",
    "grid_concatenate",
    "fits_grid_reader",
    "fits_grid_writer",
    "hdf5_grid_reader",
    "hdf5_grid_writer",
]


class GridRead(registry.UnifiedReadWrite):
    def __init__(self, instance, cls):
        super().__init__(instance, cls, "read")

    def __call__(self, *args, **kwargs):
        return registry.read(self._cls, *args, **kwargs)


class GridWrite(registry.UnifiedReadWrite):
    def __init__(self, instance, cls):
        super().__init__(instance, cls, "write")

    def __call__(self, *args, **kwargs):
        registry.write(self._instance, *args, **kwargs)


def build_data(data, unit):
    if unit is None:
        return data
    elif unit == "time":
        return Time(data, format="gps")
    else:
        return u.Quantity(data, unit)


class Axes(NDDataArray):
    r"""Collection of equal sized multi dimensional arrays with named axes."""

    def __init__(self, data, names, *args, **kwargs):
        data, units = self.prep_data(data)

        super().__init__(data, *args, **kwargs)

        self.units: list = list(units)
        self.names: list = list(names)

        self.meta: dict = {}
        for i, n in enumerate(names):
            self.meta[f"A{i}"] = n
            self.meta[f"U{i}"] = units[i]

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.names:
                i = self.names.index(item)
                return build_data(super().data[i], self.units[i])
            else:
                raise ValueError(f"{item} not in names.")

        if isinstance(item, int):
            if item < len(super().data):
                return build_data(super().data[i], self.units[i])
            else:
                raise IndexError(f"Array index ({item}) out of bounds.")

    def prep_data(self, data):
        units = [None] * len(data)
        for i, d in enumerate(data):
            if isinstance(d, u.Quantity):
                units[i] = str(d.unit)
                data[i] = d.value

            if isinstance(d, Time):
                units[i] = "time"
                data[i] = d.gps

        data = np.array(data)
        return data, units

    def __repr__(self):
        rep = "Axes {\n"
        rep += f"meta : {sorted(self.meta.items())}\n"
        for i in range(len(self.names)):
            rep += f"{self.names[i]} {super().data[i].shape}: {build_data(super().data[i], self.units[i])}\n"

        rep += "}"

        return rep

    def __str__(self) -> str:
        return repr(self)


class Grid(NDDataArray):
    def __init__(self, axes, axis_names, *args, **kwargs):
        super().__init__([0], *args, **kwargs)
        if len(axes) != len(axis_names):
            raise ValueError("Must give same number of names as axes.")

        self.axes = axes
        """
        N element list with 1D arrays of different length, corresponding to data
        array dimension size.
        """
        self.names = axis_names

        self.meta: dict = {}
        for i, n in enumerate(self.names):
            self.meta[f"AXES{i}"] = n
            for key, value in self.axes[i].meta.items():
                self.meta[f"I{i}{key}"] = value
        """
        scalar value metadata dictionary.
        """

    read = registry.UnifiedReadWriteMethod(GridRead)
    """
    astropy UnifiedReadWriteMethod for reading files into Grid objects.
    """

    write = registry.UnifiedReadWriteMethod(GridWrite)
    """
    astropy UnifiedReadWriteMethod for writing Grid objects to files.
    """

    def __repr__(self):
        rep = "Grid {\n"
        rep += f"meta : {sorted(self.meta.items())}\n"

        for axis, name in zip(self.axes, self.names):
            rep += f"{name} ({axis.shape}): {str(axis)}\n"

        rep += "}"

        return rep

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other):
        return (
            np.array_equal(self.data, other.data)
            and self.names == other.names
            and np.all([np.array_equal(s, o) for s, o in zip(self.axes, other.axes)])
        )

    def __getitem__(self, item) -> Union[np.ndarray, Any]:
        # Abort slicing if the data is a single scalar.
        if self.data.shape == ():
            raise TypeError("scalars cannot be sliced.")

        if isinstance(item, int):
            return self.axes[item]

        if isinstance(item, str):
            if item in self.names:
                i = self.names.index(item)
                return self.axes[i]
            else:
                raise ValueError(f"{item} not in names.")

        # Let the other methods handle slicing.
        kwargs = self._slice(item)
        return self.__class__(**kwargs)

    def _slice(self, item):
        kwargs = {}
        kwargs["data"] = self.data[item].squeeze()
        ax = self._slice_axes(item)
        kwargs["axes"] = ax.data
        kwargs["axis_names"] = ax.names
        return kwargs

    def _slice_axes(self, item):
        if self.axes is None:
            return Axes(None, None)
        try:
            v = [
                self.axes[i][s]
                for i, s in enumerate(item)
                if np.count_nonzero(s) > 1 or isinstance(s, slice)
            ]
            n = [
                self.names[i]
                for i, s in enumerate(item)
                if np.count_nonzero(s) > 1 or isinstance(s, slice)
            ]
            return Axes(v, n)
        except TypeError:
            # Catching TypeError in case the object has no __getitem__ method.
            # But let IndexError raise.
            raise RuntimeWarning("Axes object has no __getitem__.")


def grid_concatenate(g1, g2, axis):
    # grids have same ndim
    assert g1.ndim == g2.ndim
    # grids have same axis names
    assert g1.axis_names
    # grids have same shape everywhere except the axis dimension
    assert (
        g1.shape[:axis] + g1.shape[axis + 1 :] == g2.shape[:axis] + g2.shape[axis + 1 :]
    )

    new_axes = g1.axes
    new_axes[axis] = np.concatenate((g1.axes[axis], g2.axes[axis]))
    new_data = np.concatenate((g1.data, g2.data), axis=axis)

    return Grid(new_data, new_axes, g1.axis_names)


def fits_grid_reader(filename, **kwargs):
    with fits.open(filename, **kwargs) as f:
        keys = list(f[0].header.keys())

        axes = []
        n_axes = len([k for k in keys if k.startswith("AXES")])
        axis_names = [f[0].header[k] for k in keys if k.startswith("AXES")]
        for i in range(n_axes):
            axis_keys = [k for k in keys if k.startswith(f"I{i}")]
            names = [f[0].header[k] for k in axis_keys if k.startswith(f"I{i}A")]
            units = [f[0].header[k] for k in axis_keys if k.startswith(f"I{i}U")]

            data = []
            for k in range(len(names)):
                data.append(build_data(f[i + 1].data.field(0)[k], units[k]))

            axes.append(Axes(data, names))

    return Grid(axes=axes, axis_names=axis_names)


def fits_grid_writer(grid, filename, **kwargs):
    primary = fits.PrimaryHDU(grid.data, fits.Header(grid.meta))
    primary.add_checksum()

    hdus = [
        fits.BinTableHDU(
            AstropyTable([axis], names=[name], meta={"AXIS": name}), name=name
        )
        for axis, name in zip(grid.axes, grid.names)
    ]

    for h in hdus:
        h.add_checksum()

    hdu = fits.HDUList([primary, *hdus])
    hdu.writeto(filename, **kwargs)


def hdf5_grid_reader(filename, path="/"):
    try:
        import h5py
    except ImportError:
        raise Exception("h5py is required to read and write HDF5 files")
    pass

    with h5py.File(filename, "r") as f:
        keys = list(f[path].attrs.keys())
        axes = []
        n_axes = len([k for k in keys if k.startswith("AXES")])
        axis_names = [f[path].attrs[k] for k in keys if k.startswith("AXES")]
        for i in range(n_axes):
            axis_keys = [k for k in keys if k.startswith(f"I{i}")]
            names = [f[path].attrs[k] for k in axis_keys if k.startswith(f"I{i}A")]
            units = [f[path].attrs[k] for k in axis_keys if k.startswith(f"I{i}U")]
            data = []
            for k in range(len(names)):
                data.append(
                    build_data(f[path]["__grid_axes__"][axis_names[i]][()][k], units[k])
                )

            axes.append(Axes(data, names))

    return Grid(axes=axes, axis_names=axis_names)


def hdf5_grid_writer(grid, filename, path="/", **kwargs):
    try:
        import h5py
    except ImportError:
        raise Exception("h5py is required to read and write HDF5 files")

    if "overwrite" in kwargs.keys():
        if not kwargs["overwrite"]:
            mode = "w-"
        else:
            mode = "a"
    else:
        kwargs["overwrite"] = False
        mode = "a"

    with h5py.File(filename, mode) as f:
        grp = f[path] if path in f else f.create_group(path)

        if kwargs["overwrite"] and "__grid_axes__" in grp:
            del grp["__grid_axes__"]
        ax = grp.create_group("__grid_axes__")

        for axis, name in zip(grid.axes, grid.names):
            ax.create_dataset(name, data=axis)

        for k, v in grid.meta.items():
            grp.attrs[k] = v


registry.register_reader("fits", Grid, fits_grid_reader)
registry.register_writer("fits", Grid, fits_grid_writer)
registry.register_identifier("fits", Grid, fits.connect.is_fits)
registry.register_reader("hdf5", Grid, hdf5_grid_reader)
registry.register_writer("hdf5", Grid, hdf5_grid_writer)
registry.register_identifier("hdf5", Grid, hdf5.is_hdf5)


def main():
    x = np.linspace(0, 10, 100) * u.deg
    y = np.linspace(10, 20, 100) * u.deg
    z = np.linspace(20, 30, 100) * u.m

    start_time = Time("2021-01-01T00:00:00", format="isot", scale="utc")
    times = (
        start_time
        + np.linspace(
            0,
            800,
            100,
        )
        * u.min
    )

    axes = Axes(
        data=[times, x, y, z],
        names=["times", "x", "y", "z"],
    )

    axes2 = Axes(
        data=[x, y],
        names=["x", "y"],
    )

    grid = Grid(
        [axes, axes2, axes], axis_names=["detector_loc", "source_loc", "source_loc2"]
    )

    # os.remove("test.fits")
    import os

    if os.path.exists("test.h5"):
        os.remove("test.h5")
    grid.write("test.h5", overwrite=True)

    grid2 = Grid.read("test.h5")
    print(grid2)
    quit()
    print(grid2["detector_loc"]["times"].isot)


if __name__ == "__main__":
    main()
