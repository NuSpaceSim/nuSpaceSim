# The Clear BSD License
#
# Copyright (c) 2021 Alexander Reustle and the NuSpaceSim Team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#      * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""N-Dimensional Gridded Data class with labeled Axes."""

from astropy.io import fits, registry
from astropy.io.misc import hdf5
from astropy.nddata import NDDataArray
from astropy.table import Table as AstropyTable

from typing import Union, Iterable

import numpy as np

__all__ = [
    "NssGrid",
    "NssGridRead",
    "NssGridWrite",
    "fits_nssgrid_reader",
    "fits_nssgrid_writer",
    "hdf5_nssgrid_reader",
    "hdf5_nssgrid_writer",
]


class NssGridRead(registry.UnifiedReadWrite):
    def __init__(self, instance, cls):
        super().__init__(instance, cls, "read")

    def __call__(self, *args, **kwargs):
        return registry.read(self._cls, *args, **kwargs)


class NssGridWrite(registry.UnifiedReadWrite):
    def __init__(self, instance, cls):
        super().__init__(instance, cls, "write")

    def __call__(self, *args, **kwargs):
        registry.write(self._instance, *args, **kwargs)


class NssGrid(NDDataArray):
    r"""Multidimensional Gridded data object with support for axes."""

    def __init__(self, data, axes, axis_names, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

        if len(axes) != len(axis_names):
            raise ValueError("Must give same number of names as axes.")

        if len(axes) != data.ndim:
            raise ValueError(
                f"Must give same number of axes({axes}) as grid dimensions ({data.ndim})."
            )

        for a in axes:
            if a.ndim != 1:
                raise ValueError(f"Each axis must be 1D. Got {a.ndim} from {a.shape}.")

        for i in range(data.ndim):
            if axes[i].shape[0] != data.shape[i]:
                raise ValueError("Axes lengths must correspond to grid dimensions.")

        self.axes = axes

        self.axis_names = axis_names

        self.meta = {**{f"AXIS{i}": n for i, n in enumerate(self.axis_names)}}

    read = registry.UnifiedReadWriteMethod(NssGridRead)
    write = registry.UnifiedReadWriteMethod(NssGridWrite)

    def __repr__(self):
        rep = "NssGrid {\n"
        rep += f"meta : {repr(self.meta)}\n"
        for axis, name in zip(self.axes, self.axis_names):
            rep += f"{name} ({len(axis)}): {axis}\n"
        rep += f"data: {repr(super().data)}\n"
        rep += f"dims: {super().data.shape}\n"
        rep += "}"
        return rep

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other):
        return (
            np.array_equal(self.data, other.data)
            and self.axis_names == other.axis_names
            and np.all([np.array_equal(s, o) for s, o in zip(self.axes, other.axes)])
        )

    def __getitem__(self, item):
        # Abort slicing if the data is a single scalar.
        if self.data.shape == ():
            raise TypeError("scalars cannot be sliced.")

        # Let the other methods handle slicing.
        kwargs = self._slice(item)
        return self.__class__(**kwargs)

    def _slice(self, item):
        """Collects the sliced attributes and passes them back as `dict`.

        Parameters
        ----------
        item : slice
            The slice passed to ``__getitem__``.

        Returns
        -------
        dict :
            Containing all the attributes after slicing - ready to
            use them to create ``self.__class__.__init__(**kwargs)`` in
            ``__getitem__``.
        """
        kwargs = {}
        kwargs["data"] = self.data[item]
        # Try to slice some attributes
        kwargs["axes"] = self._slice_axes(item)
        kwargs["axis_names"] = self._slice_axis_name(item)
        return kwargs

    def _slice_axes(self, item):
        if self.axes is None:
            return None
        try:
            return [self.axes[i][s] for i, s in enumerate(item) if isinstance(s, slice)]
        except TypeError:
            # Catching TypeError in case the object has no __getitem__ method.
            # But let IndexError raise.
            raise RuntimeWarning("Axes object has no __getitem__.")

    def _slice_axis_name(self, item):
        if self.axis_names is None:
            return None
        try:
            return [
                self.axis_names[i] for i, s in enumerate(item) if isinstance(s, slice)
            ]
        except TypeError:
            pass
        return self.axis_names

    def index_name(self, name):
        r"""Locate the index of axis ``name``."""
        if name not in self.axis_names:
            raise ValueError(f"name: {name} not in axis_names {self.axis_names}")
        return self.axis_names.index(name)

    def index_where_eq(self, name, value):
        r"""Locate the index in axis ``name`` where element is equal to ``value``."""
        ax = self.index_name(name)
        if value not in self.axes[ax]:
            raise ValueError(
                f" {value} not in {self.axis_names[ax]} axis: {self.axes[ax]}"
            )
        idx = np.where(self.axes[ax] == value)[0][0]
        return idx

    def slc(self, axis_name, axis_val, axis_index):
        pass
        none_list = [None] * self.ndim
        d_slice = slice(*none_list)
        g_slice = [d_slice] * self.ndim
        g_slice[axis_index] = self.index_where_eq(axis_name, axis_val)
        return self[tuple(g_slice)]


class NssAxes:
    r"""Collection of differently sized, named 1D arrays"""

    def __init__(
        self, values: Union[float, Iterable], names: Union[str, Iterable[str]]
    ):

        if not isinstance(values, Iterable):
            values = list([values])

        if not isinstance(names, Iterable):
            names = [names]

        self.values: list[float] = list(values)
        self.names: list[str] = list(names)

    def __getitem__(self, item):

        if isinstance(item, str):
            if item in self.names:
                i = self.names.index(item)
                return self.values[i]
            else:
                raise ValueError(f"{item} not found in names.")

        if isinstance(item, int):
            if item < len(self.values):
                return self.values[item]
            else:
                raise IndexError(f"Array index ({item}) out of bounds.")


def fits_nssgrid_reader(filename, **kwargs):

    with fits.open(filename, **kwargs) as f:
        naxis = f[0].header["NAXIS"]
        axis_names = [f[0].header[f"AXIS{i}"] for i in range(naxis)]
        griddata = f[0].data
        axes = [f[i + 1].data.field(0) for i in range(naxis)]

    return NssGrid(griddata, axes=axes, axis_names=axis_names)


def fits_nssgrid_writer(grid, filename, **kwargs):

    primary = fits.PrimaryHDU(grid.data, fits.Header(grid.meta))
    primary.add_checksum()

    hdus = [
        fits.BinTableHDU(
            AstropyTable([axis], names=[name], meta={"AXIS": name}), name=name
        )
        for axis, name in zip(grid.axes, grid.axis_names)
    ]

    for h in hdus:
        h.add_checksum()

    hdu = fits.HDUList([primary, *hdus])
    hdu.writeto(filename, **kwargs)


def hdf5_nssgrid_reader(filename, path="/"):
    try:
        import h5py
    except ImportError:
        raise Exception("h5py is required to read and write HDF5 files")
    pass

    with h5py.File(filename, "r") as f:
        griddata = f[path]["__nss_grid_data__"][()]
        axis_names = [f[path].attrs[f"AXIS{i}"] for i in range(griddata.ndim)]
        axes = [f[path]["__nss_grid_axes__"][name][()] for name in axis_names]

    return NssGrid(griddata, axes=axes, axis_names=axis_names)


def hdf5_nssgrid_writer(grid, filename, path="/", **kwargs):
    try:
        import h5py
    except ImportError:
        raise Exception("h5py is required to read and write HDF5 files")

    if "overwrite" in kwargs.keys():
        if kwargs["overwrite"] == False:
            mode = "w-"
        else:
            mode = "w"
    else:
        mode = "a"

    with h5py.File(filename, mode) as f:

        grp = f[path] if path in f else f.create_group(path)
        grp.create_dataset("__nss_grid_data__", shape=grid.shape, data=grid.data)
        ax = grp.create_group("__nss_grid_axes__")
        for axis, name in zip(grid.axes, grid.axis_names):
            ax.create_dataset(name, data=axis)

        for k, v in grid.meta.items():
            grp.attrs[k] = v


with registry.delay_doc_updates(NssGrid):
    registry.register_reader("fits", NssGrid, fits_nssgrid_reader)
    registry.register_writer("fits", NssGrid, fits_nssgrid_writer)
    registry.register_identifier("fits", NssGrid, fits.connect.is_fits)
    registry.register_reader("hdf5", NssGrid, hdf5_nssgrid_reader)
    registry.register_writer("hdf5", NssGrid, hdf5_nssgrid_writer)
    registry.register_identifier("hdf5", NssGrid, hdf5.is_hdf5)
