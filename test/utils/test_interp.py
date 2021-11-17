import numpy as np
from nuspacesim.utils.grid import NssGrid
from nuspacesim.utils.interp import grid_slice_interp


def test_grid_interp():
    data = np.arange(8 * 16).reshape(8, 16)
    x = np.arange(8)
    y = np.arange(0, 4.0, 0.25)
    grid = NssGrid(data, [x, y], ["x", "y"])

    grd_a = grid_slice_interp(grid, 4.5, "x")
    grd_b = grid_slice_interp(grid, 4.5, 0)
    assert np.array_equal(grd_a, grd_b)
    grd_b = NssGrid(np.arange(72, 72 + 16), [y], ["y"])
    assert np.array_equal(grd_a, grd_b)

    grd_a = grid_slice_interp(grid, 1.1, "y")
    grd_b = grid_slice_interp(grid, 1.1, 1)
    assert np.array_equal(grd_a, grd_b)

    grd_b = NssGrid(np.arange(4.4, 116.7, 16, dtype=grd_a.dtype), [x], ["x"])

    assert np.array_equal(grd_a.axes, grd_b.axes)
    assert np.array_equal(grd_a.axis_names, grd_b.axis_names)
    assert np.allclose(grd_a.data, grd_b.data)
    assert np.allclose(grd_a, grd_b)


if __name__ == "__main__":
    test_grid_interp()
