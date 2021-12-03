import numpy as np

from nuspacesim.utils.grid import NssGrid


def test_grid_create():

    data = np.arange(8 * 16).reshape(8, 16)
    x = np.arange(8)
    y = np.arange(0, 4.0, 0.25)
    grid = NssGrid(data, [x, y], ["x", "y"])

    print(grid)

    assert np.array_equal(grid.data, data)
    assert np.array_equal(grid.axes[0], x)
    assert np.array_equal(grid.axes[1], y)


def test_axes_slice():
    data = np.arange(8 * 16).reshape(8, 16)
    x = np.arange(8)
    y = np.arange(0, 4.0, 0.25)
    grid = NssGrid(data, [x, y], ["x", "y"])

    assert np.array_equal(grid["x"], x)
    assert np.array_equal(grid["y"], y)
    assert not np.array_equal(grid["x"], y)
    assert not np.array_equal(grid["y"], x)

    assert np.array_equal(grid[0], x)
    assert np.array_equal(grid[1], y)
    assert not np.array_equal(grid[0], y)
    assert not np.array_equal(grid[1], x)

    assert grid[grid["x"] == 4, :] == NssGrid(data[4, :], [y], ["y"])
    assert grid[grid["x"] == 4, grid["y"] >= 0] == NssGrid(data[4, :], [y], ["y"])
    assert grid[grid["x"] > 4, grid["y"] == 1.0] == NssGrid(data[5:, 4], [x[5:]], ["x"])
    assert grid[grid["x"] == 4, 2:] == NssGrid(data[4, 2:], [y[2:]], ["y"])


if __name__ == "__main__":
    test_grid_create()
    test_axes_slice()
