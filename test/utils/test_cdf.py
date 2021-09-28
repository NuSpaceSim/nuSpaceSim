from nuspacesim.utils import cdf
# from nuspacesim.utils.interp import grid_interpolator

from importlib_resources import as_file, files
from nuspacesim.core import NssConfig
from nuspacesim.utils.grid import NssGrid
import numpy as np

# def test_cdf_sample_factory():
# cdf.cdf_sample_factory

if __name__ == "__main__":

    config = NssConfig()

    with as_file(files("nuspacesim.data.RenoNu2TauTables") / "nu2tau_cdf.hdf5") as file:
        tau_cdf_grid = NssGrid.read(
            file,
            format="hdf5",
            path=f"/log_nu_e_{config.simulation.log_nu_tau_energy}",
        )

    # tau_cdf_sample = cdf.cdf_sample_factory(tau_cdf_grid, 0)
    tau_cdf_legacy = cdf.legacy_cdf_interp(tau_cdf_grid)

    points = np.radians(np.array([1, 3, 5]))

    # u = np.random.uniform(tau_cdf_grid.axes[1][0], 1, points.shape)
    # print(tau_cdf_grid)
    u = np.full(points.shape, tau_cdf_grid.axes[0][3])

    # print("tau_cdf_grid:", tau_cdf_grid)
    # print("points:", points)
    # print("u:", u)
    # print("tau_cdf_legacy(points, u=u):", tau_cdf_legacy(points, u=u))
    # print("tau_cdf_sample(points, u=u):", tau_cdf_sample((points), u=u))

    RegularGridInterpolator
    
