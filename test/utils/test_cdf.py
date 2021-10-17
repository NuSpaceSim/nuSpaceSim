from nuspacesim.utils import cdf

from nuspacesim.utils.grid import NssGrid
import numpy as np

try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files


def test_grid_inverse_sampler():
    # Currently only used for tau cdf tables.
    # grid of tau_cdf tables
    with as_file(files("nuspacesim.data.nupyprop_tables") / "nu2tau_cdf.hdf5") as file:
        tau_cdf_grid = NssGrid.read(file, format="hdf5")
        tau_cdf_sampler = cdf.grid_inverse_sampler(tau_cdf_grid, 8.0)
        betas = np.random.uniform(np.radians(1), np.radians(42), size=512)
        sample = tau_cdf_sampler(betas)
        assert not np.allclose(sample, 0.0)
        assert not np.allclose(sample, 1.0)
