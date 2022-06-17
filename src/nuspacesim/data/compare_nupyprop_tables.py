from nuspacesim.utils.grid import NssGrid

pexit_v0 = NssGrid.read(
    "nupyprop_tables/nu2tau_pexit.0.h5", path="pexit_regen", format="hdf5"
)
pexit_v1 = NssGrid.read(
    "nupyprop_tables/nu2tau_pexit.1.h5", path="pexit_regen", format="hdf5"
)
pexit_v2 = NssGrid.read(
    "nupyprop_tables/nu2tau_pexit.2.h5", path="pexit_regen", format="hdf5"
)

assert pexit_v2[:, :9] == pexit_v1[:, :9]
assert pexit_v2[:, 9:] == pexit_v0

cdf_v0 = NssGrid.read("nupyprop_tables/nu2tau_cdf.0.h5", path="/", format="hdf5")
cdf_v1 = NssGrid.read("nupyprop_tables/nu2tau_cdf.1.h5", path="/", format="hdf5")
cdf_v2 = NssGrid.read("nupyprop_tables/nu2tau_cdf.2.h5", path="/", format="hdf5")

assert cdf_v2[:, :9, :] == cdf_v1[:, :9, :]
assert cdf_v2[:, 9:, :] == cdf_v0
