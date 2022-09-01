"""
used to conver old data tables to more heirarchal hdf5s
"""


import numpy as np
import h5py


## Data set with shape (5, 5) and list containing column names as string
# data = np.random.rand(5, 5)
e = "./electron_EAS_table.h5"

e_data = h5py.File(e, "r")
e_data = np.array(e_data["EASdata_11"])


g = "./gamma_EAS_table.h5"

g_data = h5py.File(g, "r")
g_data = np.array(g_data["EASdata_22"])

p = "./pion_EAS_table.h5"

p_data = h5py.File(p, "r")
p_data = np.array(p_data["EASdata_211"])

# pro = "./proton_EAS_table.h5"

# pro_data = h5py.File(pro, "r")
# pro_data = np.array(pro_data["EASdata_100"])
# col_names = [
#     "PID",
#     "log10E[ev]",
#     "ZenithAng[deg]",
#     "AziAng[deg,0=E]",
#     "Nmax[N]",
#     "Xmax[g/cm^2]",
#     "Psuedo-X_0[g/cm^2]",
#     "poly_1",
#     "poly_2",
#     "poly_3",
#     "Chi^2/dof/sqrt(Nmax)",
# ]
# # ## Create file pointer
# ds_dt = np.dtype(
#     {
#         "names": col_names,
#         "formats": [(float)] * 11,
#     }
# )
# e_h5 = np.rec.fromarrays(e_data.T, dtype=ds_dt)

data_set_name = "gh_20_km.h5"
with h5py.File(data_set_name, "w") as fp:

    ## Store data
    ##fp["sub"] = data
    # comment = fp.create_dataset("ReadMe", data=col_names)
    ds1 = fp.create_dataset("electron", data=e_data)
    ds2 = fp.create_dataset("gamma", data=g_data)
    ds3 = fp.create_dataset("pion", data=p_data)
    # ds3 = fp.create_dataset("proton", data=pro_data)
