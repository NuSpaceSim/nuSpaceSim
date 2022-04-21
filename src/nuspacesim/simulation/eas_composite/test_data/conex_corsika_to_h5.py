import numpy as np
import h5py

# =============================================================================
# :********************:
# Aug.21 2019 Y.Akaike
# **********************
# Jun 14, 2021 corrected by JFK
# ***************
#   //   Gaisser-Hillas parameters
#   //
#   //  double t       = slant depth [g/cm2]  ! t is the dependent variable
#   //  double anmax   = val4 ! Nmax
#   //  double tmax    = val5  ! Xmax
#   //  double t0      = val6 ! XO
#   //  double lambda  = va]7+val8*t+val9*t*t; ! Lambda (which depends on t)
#   //  double expo = (tmax-t0)/lambda;
#   //  double f = anmax*pow((t-t0)/(tmax-t0),expo);  ! this gives Gaiser-Hillas function
#   //  expo = (tmax-t)/lambda;
#   //  f = f*exp(expo);  ! this gives Gaiser-Hillas function
# Primary
# val0:  "Particle" : particle ID (100:proton, 11:electron, 13:mu, 211:pi, 22:gamma)
# val1:  "lgE"      : log10 of the primary energy in eV
# val2:  "zenith"   : zenith angle in degree
# val3:  "azimuth"  : azimuth angle in degree (0 = East)
# GH-function
# val4:  "Nmax"     : Number of charged particles above cut-off at the shower maximum
# val5:  "Xmax"     : GH fit result for slant depth of the shower maximum (g/cm^2)
# val6:  "X0"       : "pseudo" first interaction point for GH fit
# val7:  "p1"       : first parameter for the polynomial function of the GH fit
# val8:  "p2"       : second parameter for the polynomial function of the GH fit
# val9:  "p3"       : third parameter for the polynomial function of the GH fit
# val10: "chi2"     : Chi squared / number of degree of freedom / sqrt (Nmax) for the fit
# (ex.) dumpGH_conex_p_E17_95deg_0km_eposlhc_435865130_100.dat
#       val0        val1        val2        val3        val4        val5        val6        val7        val8        val9       val10
# ----------------------------------------------------------------------------------------------------------------------------------
#        100  1.7000e+01  9.5000e+01  0.0000e+00  6.4879e+07  6.9425e+02  4.2095e+01  9.2704e+01 -6.6730e-02  3.6921e-05  2.1534e-01
#        100  1.7000e+01  9.5000e+01  0.0000e+00  6.2296e+07  6.6206e+02 -3.7594e+01  8.8569e+01 -5.8626e-02  3.3529e-05  2.3970e-01
#        100  1.7000e+01  9.5000e+01  0.0000e+00  5.9574e+07  8.1890e+02  1.3355e+02  9.4356e+01 -6.1632e-02  3.4648e-05  2.8097e-01
#        100  1.7000e+01  9.5000e+01  0.0000e+00  5.7770e+07  1.0016e+03 -1.8258e+02  5.0079e+01 -6.9525e-03  3.7304e-06  2.5226e+01
# =============================================================================
## Data set with shape (5, 5) and list containing column names as string
# data = np.random.rand(5, 5)
path = "./electron_EAS_table.h5"
data = h5py.File(path, "r")
electron = data.get("")
# col_names = ["a", "b", "c", "d", "e"]
# ## Create file pointer


# with h5py.File("data_set_2.HDF5", "w") as fp:
#     ds_dt = np.dtype(
#         {"names": col_names, "formats": [(float), (float), (float), (float), (float)]}
#     )
#     rec_arr = np.rec.array(data, dtype=ds_dt)
#     ## Store data
#     ##fp["sub"] = data
#     ds1 = fp.create_dataset("sub", data=rec_arr)
