To read
with h5py.File("./filenam.h5py", "r") as f:
    muons = np.array(f["muons"])
    electron_positrons = np.array(f["electron_positron"])
    charged = np.array(f["charged"])
    gammas = np.array(f["gammas"])
    hadrons = np.array(f["hadrons"])

each component has a table with
angle | slope | slope uncertainty 1-sigma| intercept | intercept uncertainty 1-sigma