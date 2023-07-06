To read
with h5py.File("./nmax_rms_params.h5py", "r") as f:
    leptonic = np.array(f["leptonic"])
    one_body_kpi = np.array(f["one_body_kpi"])
    with_pi0 = np.array(f["with_pi0"])
    no_pi0 = np.array(f["no_pi0"])

one can also add the prefix of "mean_" or "rms_" for each of the key above to find the mean shower for that decay channel grouping

each decay channel grouping has a table with the columns

gaussian with exponential tail: https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
lambda | sigma | mu | maximum value truncation


#%% shower decay channels
lepton_decay = [300001, 300002]
had_pionkaon_1bod = [200011, 210001]
# fmt: off
had_pi0 = [300111, 310101, 400211, 410111, 410101, 410201, 401111, 400111, 500131,
           500311, 501211, 501212, 510301, 510121, 510211, 510111, 510112, 600411,
           600231,
           ]

had_no_pi0 = [310001, 311001, 310011, 311002, 311003, 400031, 410021, 410011, 410012,
              410013, 410014, 501031, 501032, 510031, 600051,
              ]
# fmt: on
