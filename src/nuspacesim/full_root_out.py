import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
def full_root_out(n,maxangle,NuEnergy,TauEnergy,energy_threshold,groundecef,vecef,decayecef,altdec,beta,azim,gpsarray,tauExitProb):

    rootfile='all_events_data.root'
    zenith = 90 + np.degrees(beta)
    # plt.plot(zenith,tauExitProb,'.',markersize=0.1)
    # plt.yscale('log')
    # plt.hist(zenith,bins=100)
    # plt.savefig('testprob.png')
    # Useful variables to fill the conex File

    #Initialize some variables for GH fit
    nuEmax=np.array([np.max(NuEnergy)], dtype='f8')
    nuEmin=np.array([np.min(NuEnergy)], dtype='f8')

    branches_header = {
        "n": np.dtype('i4')
        , "maxangle": np.dtype('f8')
        , "e_thr": np.dtype('f8')
        , "lgEmin": np.dtype('f8')
        , "lgEmax": np.dtype('f8')
    }
    branches_shower = {
        "lgE": np.dtype('f4')
        , "lgnuE": np.dtype('f4')
        , "ExitProb": np.dtype('f4')  
        , "zenith": np.dtype('f4')
        , "azimuth": np.dtype('f4')
        , "xcoreecef": np.dtype('f4')
        , "ycoreecef": np.dtype('f4')
        , "zcoreecef": np.dtype('f4')
        , "xaxis": np.dtype('f4')
        , "yaxis": np.dtype('f4')
        , "zaxis": np.dtype('f4')
        , "xdecay": np.dtype('f4')
        , "ydecay": np.dtype('f4')
        , "zdecay": np.dtype('f4')
        , "altDec": np.dtype('f4')
        , "eventid": np.dtype('i4')
    }

    f = uproot.recreate(rootfile)

    f.mktree("Header", branches_header, title="run header")
    f.mktree("Shower", branches_shower, title="shower info")
    f["Header"].extend({
        "n": [n]
        , "maxangle": [maxangle]  # Proton type (no specific ID for tau)
        , "e_thr": [energy_threshold]
        , "lgEmin": nuEmin
        , "lgEmax": nuEmax
        })
    f["Shower"].extend({
        "lgE": TauEnergy
        , "lgnuE": NuEnergy
        , "ExitProb": tauExitProb
        , "zenith": zenith  # 90+np.degree(beta_rad)
        , "azimuth": np.degrees(azim)
        , "xcoreecef": groundecef[:,0]
        , "ycoreecef": groundecef[:,1]
        , "zcoreecef": groundecef[:,2]
        , "xaxis": vecef[:,0]
        , "yaxis": vecef[:,1]
        , "zaxis": vecef[:,2]
        , "xdecay": decayecef[:,0]
        , "ydecay": decayecef[:,1]
        , "zdecay": decayecef[:,2]
        , "altDec": altdec
        , "eventid": gpsarray
    })
    f.close()