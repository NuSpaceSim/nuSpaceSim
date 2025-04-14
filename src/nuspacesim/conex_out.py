import uproot
import numpy as np
import awkward as ak
from scipy.optimize import curve_fit
from .simulation.eas_optical.atmospheric_models import slant_depth
from .simulation.eas_optical.shower_properties import path_length_tau_atm
from .augermc import ecef_to_utm, earth_radius_centerlat
import matplotlib.pyplot as plt

#from .simulation.auger_sim import geomsim as asim
def GH(X, X0, Xmax, Nmax, p3, p2, p1):

    return Nmax * ((X - X0) / (Xmax - X0)) ** ((Xmax - X0) / (p3 * X ** 2 + p2 * X + p1)) * np.exp(
        (Xmax - X) / (p3 * X ** 2 + p2 * X + p1))

def conex_out(profiles,id,groundecef,vecef,beta,TauEnergy,Zfirst,azim,gpsarray,NuEnergy,tauExitProb,h):

    
    n=np.size(Zfirst)
    D = ak.ArrayBuilder()
    Xfirst=[]
    endatm=[]
    #Calculate Xfirst starting at sea level
    for i in range(n): #N AQUI
        Xfirst=np.append(Xfirst,slant_depth(0,Zfirst[i],np.pi / 2 - beta[i],earth_radius_centerlat/1000)[0])
        endatm=np.append(endatm,slant_depth(0,100,np.pi / 2 - beta[i],earth_radius_centerlat/1000)[0])

    X=ak.values_astype(ak.Array(profiles)[:,0]+Xfirst,np.float32) #    X=ak.values_astype(ak.Array(profiles)[:,0]+Xfirst,np.float32)
    Z=ak.values_astype(profiles,np.float32)[:,1]
    RN=ak.values_astype(profiles,np.float32)[:,2]
    nX = ak.to_numpy(ak.num(X, axis=1))

    Xlast=X[:,-1]
    maskendatm=(Xlast<endatm)



    print(np.sum(~maskendatm),'events with Xfirst out of atm')

    #""" REMOVED TEMPORARILY. EITHER FRED G IMPLEMENTATION FIXES THIS (THE IF) OR EXTEND PROFILE WITH GH FIT 
    #Xfirstmax=2*10**4
    """ mask2=(Xfirst<=Xfirstmax) #Remove too long profiles (which produce errors in offline)
    xfirsthigh=(~mask2).sum()
    print(Xfirst[~mask2],Zfirst[~mask2])
    for i in range(n):
        if RN[i][-1]>np.max(RN[i])*0.5:
            mask2[i]=False """
    mask2=maskendatm
    #nmasked=np.sum(~mask2)
    beta = beta[mask2]  
    Zfirst = Zfirst[mask2]
    TauEnergy = TauEnergy[mask2]
    Xfirst = Xfirst[mask2]
    NuEnergy = NuEnergy[mask2]
    tauExitProb = tauExitProb[mask2]
    gpsarray = gpsarray[mask2]
    azim = azim[mask2]
    id=id[mask2]
    groundecef = groundecef[mask2,:]

    n=np.size(Zfirst)

    zenith = 90 + np.degrees(beta)
    dEdXratio=0.0025935  #0.0025935 when comparing with Conex. This paper says 0.00219, but for general cosmic ray, not electrons only. https://doi.org/10.1016/S0927-6505(00)00101-8 in GeV /

    X= X[mask2]
    Z = Z[mask2]
    #Z = Z - Z[:, 0, None]   #Shift profile to start at 0 THIS SHOULD BE UNCOMMENTED
    RN = RN[mask2]
    # Useful variables to fill the conex File
    PID = np.array([100], dtype='i4')  # Proton type for Conex
    zmin = np.array([90], dtype='i4')
    zmax = np.array([132], dtype='i4')
    nan4 = np.array([np.nan], dtype='f4')
    nan8 = np.array([np.nan], dtype='f8')
    nuEmax=np.array([np.max(NuEnergy)], dtype='f8')
    nuEmin=np.array([np.min(NuEnergy)], dtype='f8')

    int4 = np.array([-1], dtype='i4')
    intn = np.full(n, -1, dtype='i4')
    nan4n = np.full(n, np.nan, dtype='f4')
    nan8n = np.full(n, np.nan, dtype='f8')
    Xempty = ak.zeros_like(X)
    OutputVersion = np.array([2.51], dtype='f4')
    b = [np.full(31, nan8)]
    a = ak.to_regular(ak.Array(b))
    Eground = np.full((n, 3), nan4)
    Eg = ak.to_regular(ak.Array(Eground))
    easting, northing, height, zonenumber, zoneletter = ecef_to_utm(groundecef)

    #Initialize some variables for GH fit
    Xmax = np.empty(n, dtype='f4')
    Nmax = np.empty(n, dtype='f4')
    X0 = np.empty(n, dtype='f4')
    p1 = np.empty(n, dtype='f4')
    p2 = np.empty(n, dtype='f4')
    p3 = np.empty(n, dtype='f4')
    chi2 = np.zeros(n, dtype='f4')
    Xmx = np.empty(n, dtype='f4')
    Nmx = np.empty(n, dtype='f4')
    
    for i in range(n):   #AQUI PONER N
        Zi=np.array(Z[i])
        #H0=path_length_tau_atm(h/1000, beta[i])   #Donde empieza la distance array? Al inicio de la atmosfera? En el punto de decay? Desde el core??? (seguro que no)
        D0=path_length_tau_atm(h/1000, beta[i], Re=earth_radius_centerlat) 
        D.append(path_length_tau_atm(Zi, beta[i],Re=earth_radius_centerlat)-D0)  #Build distance array from core (surface of Earth at h=1416m)

        x=np.array(X[i])
        # Shift profile in X and reduce magnitude in N to simplify the fits.
        x0=x[0]
        x=x-x0
        y=np.array(RN[i])/1e5

        init = np.empty(6)
        max_pos = np.argmax(y)
        y=y[0:max_pos*2]  #Only interested in profile around the maximum, disregard the tail
        x=x[0:max_pos*2]

        #Best initial values for a good, fast and reliable fit.
        init[1]=x[max_pos]
        init[0]=-0.30943336*init[1]
        init[2]=y[max_pos]
        init[3:] = [1e-7, 4e-4, 44]

        popt, __ = curve_fit(GH, x, y, p0=init, maxfev=10000)
        yfit = GH(x, *popt)

        #Calculate chi**2
        for j in range((yfit).size):
            chi2[i] += (y[j] - yfit[j]) ** 2 / (y[j])

        chi2[i] =chi2[i] / (nX[i] - 6) / (np.sqrt(popt[2]*1e5))

        g3=popt[3]
        g2=popt[4]
        g1=popt[5]

        #Undo the variable change. For p1, p2, p3 this involves shifting the parabola coefficients.
        X0[i]=popt[0]+x0
        Xmax[i]=popt[1]+x0
        Nmax[i]=popt[2]*1e5
        p3[i]=popt[3]
        p2[i]=g2-2*g3*x0
        p1[i]=g1-g2*x0+g3*x0**2
        Xmx[i]=x[max_pos]+x0
        Nmx[i]=y[max_pos]*1e5
        chi2[i]=chi2[i]*1e5
    Dp = ak.values_astype(D.snapshot(), np.float32)
    rootfile = f"nss_n{n}_lgE{int(nuEmax[0])}.root"    
    print('Generating conex-like output in '+rootfile)
    #print('Number of masked events Xfirst ', xfirsthigh,'Profile incomplete ',nmasked-xfirsthigh,' total= ',nmasked)
    print('Number of valid events ', n)

    zoneletter = np.array([ord(letter) for letter in zoneletter], dtype=np.uint8)



    branches_header = {
        "Seed1": np.dtype('i4')
        , "Particle": np.dtype('i4')
        , "Alpha": np.dtype('f8')
        , "lgEmin": np.dtype('f8')
        , "lgEmax": np.dtype('f8')
        , "zMin": np.dtype('f8')
        , "zMax": np.dtype('f8')
        , "SvnRevision": np.dtype('b')
        , "Version": np.dtype('f4')
        , "OutputVersion": np.dtype('f4')
        , "HEModel": np.dtype('i4')
        , "LEModel": np.dtype('i4')
        , "HiLowEgy": np.dtype('f4')
        , "hadCut": np.dtype('f4')
        , "emCut": np.dtype('f4')
        , "hadThr": np.dtype('f4')
        , "muThr": np.dtype('f4')
        , "emThr": np.dtype('f4')
        , "haCut": np.dtype('f4')
        , "muCut": np.dtype('f4')
        , "elCut": np.dtype('f4')
        , "gaCut": np.dtype('f4')
        , "lambdaLgE": ("f8", (31,))
        , "lambdaProton": ("f8", (31,))
        , "lambdaPion": ("f8", (31,))
        , "lambdaHelium": ("f8", (31,))
        , "lambdaNitrogen": ("f8", (31,))
        , "lambdaIron": ("f8", (31,))
    }
    branches_shower = {
        "lgE": np.dtype('f4')
        , "lgnuE": np.dtype('f4')
        , "ExitProb": np.dtype('f4')  
        , "zenith": np.dtype('f4')
        , "azimuth": np.dtype('f4')
        , "easting": np.dtype('f4')
        , "northing": np.dtype('f4')
        , "height": np.dtype('f4')
        , "zonenumber": np.dtype('i4')
        , "zoneletter": np.dtype('i4')
        , "eventid": np.dtype('i4')
        , "telescopeid": np.dtype('i4')
        , "Seed2": np.dtype('i4')
        , "Seed3": np.dtype('i4')
        , "Xfirst": np.dtype('f4')
        , "Hfirst": np.dtype('f4')
        , "XfirstIn": np.dtype('f4')
        , "altitude": np.dtype('f8')
        , "X0": np.dtype('f4')
        , "Xmax": np.dtype('f4')
        , "Nmax": np.dtype('f4')
        , "p1": np.dtype('f4')
        , "p2": np.dtype('f4')
        , "p3": np.dtype('f4')
        , "chi2": np.dtype('f4')
        , "Xmx": np.dtype('f4')
        , "Nmx": np.dtype('f4')
        , "XmxdEdX": np.dtype('f4')
        , "dEdXmx": np.dtype('f4')
        , "cpuTime": np.dtype('f4')
        , "X": 'var * float32'
        , "N": 'var * float32'
        , "H": 'var * float32'
        , "D": 'var * float32'
        , "dEdX": 'var * float32'
        , "Mu": 'var * float32'
        , "Gamma": 'var * float32'
        , "Electrons": 'var * float32'
        , "Hadrons": 'var * float32'
        , "dMu": 'var * float32'
        , "EGround": ('f4', (3,))
    }

    f = uproot.recreate(rootfile)

    f.mktree("Header", branches_header, title="run header")
    f.mktree("Shower", branches_shower, title="shower info")
    f["Header"].extend({
        "Seed1": int4
        , "Particle": PID  # Proton type (no specific ID for tau)
        , "Alpha": nan8
        , "lgEmin": nuEmin
        , "lgEmax": nuEmax
        , "zMin": zmin
        , "zMax": zmax
        , "SvnRevision": [0]
        , "Version": [64]
        , "OutputVersion": OutputVersion
        , "HEModel": int4
        , "LEModel": int4
        , "HiLowEgy": nan4
        , "hadCut": nan4
        , "emCut": nan4
        , "hadThr": nan4
        , "muThr": nan4
        , "emThr": nan4
        , "haCut": nan4
        , "muCut": nan4
        , "elCut": nan4
        , "gaCut": nan4
        , "lambdaLgE": a
        , "lambdaProton": a
        , "lambdaPion": a
        , "lambdaHelium": a
        , "lambdaNitrogen": a
        , "lambdaIron": a})
    f["Shower"].extend({
        "lgE": TauEnergy
        , "lgnuE": NuEnergy
        , "ExitProb": tauExitProb
        , "zenith": zenith  # 90+np.degree(beta_rad)
        , "azimuth": np.degrees(azim)
        , "easting": easting
        , "northing": northing
        , "height": height
        , "zonenumber": zonenumber
        , "zoneletter": zoneletter
        , "eventid": gpsarray
        , "telescopeid": id
        , "Seed2": intn
        , "Seed3": intn
        , "Xfirst": Xfirst
        , "Hfirst": Zfirst * 1000
        , "XfirstIn": ak.ones_like(nan4n) * 0.5
        , "altitude": nan8n
        , "X0": X0
        , "Xmax": Xmax
        , "Nmax": Nmax
        , "p1": p1
        , "p2": p2
        , "p3": p3
        , "chi2": chi2
        , "Xmx": Xmx
        , "Nmx": Nmx
        , "XmxdEdX": Xmx
        , "dEdXmx": Nmx * dEdXratio
        , "cpuTime": nan4n
        , "X": X
        , "N": RN
        , "H": Z * 1000  # in meters
        , "D": Dp * 1000
        , "dEdX": RN * dEdXratio
        , "Mu": Xempty  # Xempty
        , "Gamma": Xempty  # Xempty
        , "Electrons": RN
        , "Hadrons": Xempty # Xempty
        , "dMu": Xempty  # Xempty
        , "EGround": Eg

    })
    f.close()
