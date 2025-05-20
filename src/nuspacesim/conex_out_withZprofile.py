import uproot
import numpy as np
import awkward as ak
from scipy.optimize import curve_fit
#from .simulation.eas_optical.atmospheric_models import slant_depth
from .simulation.eas_optical.shower_properties import path_length_tau_atm
from .augermc import ecef_to_utm, earth_radius_centerlat, slant_depth
import matplotlib.pyplot as plt

#from .simulation.auger_sim import geomsim as asim
def GH(X, X0, Xmax, Nmax, p3, p2, p1):
    gh_lam =np.array(p3 * X ** 2 + p2 * X + p1)
    #gh_lam[gh_lam > 100.0] = 100.0s
    #gh_lam[gh_lam < 1.0e-5] = 1.0e-5

    return Nmax * ((X - X0) / (Xmax - X0)) ** ((Xmax - X0) / gh_lam) * np.exp(
        (Xmax - X) / gh_lam)

def conex_out(profiles,id,groundecef,vecef,beta,TauEnergy,Zfirst,azim,gpsarray,NuEnergy,tauExitProb,h, ghparams,XfirstOffline):

    
    n=np.size(Zfirst)
    D = ak.ArrayBuilder()
    Xfirst=[]
    ghparams=np.array(ghparams)
    Z=ak.values_astype(profiles,np.float32)[:,1]
    RN=ak.values_astype(profiles,np.float32)[:,2]
    nan_mask = np.isnan(RN)
    maskheight=(Z<30)
    nZ = ak.to_numpy(ak.num(Z, axis=1))

    maskheight = maskheight & (~nan_mask)
    Z=Z[maskheight]
    nZmask = ak.to_numpy(ak.num(Z, axis=1))

    RN=RN[maskheight]
    #Calculate Xfirst starting at sea level
    Xfirst=XfirstOffline

    X=ak.values_astype(ak.Array(profiles)[:,0]+Xfirst,np.float32) #    X=ak.values_astype(ak.Array(profiles)[:,0]+Xfirst,np.float32)
    nX = ak.to_numpy(ak.num(X, axis=1))

    X=X[maskheight]
    nXmask = ak.to_numpy(ak.num(X, axis=1))
    bign=(nZ>1000)
    print(nZ[bign],nZmask[bign])
    print(nX[bign],nXmask[bign])

    #maskendatm=(Xlast<endatm*10)




    #print(np.sum(~maskendatm),'events with Xfirst out of atm')
    #if np.sum(~maskendatm)>0:
    #    print('ATTENTION\n \n \n \n ATTENTION, \n \n \n SOME EVENTS HAVE XLAST OUT OF ATMOSPHERE')
    #""" REMOVED TEMPORARILY. EITHER FRED G IMPLEMENTATION FIXES THIS (THE IF) OR EXTEND PROFILE WITH GH FIT 
    #Xfirstmax=2*10**4
    """ mask2=(Xfirst<=Xfirstmax) #Remove too long profiles (which produce errors in offline)
    xfirsthigh=(~mask2).sum()
    print(Xfirst[~mask2],Zfirst[~mask2])
    for i in range(n):
        if RN[i][-1]>np.max(RN[i])*0.5:
            mask2[i]=False 
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
    groundecef = groundecef[mask2,:]"""

    n=np.size(Zfirst)

    zenith = 90 + np.degrees(beta)
    dEdXratio=0.0025935  #0.0025935 when comparing with Conex. This paper says 0.00219, but for general cosmic ray, not electrons only. https://doi.org/10.1016/S0927-6505(00)00101-8 in GeV /

    #X= X[mask2]
    #Z = Z[mask2]
    #Z = Z - Z[:, 0, None]   #Shift profile to start at 0 THIS SHOULD BE UNCOMMENTED
    #RN = RN[mask2]
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
    chi2old = np.zeros(n, dtype='f4')

    Xmx = np.empty(n, dtype='f4')
    Nmx = np.empty(n, dtype='f4')
    shiftedparams=ghparams.copy()
    fig, axs = plt.subplots(2, 1, figsize=(14, 20), dpi=200)


    for i in range(n):   #WITH XFIRST    
        #RN[i]=RN[i][~nan_mask]   #Why is this happening? Check later
        #X[i]=X[i][~nan_mask]
        #Z[i]=Z[i][~nan_mask]

        D0=path_length_tau_atm(h/1000, beta[i], Re=earth_radius_centerlat) 
        D.append(path_length_tau_atm(Z[i], beta[i],Re=earth_radius_centerlat)-D0)  #Build distance array from core (surface of Earth at h=1416m)

        shiftedparams[i,0]=ghparams[i,0]+Xfirst[i] 
        shiftedparams[i,1]=ghparams[i,1]+Xfirst[i]
        shiftedparams[i,4]=ghparams[i,4]-2*ghparams[i,3]*Xfirst[i]
        shiftedparams[i,5]=ghparams[i,5]-ghparams[i,4]*Xfirst[i]+ghparams[i,3]*Xfirst[i]**2
        yfitorig=GH(X[i], *shiftedparams[i])
        max_pos = np.argmax(RN[i])
        #Calculate chi**2
        for j in range(len(yfitorig)):
            chi2[i] += (RN[i][j] - yfitorig[j]) ** 2 / (RN[i][j])


        chi2[i] =chi2[i] / (len(X[i]) - 6) 
        X0[i]=shiftedparams[i,0]
        Xmax[i]=shiftedparams[i,1]
        Nmax[i]=shiftedparams[i,2]
        p3[i]=shiftedparams[i,3]
        p2[i]=shiftedparams[i,4]
        p1[i]=shiftedparams[i,5]
        Xmx[i]=X[i,max_pos]
        Nmx[i]=RN[i,max_pos]

        
        x=np.array(X[i])
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
        for j in range(len(yfit)):
            chi2old[i] += (y[j] - yfit[j]) ** 2 / (y[j])

        chi2old[i] =1e5*chi2old[i] / (len(y) - 6) #/ (np.sqrt(popt[2]*1e5)) why is this here??
        
        nancheck = np.isnan(chi2[i])
        axs[0].plot(Z[i],RN[i],linewidth=0.2)
        axs[1].plot(X[i],RN[i],linewidth=0.2)

        if i==-1:#chi2[i]>=1 or chi2[i]==0 or nancheck:
            y=RN[i]
            print('RN',y[y<0.1])

            print((len(X[i]) - 6))
            print('Original params',ghparams[i])
            #print('Fit params',popt)
            print('Chi2',chi2[i])
            print('Chi2 old',chi2old[i])
            print(i)
            plt.figure(figsize=(10, 6),dpi=250)
            plt.plot(X[i],RN[i],'.',markersize=4,label=f'Energy={TauEnergy[i]}')
            plt.plot(x+x0,yfit*1e5,linewidth=2,label='GH fit',alpha=0.7,color='green')
            plt.plot(X[i],yfitorig,linewidth=1,label='GH of original params',alpha=0.7,color='red')
            plt.title('Example profile, GH fit, and GH function from original parameters')
            plt.xlabel('Slant depth (g/cm2)')
            plt.ylabel('Charged particles')
            plt.grid()
            print('Data Xmax, Nmax',Xmx[i],Nmx[i])
            print('Fit Xmax, Nmax',Xmax[i],Nmax[i])
            print('Fit X0, chi2 ',X0[i],chi2[i])
            plt.yscale('log')
            plt.legend()
            plt.savefig('proftest.png')
            print(i)
    axs[0].grid(True)
    axs[1].grid(True)
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')

    fig.savefig('profilesvsheightandslant.png')
    mask=(chi2>1) & (chi2==np.NaN) & (chi2<=0)
    print('chi2',chi2,chi2[mask])
    Dp = ak.values_astype(D.snapshot(), np.float32)
    rootfile = f"nss_n{n}_lgE{int(nuEmax[0])}.root"    
    print('Generating conex-like output in '+rootfile)
    #print('Number of masked events Xfirst ', xfirsthigh,'Profile incomplete ',nmasked-xfirsthigh,' total= ',nmasked)
    print('Number of valid events ', n)

    zoneletter = np.array([ord(letter) for letter in zoneletter], dtype=np.uint8)
    Zempty=ak.full_like(Z,0)


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
