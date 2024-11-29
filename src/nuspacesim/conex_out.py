import uproot
import numpy as np
import awkward as ak
from scipy.optimize import curve_fit
from .simulation.eas_optical.shower_properties import slant_depth_trig_approx as slant_depth_old
from .simulation.eas_optical.atmospheric_models import slant_depth
from .simulation.eas_optical.shower_properties import path_length_tau_atm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#from .simulation.auger_sim import geomsim as asim

def conex_out(data, profiles):

    def GH(X, X0, Xmax, Nmax, p3, p2, p1):

        return Nmax * ((X - X0) / (Xmax - X0)) ** ((Xmax - X0) / (p3 * X ** 2 + p2 * X + p1)) * np.exp(
            (Xmax - X) / (p3 * X ** 2 + p2 * X + p1))
    Zfirst = data["altDec"]
    TauEnergy = np.log10(data["tauEnergy"]) + 9 #Tau energy in log E/eV units
    """
    plt.hist(np.degrees(data["beta_rad"]),bins=50)
    time=datetime.datetime.now().microsecond
    plt.xlabel('Emergence angle (deg')
    plt.ylabel('Counts')
    plt.title('Histogram of thrown beta_tr uniform on the sphere')
    plt.savefig('beta_hist{}.png'.format(time))
    plt.figure()
    plt.hist(data["altDec"][data["altDec"]<2],bins=50)
    time=datetime.datetime.now().microsecond
    plt.savefig('altDec{}.png'.format(time))
    plt.figure()
    plt.plot(np.degrees(data["beta_rad"]),data["tauEnergy"],'.')
    plt.savefig('hola.png')
    """
    Zmax=20
    Zmin=0
    TauEnergyMin=20
    # Useful masks (Zfirst masks are necessary between 0 and 20km). To introduce masks you must also change them in simulation/eas_optical/eas.py
    mask = (Zfirst >= Zmin) & (Zfirst <= Zmax) #& (TauEnergy >= TauEnergyMin) # & (np.degrees(data["beta_rad"]) <= 3)  & (data["lenDec"]>=58) & (data["lenDec"]<=60)
    beta = data["beta_rad"][mask]
    Zfirst = data["altDec"][mask]
    n = np.size(Zfirst)
    TauEnergy = TauEnergy[mask]
    H = ak.ArrayBuilder()
    
    Xfirst=[]
    for i in range(n):
        Xfirst=np.append(Xfirst,slant_depth(0,Zfirst[i],np.pi / 2 - beta[i])[0])
    """
    Xfirstold = slant_depth_old(0, Zfirst, np.pi / 2 - beta)
    print(Xfirstold.size,Xfirst.size,'HOLA')
    plt.figure()
    bins = np.linspace(0, 40000, 50)
    plt.hist(Xfirstold,bins,label='Old Xfirst',alpha=0.5)
    plt.hist(Xfirst,bins,label='new Xfirst',alpha=0.5,linewidth=1.5)
    plt.ylabel('Counts')
    plt.xlabel('Xfirst')
    plt.grid()
    #plt.xscale('log')
    plt.yscale('log')
    plt.legend()   
    plt.title('Slant depths hist')
    plt.savefig('Xfirsttest.png')
    """

    X=ak.values_astype(ak.Array(profiles)[:,0],np.float32) #    X=ak.values_astype(ak.Array(profiles)[:,0]+Xfirst,np.float32)
    Z=ak.values_astype(profiles,np.float32)[:,1]
    RN=ak.values_astype(profiles,np.float32)[:,2]
    nX = ak.to_numpy(ak.num(X, axis=1))

    Xfirstmax=3*10**4
    mask2=(Xfirst<=Xfirstmax) #Remove too long profiles (which produce errors in offline)
    for i in range(n):
        if RN[i][-1]>np.max(RN[i])*0.5:
            mask2[i]=False
    nmasked=np.sum(~mask2)+np.sum(~mask)

    beta = beta[mask2]
    Zfirst = Zfirst[mask2]
    n = np.size(Zfirst)
    TauEnergy = TauEnergy[mask2]
    Xfirst = Xfirst[mask2]

    azim = 0*360 * np.random.rand(n) #Random azimuth
    zenith = 90 + np.degrees(beta)
    dEdXratio=0.0025935  #0.0025935 when comparing with Conex. This paper says 0.00219, but for general cosmic ray, not electrons only. https://doi.org/10.1016/S0927-6505(00)00101-8 in GeV /
    rootfile='nss_to_conex.root'
    print('Generating conex-like output in '+rootfile)
    print('Number of masked events ', nmasked)
    print('Number of valid events ', n)

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

    dist99=np.zeros(n)
    for i in range(n):
        Zi = np.array(Z[i])
        H.append(path_length_tau_atm(Zi, beta[i]))  #Build distance array
        RNmax=np.max(RN[i])
        RNargmax=np.argmax(RN[i])
        RNratio=np.array(RN[i]/RNmax)
        pos99=int(np.argmax(RNratio[RNargmax:]<0.05)+RNargmax)
        dist=path_length_tau_atm(Zi, beta[i])-path_length_tau_atm(Zi, beta[i])[0]
        dist99[i]=dist[pos99]
        if i==0:
            plt.figure()
            plt.plot(X[i],RN[i],label=f'Energy={TauEnergy[i]}')
            plt.scatter(X[i][pos99],RN[i][pos99])
            plt.grid()
            print(X[i][pos99])
            #plt.yscale('log')
            plt.legend()
            plt.savefig('XDectest.png')

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

        popt, pcov = curve_fit(GH, x, y, p0=init, maxfev=1000000)
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
    Hp = ak.values_astype(H.snapshot(), np.float32)
    #Hp = -Hp + Hp[:, 0, None]  #Shift Distance array to start at 0 and be negative (according to Conex) THIS SHOULD BE UNCOMMENTED

    #Calculate average lendecay
    dist99=dist99[dist99<80]
    plt.figure()
    plt.hist(dist99,bins=50)
    plt.grid()
    plt.xlabel('distance (km)')
    plt.title(f'distance of shower when N=1% of Nmax, Emin={TauEnergyMin}')
    plt.savefig('distto1percenthistE19.png')

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
        , "zenith": np.dtype('f4')
        , "azimuth": np.dtype('f4')
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
        , "lgEmin": nan8
        , "lgEmax": nan8
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
        , "zenith": zenith  # 90+np.degree(beta_rad)
        , "azimuth": azim
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
        , "D": Hp * 1000
        , "dEdX": RN * dEdXratio
        , "Mu": Xempty  # Xempty
        , "Gamma": Xempty  # Xempty
        , "Electrons": RN
        , "Hadrons": Xempty # Xempty
        , "dMu": Xempty  # Xempty
        , "EGround": Eg

    })
    f.close()
