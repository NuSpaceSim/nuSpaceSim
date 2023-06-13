
import uproot
import os
import csv
import numpy as np
import awkward as ak
import math
from scipy.optimize import curve_fit
import warnings

import matplotlib.pyplot as plt
from .simulation.geometry.region_geometry import RegionGeom
from .simulation.eas_optical.shower_properties import slant_depth_trig_approx as slant_depth
warnings.simplefilter('ignore', np.RankWarning)

def conex_out_GH4(data,geom):

    def GH(X, X0, Xmax, Nmax, p1):

        return Nmax * ((X - X0) / (Xmax - X0)) ** ((Xmax - X0) / (p1)) * np.exp(
            (Xmax - X) / (p1))
    Zfirst=data["altDec"]
    mask=(Zfirst<=20)
    beta=data["beta_rad"][mask]
    Zfirst=data["altDec"][mask]
    TauEnergy=data["tauEnergy"][mask]
    n=np.size(Zfirst)
    X=ak.ArrayBuilder()
    Z=ak.ArrayBuilder()
    RN=ak.ArrayBuilder()
    Xfirst=slant_depth(0,Zfirst,math.pi/2-beta)
    with open('showerdata.csv','r') as file:
        lines=csv.reader(file)
        for i,line in enumerate(lines):
            if i % 3 == 0:
                X.append(Xfirst[int(i/3)]+np.fromstring(line[0], sep=' ',dtype='f4'))
            elif i % 3 == 1:
                Z.append(np.fromstring(line[0], sep=' ',dtype='f4'))
            else:
                RN.append(np.fromstring(line[0], sep=' ',dtype='f4'))
    nX=ak.num(X,axis=1  )
    nX=ak.to_numpy(nX)
    nRN = ak.num(RN, axis=1)
    nRN = ak.to_numpy(nRN)
    mask2=(nX>0) & (nX==nRN)
    beta=beta[mask2]
    Zfirst=Zfirst[mask2]
    TauEnergy=TauEnergy[mask2]
    Xfirst=Xfirst[mask2]
    n=np.size(Zfirst)
    nX=nX[mask2]
    X=ak.values_astype(X.snapshot(),np.float32)[mask2]
    Z=ak.values_astype(Z.snapshot(),np.float32)[mask2]
    #print('These are the altDec values\n', np.array(Zfirst),'\n These are the first Z values \n',np.array(Z[:,0]))
    RN=ak.values_astype(RN.snapshot(),np.float32)[mask2]
    PID = np.array([100], dtype='i4') #Proton type to make it work
    zmin=np.array([90], dtype='i4')
    zmax=np.array([132], dtype='i4')
    nan4 = np.array([np.NaN], dtype='f4')
    nan8 = np.array([np.NaN], dtype='f8')
    int4=np.array([-1],dtype='i4')
    intn=np.full(n,-1,dtype='i4')
    nan4n=np.full(n,np.NaN,dtype='f4')
    nan8n=np.full(n,np.NaN,dtype='f8')
    #X=ak.Array([X])
    #Z=ak.Array([Z])
    #RN=ak.Array([RN])
    Xempty=ak.zeros_like(X)
    OutputVersion=np.array([2.51],dtype='f4')
    b = [np.full(31, nan8)]
    Eground=np.full((3,n), nan4)
    Eg=ak.to_regular(ak.Array(Eground))
    a = ak.to_regular(ak.Array(b))
    Xmax=np.empty(n,dtype='f4')
    Nmax=np.empty(n,dtype='f4')
    X0=np.empty(n,dtype='f4')
    p1=np.empty(n,dtype='f4')
    p2=np.empty(n,dtype='f4')
    p3=np.empty(n,dtype='f4')
    chi2=np.zeros(n,dtype='f4')
    Xmx=np.empty(n,dtype='f4')
    Nmx=np.empty(n,dtype='f4')
    print('Number of valid events ',n)
    count = np.sum(nX > 500)
    print('Number of weird events ', count )


    for i in range (n):
        x=np.array(X[i])
        y=np.array(RN[i])/1e5
        init = np.empty(4)
        max_pos = np.argmax(y)
        #print('Xmax position ',max_pos,' Xlength ', x.size ,' RNlength ', y.size, 'Xfirst ',Xfirst[i], 'Xmax ', x[max_pos])
        polyopt = np.polyfit(x[max_pos - 1:max_pos + 2], y[max_pos - 1:max_pos + 2], 2)
        init[0] = -100
        init[1] = np.abs(-polyopt[1] / 2 / polyopt[0])
        init[2] =np.abs( np.polyval(polyopt, init[1]))
        init[3] = 45.94
        bounds = ([-1000, 0, 0, 0, -100, 0.1], [x[max_pos],  np.inf, np.inf, 10, 100, 10000])
        popt, pcov = curve_fit(GH, x[0:max_pos*2], y[0:max_pos*2], p0=init,maxfev=1000000)
        yfit = GH(x[0:max_pos*2], *popt)
        for j in range((y[0:max_pos*2]).size):
            chi2[i] += (y[j] - yfit[j]) ** 2 / (yfit[j]*0.01)
        chi2[i] =chi2[i] / (nX[i] - 6)
        print(i, ' Chi2 ',chi2[i])
        if chi2[i]>100 or math.isnan(chi2[i]):
            print('Fit params, X0, Xmax, Nmax, p3, p2, p1(lambda) ', popt)
            plt.figure()
            plt.plot(x,y,label='Profile')
            plt.plot(x[0:max_pos*2],yfit, label='GH fit')
            kurs = "%i.png" % i
            plt.savefig(kurs, format='png')
            plt.close()
        else:
            plt.figure()
            plt.plot(x,y,label='Profile')
            plt.plot(x[0:max_pos*2],yfit, label='GH fit')
            kurs = "%i.png" % i
            plt.savefig('good'+kurs, format='png')
            plt.close()

        X0[i]=popt[0]
        Xmax[i]=popt[1]
        Nmax[i]=popt[2]*1e5
        p1[i]=popt[3]
        Xmx[i]=x[max_pos]
        Nmx[i]=y[max_pos]*1e6
    branches_header = {
        "Seed1": np.dtype('i4')
        , "Particle": np.dtype('i4')
        , "Alpha": np.dtype('f8')
        , "lgEmin": np.dtype('f8')
        , "lgEmax": np.dtype('f8')
        , "zMin": np.dtype('f8')
        , "zMax": np.dtype('f8')
        , "SvnRevision": np.dtype('b')
        # ,"SvnRevision":np.dtype('string')
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
    f = uproot.recreate('nss_to_conex.root')

    f.mktree("Header", branches_header, title="run header")
    f.mktree("Shower", branches_shower, title="shower info")
    f["Header"].extend({
        "Seed1": int4
        , "Particle": PID #No distinction between tau- and tau+
        , "Alpha": nan8   #UPDATE THIS
        , "lgEmin": nan8
        # UPDATE THIS
        , "lgEmax": nan8  # UPDATE THIS
        , "zMin": zmin
        , "zMax": zmax
        , "SvnRevision": [1]
        # ,"SvnRevision":np.dtype('string')
        , "Version": nan4
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
        "lgE": TauEnergy       #TauEnergy
        , "zenith": 90+np.degrees(beta)  #90-np.degree(beta_rad)
        , "azimuth": 2*math.pi*np.random.rand(n) #phiTrSubV
        , "Seed2": intn
        , "Seed3": intn
        , "Xfirst": Xfirst
        , "Hfirst": Zfirst  #altDec
        , "XfirstIn": nan4n
        , "altitude": nan8n
        , "X0": X0
        , "Xmax": Xmax
        , "Nmax": Nmax
        , "p1": p1
        , "p2": nan4n
        , "p3": nan4n
        , "chi2": chi2
        , "Xmx": Xmx
        , "Nmx": Nmx
        , "XmxdEdX": nan4n
        , "dEdXmx": nan4n
        , "cpuTime": nan4n
        , "nX": nX
        , "X": X
        , "N": Z
        , "H": Xempty
        , "D": RN
        , "dEdX": Xempty
        , "Mu": Xempty
        , "Gamma": Xempty
        , "Electrons": RN
        , "Hadrons": Xempty
        , "dMu": Xempty
        , "EGround": Eg

    })
    header = f["Header"]
    shower = f["Shower"]
    f.close()
    os.remove("showerdata.csv")