import uproot
import os
import csv
import numpy as np
import awkward as ak
import datetime
import math
import shutil
from scipy.optimize import curve_fit
import warnings

import matplotlib.pyplot as plt
from .simulation.geometry.region_geometry import RegionGeom
from .simulation.eas_optical.shower_properties import slant_depth_trig_approx as slant_depth
from .simulation.eas_optical.shower_properties import path_length_tau_atm

warnings.simplefilter('ignore', np.RankWarning)

def conex_out(data,geom):

    def GH(X, X0, Xmax, Nmax, p3, p2, p1):

        return Nmax * ((X - X0) / (Xmax - X0)) ** ((Xmax - X0) / (p3 * X ** 2 + p2 * X + p1)) * np.exp(
            (Xmax - X) / (p3 * X ** 2 + p2 * X + p1))
    Zfirst=data["altDec"]
    TauEnergy=np.log10(data["tauEnergy"])+9
    mask=(Zfirst>=0.01) #& (Zfirst<=0.05)   & (np.degrees(data["beta_rad"])>=4.9) & (np.degrees(data["beta_rad"])<=5.1)& (TauEnergy >= 18.45)& (TauEnergy <= 18.55)#& (data["lenDec"]<=5)
    beta=data["beta_rad"][mask]
    Zfirst=data["altDec"][mask]
    TauEnergy=TauEnergy[mask]
    print('Number of masked altitude/shower energy profiles ' ,np.sum(~mask))
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
    nZ = ak.num(Z, axis=1)
    nZ = ak.to_numpy(nZ)

    mask2= (nX==nRN) & (nX==nZ)
    print('Number of incoherent profiles ' ,np.sum(~mask2))
    beta=beta[mask2]
    Zfirst=Zfirst[mask2]
    TauEnergy=TauEnergy[mask2]
    Xfirst=Xfirst[mask2]
    n=np.size(Zfirst)
    azim=360* np.random.rand(n)*0
    zenith=90+np.degrees(beta)
    #zenith=np.full_like(zenith,150)
    """
    plt.hist((data["lenDec"][mask])[mask2],50)
    plt.xlabel('distance (km)')
    plt.ylabel('Counts')
    #plt.xlim(0,50)
    #plt.yscale('log')
    kurs = "%i" % Xfirst.size
    nX=nX[mask2]
    plt.title('Histogram of distance travelled until decay, n='+kurs)
    plt.grid(visible=True)
    plt.savefig('Hist_lenDec')

    plt.figure()
    plt.hist((data["altDec"][mask])[mask2],50)
    plt.xlabel('height (km)')
    plt.ylabel('Counts')
    #plt.xlim(0,3)
    #plt.yscale('log')
    kurs = "%i" % Xfirst.size
    plt.grid(visible=True)
    plt.title('Histogram of height of decay, n='+kurs)
    plt.savefig('Hist_altDec')

    plt.figure()
    plt.hist(TauEnergy,50)
    plt.xlabel('log10 Shower Energy in eV')
    plt.ylabel('Counts')
    plt.xlim()
    #plt.yscale('log')
    kurs = "%i" % Xfirst.size
    plt.title('Histogram of shower energies, n='+kurs)
    plt.grid(visible=True)
    plt.savefig('Hist_shower_energy')

    plt.figure()
    plt.hist(zenith,50)
    plt.xlabel('Zenith angle (deg)')
    plt.ylabel('Counts')
    #plt.yscale('log')
    kurs = "%i" % Xfirst.size
    plt.title('Histogram of showers zenith angle, n='+kurs)
    plt.grid(visible=True)
    plt.savefig('Hist_zenith')

    plt.figure()
    plt.hist(azim,50)
    plt.xlabel('Azimuth angle (deg)')
    plt.ylabel('Counts')
    #plt.yscale('log')
    kurs = "%i" % Xfirst.size
    plt.title('Histogram of showers azimuth angle, n='+kurs)
    plt.grid(visible=True)
    plt.savefig('Hist_azimuth')
    plt.figure()
    """

    X=ak.values_astype(X.snapshot(),np.float32)[mask2]
    Z=ak.values_astype(Z.snapshot(),np.float32)[mask2]
    #print('These are the altDec values\n', np.array(Zfirst),'\n These are the first Z values \n',np.array(Z[:,0]))


#ATTENTION!!!
    RN=ak.values_astype(RN.snapshot(),np.float32)[mask2]*24
# ATTENTION!!! FACTOR ON RN




    nRN = ak.num(RN, axis=1)
    nRN = ak.to_numpy(nRN)
    H=ak.ArrayBuilder()
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

    Xempty=ak.zeros_like(X)
    OutputVersion=np.array([2.51],dtype='f4')
    b = [np.full(31, nan8)]
    a = ak.to_regular(ak.Array(b))
    Eground=[np.full((3,n), nan4)]
    Eg=ak.to_regular(ak.Array(Eground))
    #nEg = ak.num(Eground, axis=1)
    #nEg = ak.to_numpy(nEg)
    Xmax=np.empty(n,dtype='f4')
    Nmax=np.empty(n,dtype='f4')
    X0=np.empty(n,dtype='f4')
    p1=np.empty(n,dtype='f4')
    p2=np.empty(n,dtype='f4')
    p3=np.empty(n,dtype='f4')
    chi2=np.zeros(n,dtype='f4')
    Xmx=np.empty(n,dtype='f4')
    Nmx=np.empty(n,dtype='f4')
    heightMax=np.empty(n,dtype='f4')
    lenMax=np.empty(n,dtype='f4')
    print('Number of valid events ',n)
    #count = np.sum(nX > 500)
    #print('Number of weird events ', count )
    k=0
    print(nX)
    for i in range (n):
        x=np.array(X[i])
        y=np.array(RN[i])/1e5
        Zi=np.array(Z[i])
        H.append(path_length_tau_atm(Zi,beta[i]))
        Hp = ak.values_astype(H.snapshot(), np.float32)
        zen = "%f" % zenith[i]
        """
        plt.figure()
        
        plt.plot(Zi, Hp[i] , label = 'Zenith ='+zen)
        plt.xlabel('height, km')
        plt.ylabel('distance, km')
        plt.title('Distance on shower axis versus altitude')
        # plt.plot(x[0:max_pos*2],GH(x[0:max_pos*2],*init),label='GH initial fit')
        plt.legend()
        kurs = "%i.png" % i
        plt.savefig('distance_to_height_'+kurs, format='png')
        plt.close()
        """
        init = np.empty(6)
        max_pos = np.argmax(y)

        """if x.size>1000:
            k+=1
            print(max_pos,x.size)"""
        #print('Xmax position ',max_pos,' Xlength ', x.size ,' RNlength ', y.size, 'Xfirst ',Xfirst[i], 'Xmax ', x[max_pos])
        polyopt = np.polyfit(x[max_pos - 1:max_pos + 2], y[max_pos - 1:max_pos + 2], 2)
        #init[1] = -polyopt[1] / 2 / polyopt[0]
        init[1]=x[max_pos]
        init[0]=init[1]-800
        #init[2] = np.polyval(polyopt, init[1])
        init[2]=y[max_pos]
        init[3:] = [0.000002, -0.001, 45]
        if init[1]>4000:
            init[3:] = [0, 0, 44]
        bounds = ([-1000, 0, 0, 0, -100, 0.1], [x[max_pos],  np.inf, np.inf, 10, 100, 10000])
        popt, pcov = curve_fit(GH, x[0:max_pos*2], y[0:max_pos*2], p0=init,maxfev=1000000)
        yfit = GH(x[0:max_pos*2], *popt)
        for j in range((y[0:max_pos*2]).size):
            chi2[i] += (y[j] - yfit[j]) ** 2 / (y[j])
        chi2[i] =chi2[i] / (nX[i] - 6)
        if math.isnan(chi2[i]):
            chi2[i]=1
        if False:#x.size>2000:#chi2[i]>1 or math.isnan(chi2[i]):#max_pos==x.size-1:#
            k+=1
            #print(max_pos,x.size)
            #print('Fit params, X0, Xmax, Nmax, p3, p2, p1(lambda) ', popt)
            plt.figure()
            plt.plot(x,y*1e5,label='Profile, zenith= '+zen)
            plt.xlabel('slant depth, gcm^2')
            plt.ylabel('Particle count')
            plt.title('Profile as a function of slant depth, dmax= '+"%f" % Hp[i][max_pos])
            #plt.plot(x[0:max_pos*2],GH(x[0:max_pos*2],*init),label='GH initial fit')
            plt.legend()
            kurs = "%i.png" % i
            plt.savefig('profile_vs_dist_'+kurs, format='png')
            plt.close()
        #else:
        #    print('Good fit, Chi2 ',chi2[i], ' popt ',popt )

        X0[i]=popt[0]
        Xmax[i]=popt[1]
        heightMax[i]=Z[i][max_pos]
        lenMax[i]=Hp[i][max_pos]
        Nmax[i]=popt[2]*1e5
        p3[i]=popt[3]
        p2[i]=popt[4]
        p1[i]=popt[5]
        Xmx[i]=x[max_pos]
        Nmx[i]=y[max_pos]*1e5
    #print('Number of profiles >1000', k)
    """
    plt.hist(heightMax,100)
    plt.xlabel('height (km)')
    plt.ylabel('Counts')
    plt.grid(visible=True)
    plt.xlim(0,10)
    #plt.yscale('log')
    kurs = "%i" % Xfirst.size
    plt.title('Histogram of Xmax altitude, n='+kurs)
    plt.savefig('Hist_heightMax')
    plt.figure()
    """
    data_list = [('Nmax \t \t \t \t', Nmx),('lenDec \t \t \t \t', (data["lenDec"][mask])[mask2]), ('lenMax \t \t \t \t', lenMax),('altDec \t \t \t \t \t', (data["altDec"][mask])[mask2]), ('HeightMax \t \t \t \t ', heightMax), ('ShowerEnergy \t \t \t \t ', (data["tauEnergy"][mask])[mask2]), ('Zenith \t \t \t \t ', zenith),
                 ('azimuth \t \t \t', azim)]

    np.savetxt('data.txt', np.column_stack([data for name, data in data_list]),
               header=' '.join([name for name, data in data_list]))
    nH = ak.num(H, axis=1)
    nH = ak.to_numpy(nH)
    print('Las RN ', nRN)
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
        #, "nEGround": nEGround
        #, "EGround": ('f4', (3,))
    }
    now = datetime.datetime.now()  # get the current date and time
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  # format the date and time as a string

    file_name = f"nss_conex_{timestamp}.root"
    f = uproot.recreate("nss_to_conex.root")

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
        , "SvnRevision": [0]
        # ,"SvnRevision":np.dtype('string')
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
        "lgE": TauEnergy       #TauEnergy
        , "zenith": zenith  #90+np.degree(beta_rad)
        , "azimuth": azim #phiTrSubV
        , "Seed2": intn
        , "Seed3": intn
        , "Xfirst": Xfirst
        , "Hfirst": Zfirst*1000  #altDec in m
        , "XfirstIn": ak.ones_like(nan4n)*0.5#nan4n  !!!!!!!!!!!
        , "altitude": ak.zeros_like(nan8n)*0.5#nan8n !!!!!!!!!!!
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
        , "dEdXmx": Nmx*0.0026
        , "cpuTime": nan4n
        #, "nX": nX
        , "X": X
        , "N": RN
        , "H": Z*1000 #in meters
        , "D": Hp*1000
        , "dEdX": RN*0.0026#https://doi.org/10.1016/S0927-6505(00)00101-8 in GeV // 5*10^6 dEdXmax for 10^18.5 showers. (consistently getting 1.75*10^5 at Xmax)
        , "Mu": RN #Xempty
        , "Gamma": RN #Xempty
        , "Electrons": RN
        , "Hadrons": RN  #Xempty
        , "dMu": RN/3333 #Xempty
        #, "EGround": Eg

    })
    shutil.copyfile("nss_to_conex.root",file_name)
    header = f["Header"]
    shower = f["Shower"]
    f.close()
    os.remove("showerdata.csv")