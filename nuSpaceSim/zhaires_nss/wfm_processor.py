import numpy as np
from scipy.optimize import curve_fit
import os as os
import matplotlib.pyplot as plt

def load_file(fname = 'ZHAireS_wfm_z_50_h_0.npz'):
    f = np.load(fname)
    
    time = f['time']
    antView = f['ant_view_ang']
    Afield = f['Afield']
    Efield = f['Efield']
    return time, antView, Afield, Efield

def process_wfms(time, efield):
    dt = time[1]-time[0]
    fs = 1./dt

    Nant = efield.shape[1]
    deltaF = 10
    #freqSubset = np.arange(30., 1050., deltaF)
    freqSubset = np.arange(0., 1650., deltaF)
    N_frq = len(freqSubset)

    subS = np.zeros((Nant, N_frq, 2))

    maxE = []
    maxS = []
    peak_t = []


    for ant in range(0,Nant):
        peak_t.append(time[np.argmax(np.abs(efield[0,ant,1,:]))])
        E_fft = np.fft.rfft(efield[0,ant,1,:])
        N = len(np.abs(E_fft))
        freq = np.arange(0., N, 1.) * fs/float(len(time))
        maxE.append(np.max(np.abs(efield[0,ant,1,:])))
        Ns = len(efield[0,ant,1,:])
        df = freq[1] - freq[0]
        maxS.append(np.sum(np.abs(E_fft[1:]))*df/2.)

        for f in range(0, N_frq):
            cut = np.logical_and(freq*1e3>=freqSubset[f], freq*1e3<freqSubset[f]+deltaF)
            v1 = np.sum(np.abs(E_fft[cut]))*df
            E_cut = np.fft.irfft(E_fft[cut])
            v2 = np.max(np.abs(E_cut))*len(E_cut)*df
            subS[ant, f, 0] = v1
            subS[ant, f, 1] = v2

    return freqSubset, subS

def airshower_beam_func(psi, E_0, gauss_peak, gauss_width, E_1, width2, frac_gauss=1.):
    v = (psi - gauss_peak) / gauss_width
    E_field = E_0 * (frac_gauss*np.exp(-v**2/2.) \
            + (1-frac_gauss)/(1+v**2)) \
            + np.abs(E_1)*np.exp(-psi**2/2./width2**2)
    return E_field
def airshower_gauss_func(psi, E_0, gauss_peak, gauss_width, frac_gauss=1., E_1=0., width2=1e-9):
    return airshower_beam_func(psi, E_0, gauss_peak, gauss_width, E_1=0., width2 = 1e-9, frac_gauss = 1.)
def airshower_gauss_lor_func(psi, E_0, gauss_peak, gauss_width, frac_gauss, E_1=0., width2=1e-9):
    return airshower_beam_func(psi, E_0, gauss_peak, gauss_width, E_1=0., width2 = 1e-9, frac_gauss = frac_gauss)

def fit_subsets(freqSubset, subS, ant_view_ang, zenith, decay_h):
    E0_list     = []
    peak_list   = []
    width_list  = []
    E_1_list    = []
    width2_list = []
    freq_list    = []
    
    min_ang = np.argmin(ant_view_ang)+1
    
    N_frq = len(freqSubset)
    ref_med = -1.
    ref_MAD = -1.
    cc = -1
    for k in range(0, N_frq):
        cc+= 1
        id_pk = np.argmax(subS[:,k,0])
        med = np.median(subS[:,k,0])
        MAD = np.median(np.abs(subS[:,k,0]-med))
        mnm = np.min(subS[:,k,0])
        mxm = np.max(subS[:,k,0])
        if cc == 0:
            ref_max = mxm
            ref_med = med
            ref_MAD = MAD
        #cut = subS[:,k,0] > ref_med + 7.5*1.4826*ref_MAD
        cut = subS[:,k,0] > 0.1 * mxm
        if mxm < 5e-7:
            cut = subS[:,k,0] > mxm/3.
        if mxm < 3e-7:
            freq_list.append(freqSubset[k] + 5.)
            E0_list.append(0.)
            peak_list.append(0.)
            width_list.append(1e-9)
            E_1_list.append(0.)
            width2_list.append(1e-9)
            continue
        #cut = abs(subS[:,k,0]) > 0
        #if np.sum(cut)<10: continue
        print(np.sum(cut), zenith, decay_h, freqSubset[k])
        
        E_0  = mxm    # V/m at ground level for 10^18 eV tau lepton.
        gauss_peak = ant_view_ang[:][id_pk]      # degree. 
        hwhm_arg = np.argmin(np.abs(subS[:,k,0][cut] - 0.5*E_0))
        HWHM = np.abs(ant_view_ang[:][id_pk] - ant_view_ang[:][cut][hwhm_arg])
        gauss_width = 2.*HWHM/2.355
        E_1 = 0.
        width2 = 1e-9          # degrees
        central_distrib_flag = False
        if np.min(np.abs(ant_view_ang[:][cut]))< 0.5 and ant_view_ang[np.argmax(subS[:,k,0])]>0.5:
            truearray = np.where(cut==True)
            leftside = np.argmax(subS[:,k,0])-truearray[0][0]
            rightside = np.abs(np.argmax(subS[:,k,0])-truearray[0][-1])
            if leftside-5 > rightside:
                min_argum = np.argmin(np.abs(ant_view_ang))
                E_1 = 0.1*subS[min_argum,k,0]
                width2 = 0.5
                central_distrib_flag = True
        frac_gauss = 1. # fraction of Gaussian peak ( 1-frac_gauss is lorentzian)
        p0_g = [E_0, gauss_peak, gauss_width] # initial parameters for the curve fit.
        p0_gl = [E_0, gauss_peak, gauss_width, frac_gauss] # initial parameters for the curve fit.
        p0_bm = [E_0, gauss_peak, gauss_width, E_1, width2] # initial parameters for the curve fit.
        popt, pcov = curve_fit(airshower_gauss_func, ant_view_ang[:][cut], subS[:,k,0][cut], p0=p0_g, maxfev = 1000000)
        if central_distrib_flag:
            p0_bm = [*popt, E_1, width2] # initial parameters for the curve fit.
            p0bounds =  ([-np.inf,-np.inf,-np.inf,0,0],[np.inf, np.inf,np.inf, np.inf, 1])
            p0_bm = np.array(p0_bm)*np.random.normal(1., 1.e-1, len(p0_bm))
            popt, pcov = curve_fit(airshower_beam_func, ant_view_ang[:][cut], subS[:,k,0][cut], p0=p0_bm, maxfev = 1000000, bounds = p0bounds)
         
        freq_list.append(freqSubset[k] + 5.)
        E0_list.append(popt[0])
        peak_list.append(popt[1])
        width_list.append(popt[2])
        if central_distrib_flag:
            E_1_list.append(popt[3])
            width2_list.append(popt[4])
        if not central_distrib_flag:
            E_1_list.append(0.)
            width2_list.append(1e-9)
        #if zenith == 60 and decay_h == 6:
        #    plt.semilogy(ant_view_ang[:], subS[:,k,0], 'ko')
        #    plt.semilogy(ant_view_ang[:][cut], subS[:,k,0][cut], 'b-')
        #    if not central_distrib_flag:
        #        plt.plot(ant_view_ang[:], airshower_gauss_func(ant_view_ang[:],*popt), 'r-')
        #    if central_distrib_flag:
        #        plt.plot(ant_view_ang[:], airshower_beam_func(ant_view_ang[:],*popt), 'r-')
        #    plt.ylim(bottom=1e-10)
        #    plt.show()
    return np.column_stack((freq_list, E0_list, peak_list, width_list, E_1_list, width2_list))

def wfmProcessor():
    dirname = '/home/abl/nuspacesim/nuSpaceSim/zhaires_nss/wfms/'
    params = []
    z = []
    h = []
    for fname in os.listdir(dirname):
        zenith = int(fname.split('_')[3])
        decay_h = int(fname.split('_')[5][0])
        time, antView, Afield, Efield = load_file(dirname + fname)
        freqSubset, subS = process_wfms(time, Efield)
        ps = fit_subsets(freqSubset, subS, antView, zenith, decay_h)
        if len(ps) > 0:
            z.append(zenith)
            h.append(decay_h)
            params.append(ps)
    np.savez("/home/abl/nuspacesim/nuSpaceSim/zhaires_nss/params/ZHAireS_params_10MHz_bins.npz", zenith = z, height = h, params=params)

#wfmProcessor()
