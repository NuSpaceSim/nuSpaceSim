import numpy as np
import os as os
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files


def dispersion_phases(freq, TEC, Bcos_theta=50.0e-6):
    alpha = 1.3445e9  # (Hz/TECU)
    em_ratio = 1.759e11  # Hz / Tesla, electron charge to mass ratio
    c = 299792458.0
    C2 = alpha * TEC
    C3 = alpha * TEC * 2.0 * em_ratio * Bcos_theta
    C4a = 3.0 / 2.0 * alpha ** 2 * TEC ** 2 * c / (250.0e3)
    C4b = 3.0 / 2.0 * C2 * 3.0 * (Bcos_theta * em_ratio) ** 2
    phase2 = np.zeros(len(freq))
    phase3 = np.zeros(len(freq))
    phase4a = np.zeros(len(freq))
    phase4b = np.zeros(len(freq))
    phase2[1:] = C2 / (freq[1:])
    phase3[1:] = C3 / (freq[1:] ** 2)
    phase4a[1:] = C4a / (freq[1:] ** 3)
    phase4b[1:] = C4b / (freq[1:] ** 3)
    return phase2, phase3, phase4a, phase4b


def disperse(wfm_h, wfm_v, fs, TEC, Bcos_theta=50.0e-6):
    fft_w_h = np.fft.rfft(wfm_h)
    fft_w_v = np.fft.rfft(wfm_v)
    t = np.arange(len(wfm_h), dtype=np.float64) / fs

    fft_w_x = (fft_w_h - 1j * fft_w_v) / np.sqrt(2.0)
    fft_w_o = (fft_w_h + 1j * fft_w_v) / np.sqrt(2.0)

    N = len(fft_w_x)
    df = fs / N
    freq = np.arange(N, dtype=np.float64) * df

    phase2, phase3, phase4a, phase4b = dispersion_phases(freq, TEC, Bcos_theta)

    fft_w_x[1:] *= np.exp(1j * 2.0 * np.pi * phase2[1:])
    fft_w_x[1:] *= np.exp(1j * 2.0 * np.pi * phase3[1:])
    fft_w_x[1:] *= np.exp(1j * 2.0 * np.pi * (phase4a[1:] + phase4b[1:]))

    fft_w_o[1:] *= np.exp(1j * 2.0 * np.pi * phase2[1:])
    fft_w_o[1:] *= np.exp(-1j * 2.0 * np.pi * phase3[1:])
    fft_w_o[1:] *= np.exp(1j * 2.0 * np.pi * (phase4a[1:] + phase4b[1:]))

    fft_w_h = (fft_w_x + fft_w_o) / np.sqrt(2)
    fft_w_v = 1j * (fft_w_x - fft_w_o) / np.sqrt(2)

    return t, np.fft.irfft(fft_w_h, t.shape[0]), np.fft.irfft(fft_w_v, t.shape[0])


def load_file(fname="ZHAireS_wfm_z_50_h_0.npz"):
    f = np.load(fname)

    time = f["time"]
    antView = f["ant_view_ang"]
    Afield = f["Afield"]
    Efield = f["Efield"]
    return time, antView, Afield, Efield


def process_wfms(time, efield, frange):
    dt = time[1] - time[0]
    fs = 1.0 / dt
    N = len(time)
    df = fs / float(N)
    lf = frange[0]
    hf = frange[1]

    indmax = np.unravel_index(efield.argmax(), efield.shape)
    indmin = np.unravel_index(efield.argmax(), efield.shape)
    ind = indmax if np.abs(efield[indmax]) > np.abs(efield[indmin]) else indmin
    E_fft = np.fft.rfft(efield[ind[0], ind[1], ind[2], :])
    N = len(np.abs(E_fft))
    freq = np.arange(N, dtype=np.float64) * df

    cut = np.logical_and(freq >= lf, freq <= hf)
    E_fft[~cut] = 0
    E_cut = np.fft.irfft(E_fft, efield.shape[3])
    return E_cut


def wfm(
    f1=30e6, f2=80e6, fs=200e6, delay=3e-6, record_length=4096.0 / 200.0e6, index=1.0
):
    dt = 1.0 / fs
    N = int(record_length / dt)
    if N % 2 != 0:
        N += 1
    df = fs / float(N)
    w = np.arange(N, dtype=np.float64)
    f = df * np.arange(N, dtype=np.float64)
    fft_w = np.fft.rfft(w) * 0.0
    i_f1 = int(f1 / df)
    i_f2 = int(f2 / df)

    fft_w[i_f1:i_f2] = (f[i_f1:i_f2]) ** index
    fft_w[i_f1:i_f2] *= np.exp(-1j * 2.0 * np.pi * delay * f[i_f1:i_f2])
    w = np.fft.irfft(fft_w)
    norm = np.max(np.abs(w))

    w /= norm
    f = df * np.arange(N, dtype=np.float64)
    t = dt * np.arange(N, dtype=np.float64)
    return t, w


def noise_fake(f1, f2, fs=200.0e6, record_length=4096 / 200.0e6, index=-2.6):
    dt = 1.0 / fs
    N = int(record_length / dt)
    if N % 2 != 0:
        N += 1
    df = fs / N
    w = 0.0 * np.arange(N, dtype=np.float64)
    n_fft = np.fft.rfft(w)
    f = np.arange(0, len(n_fft), 1) * fs / N
    r_re = np.random.normal(0.0, 1.0, len(n_fft))
    r_im = np.random.normal(0.0, 1.0, len(n_fft))
    i_f1 = int(f1 / df)
    i_f2 = int(f2 / df)
    n_fft[1:] = (r_re[1:] + 1j * r_im[1:]) * (f[1:]) ** index
    n_fft[0:i_f1] *= 0.0
    n_fft[i_f2 : len(f) - 1] *= 0.0

    w_noise = np.real(np.fft.irfft(n_fft, N))
    w_noise /= np.sqrt(np.cumsum(w_noise ** 2)[len(w) - 1] / len(w))
    return w_noise


def noise(f1, f2, fs, N, index):
    dt = 1.0 / fs
    df = fs / N
    w = 0.0 * np.arange(N, dtype=np.float64)
    n_fft = np.fft.rfft(w)
    f = np.arange(len(n_fft), dtype=np.float64) * df
    r_re = np.random.normal(0.0, 1.0, len(n_fft))
    r_im = np.random.normal(0.0, 1.0, len(n_fft))
    i_f1 = int(f1 / df)
    i_f2 = int(f2 / df)
    n_fft[1:] = (r_re[1:] + 1j * r_im[1:]) * (f[1:]) ** index
    n_fft[0:i_f1] *= 0.0
    n_fft[i_f2 : len(f) - 1] *= 0.0

    w_noise = np.real(np.fft.irfft(n_fft, N))
    w_noise /= np.sqrt(np.cumsum(w_noise ** 2)[len(w) - 1] / len(w))
    return w_noise


def conv_circ(signal, kernel):
    return np.real(np.fft.ifft(np.fft.fft(signal) * np.fft.fft(kernel)))


def TEC_search(
    f1=30.0,
    f2=300.0,
    TEC=10.0,
    Bcos_theta=50.0e-6,
    SNRa=10,
    Nscan=11,
    dTEC=0.05,
    polang=-999,
):
    dirname = "/home/abl/nuspacesim/src/nuspacesim/simulation/eas_radio/zhaires/wfms/"
    frange = (f1, f2)
    for fname in os.listdir(dirname):
        zenith = int(fname.split("_")[3])
        decay_h = int(fname.split("_")[5][0])
        t, antView, Afield, Efield = load_file(dirname + fname)
        t *= 1e-9
        subS = process_wfms(t, Efield, frange)
        # norm = np.max(subS) - np.min(subS)
        norm = np.max(np.abs(subS))
        dt = t[1] - t[0]
        fs = 1.0 / dt
        N = len(np.fft.rfft(subS))
        df = fs / N
        freq = np.arange(N, dtype=np.float64) * df
        subS *= SNRa / norm
        pang = np.radians(polang)
        Nt = t.shape[0]
        if polang == -999:
            pang = np.random.uniform(-np.pi / 2.0, np.pi / 2.0)
        w2_h = np.cos(pang) * subS.copy()
        w2_v = np.sin(pang) * subS.copy()
        t3, w3_h, w3_v = disperse(w2_h, w2_v, fs, TEC, Bcos_theta)
        wn_h = noise(f1, f2, fs, Nt, -2.6)
        wn_v = noise(f1, f2, fs, Nt, -2.6)
        wn_h /= np.std(wn_h)
        wn_v /= np.std(wn_v)
        w3_h = w3_h + wn_h
        w3_v = w3_v + wn_v
        w3_h_fft = np.fft.rfft(w3_h)
        w3_v_fft = np.fft.rfft(w3_v)
        tTEC_list = np.arange(
            TEC - Nscan / 2.0 * dTEC, TEC + (Nscan / 2.0 + 1e-7) * dTEC, dTEC
        )
        if Nscan == 1:
            tTEC_list = [TEC]
        # if TEC not in tTEC_list:
        #    tTEC_list = np.insert(tTEC_list, int(np.ceil(Nscan/2)), TEC)
        v1 = []
        v2 = []

        for tTEC in tTEC_list:
            ph2, ph3, ph4a, ph4b = dispersion_phases(freq, tTEC, Bcos_theta)
            fft_wX = np.exp(-1j * 2 * np.pi * (ph2 + ph3 + ph4b)) / np.sqrt(2.0)
            fft_wO = np.exp(-1j * 2 * np.pi * (ph2 - ph3 + ph4b)) / np.sqrt(2.0)
            wX = np.fft.irfft(fft_wX, Nt)
            wO = np.fft.irfft(fft_wO, Nt)

            wP = np.fft.irfft((fft_wX + fft_wO), Nt) / np.sqrt(2.0)
            wM = np.fft.irfft(-1j * (fft_wX - fft_wO), Nt) / np.sqrt(2.0)

            corr_Xh = conv_circ(w3_h, wX)
            corr_Oh = conv_circ(w3_h, wO)
            corr_Xv = conv_circ(w3_v, wX)
            corr_Ov = conv_circ(w3_v, wO)
            corr_Ph = conv_circ(w3_h, wP)
            corr_Mh = conv_circ(w3_h, wM)
            corr_Pv = conv_circ(w3_v, wP)
            corr_Mv = conv_circ(w3_v, wM)

            v1.append(
                np.sqrt(
                    np.max(corr_Xh ** 2 + corr_Oh ** 2 + corr_Xv ** 2 + corr_Ov ** 2)
                )
            )
            v2.append(
                np.max(
                    np.sqrt(corr_Ph ** 2 + corr_Mh ** 2 + corr_Pv ** 2 + corr_Mv ** 2)
                )
            )
            # plt.figure(22)
            # plt.plot([tTEC], [np.max(np.sqrt(corr_Ph**2 + corr_Mh**2 + corr_Pv**2 + corr_Mv**2))], 'go', mfc='none')
            # plt.xlabel('TECU')
            # plt.ylabel('Amplitude')
            # plt.tight_layout()
        # plt.show()
        # return np.max(v1), np.max(v2)
        return np.array(tTEC_list), np.array(v2)


def bfield_search(
    f1=30.0,
    f2=300.0,
    TEC=10.0,
    Bcos_theta=50.0e-6,
    SNRa=10,
    Nscan=11,
    dBF=0.05,
    polang=-999,
):
    dirname = "/home/abl/nuspacesim/src/nuspacesim/simulation/eas_radio/zhaires/wfms/"
    frange = (f1, f2)
    for fname in os.listdir(dirname):
        zenith = int(fname.split("_")[3])
        decay_h = int(fname.split("_")[5][0])
        t, antView, Afield, Efield = load_file(dirname + fname)
        t *= 1e-9
        subS = process_wfms(t, Efield, frange)
        norm = np.max(np.abs(subS))
        subS /= norm
        dt = t[1] - t[0]
        fs = 1.0 / dt
        pang = np.radians(polang)
        Nt = t.shape[0]
        if polang == -999:
            pang = np.random.uniform(-np.pi / 2.0, np.pi / 2.0)
        w2_h = np.cos(pang) * subS.copy()
        w2_v = np.sin(pang) * subS.copy()
        t3, w3_h, w3_v = disperse(w2_h, w2_v, fs, TEC, Bcos_theta)
        wn_h = noise(f1, f2, fs, Nt, -2.6)
        wn_v = noise(f1, f2, fs, Nt, -2.6)
        wn_h /= np.std(wn_h)
        wn_v /= np.std(wn_v)
        w3_h = SNRa * w3_h + wn_h
        w3_v = SNRa * w3_v + wn_v
        bfield_list = np.arange(
            Bcos_theta - Nscan / 2.0 * dBF, Bcos_theta + (Nscan / 2.0 + 1e-9) * dBF, dBF
        )
        # if Bcos_theta not in bfield_list:
        #    bfield_list = np.insert(bfield_list, int(np.ceil(Nscan/2)), Bcos_theta)
        v1 = []
        v2 = []

        for B in bfield_list:
            N = len(np.fft.rfft(w3_h))
            df = fs / N
            freq = np.arange(N, dtype=np.float64) * df
            ph2, ph3, ph4a, ph4b = dispersion_phases(freq, TEC, B)
            fft_wX = np.exp(-1j * 2 * np.pi * (ph2 + ph3 + ph4b)) / np.sqrt(2.0)
            fft_wO = np.exp(-1j * 2 * np.pi * (ph2 - ph3 + ph4b)) / np.sqrt(2.0)
            wX = np.fft.irfft(fft_wX, t.shape[0])
            wO = np.fft.irfft(fft_wO, t.shape[0])

            wP = np.fft.irfft((fft_wX + fft_wO), Nt) / np.sqrt(2.0)
            wM = np.fft.irfft(-1j * (fft_wX - fft_wO), Nt) / np.sqrt(2.0)

            corr_Xh = conv_circ(wX, w3_h)
            corr_Oh = conv_circ(wO, w3_h)
            corr_Xv = conv_circ(wX, w3_v)
            corr_Ov = conv_circ(wO, w3_v)
            corr_Ph = conv_circ(wP, w3_h)
            corr_Mh = conv_circ(wM, w3_h)
            corr_Pv = conv_circ(wP, w3_v)
            corr_Mv = conv_circ(wM, w3_v)

            v1.append(
                np.sqrt(
                    np.max(corr_Xh ** 2 + corr_Oh ** 2 + corr_Xv ** 2 + corr_Ov ** 2)
                )
            )
            v2.append(
                np.max(
                    np.sqrt(corr_Ph ** 2 + corr_Mh ** 2 + corr_Pv ** 2 + corr_Mv ** 2)
                )
            )
            plt.figure(22)
            plt.plot(
                [B * 1e6],
                [
                    np.max(
                        np.sqrt(
                            corr_Ph ** 2 + corr_Mh ** 2 + corr_Pv ** 2 + corr_Mv ** 2
                        )
                    )
                ],
                "go",
                mfc="none",
            )
            plt.xlabel("Bcos_theta (uTesla)")
            plt.ylabel("Amplitude")
            plt.tight_layout()
        plt.show()
        return np.max(v1) / np.sqrt(2), np.max(v2) / np.sqrt(2)


def TEC_search_fakewf(
    f1=30.0,
    f2=300.0,
    TEC=10.0,
    Bcos_theta=50.0e-6,
    SNRa=10,
    Nscan=11,
    dTEC=0.05,
    polang=-999,
):
    fs = 200e6
    t, subS = wfm(f1, f2)
    norm = np.max(np.abs(subS))
    subS *= SNRa / norm
    pang = np.radians(polang)
    if polang == -999:
        pang = np.random.uniform(-np.pi / 2.0, np.pi / 2.0)
    w2_h = np.cos(pang) * subS.copy()
    w2_v = np.sin(pang) * subS.copy()
    t3, w3_h, w3_v = disperse(w2_h, w2_v, 200e6, TEC, Bcos_theta)
    wn_h = noise_fake(f1, f2)
    wn_v = noise_fake(f1, f2)
    wn_h /= np.std(wn_h)
    wn_v /= np.std(wn_v)
    w3_h = w3_h + wn_h
    w3_v = w3_v + wn_v
    tTEC_list = np.arange(
        TEC - Nscan / 2.0 * dTEC, TEC + (Nscan / 2.0 + 1e-3) * dTEC, dTEC
    )
    if Nscan == 1:
        tTEC_list = [TEC]
    v1 = []
    v2 = []

    for tTEC in tTEC_list:
        N = len(np.fft.rfft(w3_h))
        df = fs / N
        freq = np.arange(0.0, N, 1.0, dtype=np.float64) * df

        ph2, ph3, ph4a, ph4b = dispersion_phases(freq, tTEC, Bcos_theta)
        fft_wX = np.exp(-1j * 2 * np.pi * (ph2 + ph3 + ph4b)) / np.sqrt(2.0)
        fft_wO = np.exp(-1j * 2 * np.pi * (ph2 - ph3 + ph4b)) / np.sqrt(2.0)
        wX = np.fft.irfft(fft_wX)
        wO = np.fft.irfft(fft_wO)

        wP = np.fft.irfft((fft_wX + fft_wO)) / np.sqrt(2.0)
        wM = np.fft.irfft(-1j * (fft_wX - fft_wO)) / np.sqrt(2.0)

        corr_Xh = conv_circ(wX, w3_h)
        corr_Oh = conv_circ(wO, w3_h)
        corr_Xv = conv_circ(wX, w3_v)
        corr_Ov = conv_circ(wO, w3_v)
        corr_Ph = conv_circ(wP, w3_h)
        corr_Mh = conv_circ(wM, w3_h)
        corr_Pv = conv_circ(wP, w3_v)
        corr_Mv = conv_circ(wM, w3_v)

        v1.append(
            np.sqrt(np.max(corr_Xh ** 2 + corr_Oh ** 2 + corr_Xv ** 2 + corr_Ov ** 2))
        )
        v2.append(
            np.max(np.sqrt(corr_Ph ** 2 + corr_Mh ** 2 + corr_Pv ** 2 + corr_Mv ** 2))
        )
        plt.figure(22)
        plt.plot(
            [tTEC],
            [
                np.max(
                    np.sqrt(corr_Ph ** 2 + corr_Mh ** 2 + corr_Pv ** 2 + corr_Mv ** 2)
                )
            ],
            "go",
            mfc="none",
        )
        plt.xlabel("TECU")
        plt.ylabel("Amplitude")
        plt.tight_layout()
    plt.show()
    return np.max(v1) / np.sqrt(2), np.max(v2) / np.sqrt(2)


def ionospheric_dispersion(flow=30.0e6, fhigh=300.0e6, TEC=10.0, Bcos_theta=50.0e-6):
    """doc"""

    dirname = "/home/abl/nuspacesim/src/nuspacesim/simulation/eas_radio/zhaires/wfms/"
    frange = (flow, fhigh)
    for fname in os.listdir(dirname):
        zenith = int(fname.split("_")[3])
        decay_h = int(fname.split("_")[5][0])
        t, antView, Afield, Efield = load_file(dirname + fname)
        t *= 1e-9
        subS = process_wfms(t, Efield, frange)
        # plt.plot(t, subS)
        # plt.show()
        dt = t[1] - t[0]
        fs = 1.0 / dt
        w2_h = 0.0 * subS.copy()
        w2_v = subS.copy()
        t3, w3_h, w3_v = disperse(w2_h, w2_v, fs, TEC, Bcos_theta)
        plt.plot(t3, w3_v, alpha=0.5)
        plt.plot(t3, w3_h, alpha=0.5)
        plt.figure()
        f, t, Sxx = signal.spectrogram(w3_h, fs)
        plt.pcolormesh(t, f, np.log10(Sxx), vmin=np.log10(np.max(Sxx)) - 2.0)
        plt.ylabel("Freq (Hz)")
        plt.xlabel("Time")
        plt.figure()
        f, t, Sxx = signal.spectrogram(w3_v, fs, nperseg=2 ** 6)
        plt.pcolormesh(t, f, np.log10(Sxx), vmin=np.log10(np.max(Sxx)) - 2.0)
        plt.ylabel("Freq (Hz)")
        plt.xlabel("Time")
        plt.show()


def generate_TEC_files(f1, f2, TECerr):
    print("generating TEC file for {}-{} MHz and {} TEC error".format(f1, f2, TECerr))
    ntimes = 10
    ns = 51
    dt = (2.0 * TECerr) / float(ns)
    funcs = np.array([])
    tecs = np.array([])
    for TECu in [1.0, 5.0, 10.0, 50.0, 100.0, 150.0]:
        for i in range(0, ntimes):
            if i == 0:
                teclist, v2 = TEC_search(
                    f1=f1 * 1.0e6,
                    f2=f2 * 1.0e6,
                    TEC=TECu,
                    Bcos_theta=42e-6,
                    dTEC=dt,
                    Nscan=ns,
                    SNRa=100,
                    polang=-999,
                )
            else:
                tl, vv2 = TEC_search(
                    f1=f1 * 1.0e6,
                    f2=f2 * 1.0e6,
                    TEC=TECu,
                    Bcos_theta=42e-6,
                    dTEC=dt,
                    Nscan=ns,
                    SNRa=100,
                    polang=-999,
                )
                v2 += vv2
        v2 /= float(ntimes)
        plt.plot(teclist, v2)
        plt.show()
        func = interp1d(teclist, v2)
        tecs = np.append(tecs, TECu)
        funcs = np.append(funcs, func)
    param_dir = str(files("nuspacesim.simulation.eas_radio.ionosphere.tecparams"))
    fname = param_dir + "/f_{}_{}_TECerr_{}.npz".format(int(f1), int(f2), TECerr)
    np.savez(fname, TEC=tecs, scaling=funcs)
    print("generated TEC file successfully!")


def perfect_TEC_disperse(
    f1=30.0, f2=300.0, TEC=10.0, Bcos_theta=50.0e-6, SNRa=100, polang=-999
):
    dirname = "/home/abl/nuspacesim/src/nuspacesim/simulation/eas_radio/zhaires/wfms/"
    frange = (f1, f2)
    for fname in os.listdir(dirname):
        zenith = int(fname.split("_")[3])
        decay_h = int(fname.split("_")[5][0])
        t, antView, Afield, Efield = load_file(dirname + fname)
        t *= 1e-9
        subS = process_wfms(t, Efield, frange)
        norm = np.max(np.abs(subS))
        dt = t[1] - t[0]
        fs = 1.0 / dt
        N = len(np.fft.rfft(subS))
        df = fs / N
        freq = np.arange(N, dtype=np.float64) * df
        subS *= SNRa / norm
        pang = np.radians(polang)
        Nt = t.shape[0]
        if polang == -999:
            pang = np.random.uniform(-np.pi / 2.0, np.pi / 2.0)
        w2_h = np.cos(pang) * subS.copy()
        w2_v = np.sin(pang) * subS.copy()
        t3, w3_h, w3_v = disperse(w2_h, w2_v, fs, TEC, Bcos_theta)
        wn_h = noise(f1, f2, fs, Nt, -2.6)
        wn_v = noise(f1, f2, fs, Nt, -2.6)
        wn_h /= np.std(wn_h)
        wn_v /= np.std(wn_v)
        w3_h = w3_h + wn_h
        w3_v = w3_v + wn_v
        w3_h_fft = np.fft.rfft(w3_h)
        w3_v_fft = np.fft.rfft(w3_v)

        ph2, ph3, ph4a, ph4b = dispersion_phases(freq, TEC, Bcos_theta)
        fft_wX = np.exp(-1j * 2 * np.pi * (ph2 + ph3 + ph4b)) / np.sqrt(2.0)
        fft_wO = np.exp(-1j * 2 * np.pi * (ph2 - ph3 + ph4b)) / np.sqrt(2.0)
        wP = np.fft.irfft((fft_wX + fft_wO), Nt) / np.sqrt(2.0)
        wM = np.fft.irfft(-1j * (fft_wX - fft_wO), Nt) / np.sqrt(2.0)

        corr_Ph = conv_circ(w3_h, wP)
        corr_Mh = conv_circ(w3_h, wM)
        corr_Pv = conv_circ(w3_v, wP)
        corr_Mv = conv_circ(w3_v, wM)

        v2 = np.max(np.sqrt(corr_Ph ** 2 + corr_Mh ** 2 + corr_Pv ** 2 + corr_Mv ** 2))
        return v2


# a bunch of plots
# TEC = 10.
# dt = 0.05
# SNR_list = 10**np.arange(0, 2.1, 0.2)
# dSNR_list = 10**np.arange(-8, -5, 0.3)
# pk1_m = []
# pk2_m = []
# pk1_s = []
# pk2_s = []
# f1 = 300.
# f2 = 1000.
# ns = 51
# for dSNR in dSNR_list:
#    pk1_list = []
#    pk2_list = []
#    for k in range(0,5):
#        pk1, pk2 = bfield_search(f1=f1*1e6, f2=f2*1.e6, TEC=TEC, Bcos_theta=42e-6,
#                dBF=dSNR,
#                Nscan=5,
#                SNRa=100,
#                polang=-1,
#                )
#        pk1_list.append(pk1)
#        pk2_list.append(pk2)
#    pk1_m.append(np.mean(pk1_list))
#    pk2_m.append(np.mean(pk2_list))
#    pk1_s.append(np.std(pk1_list))
#    pk2_s.append(np.std(pk2_list))
# plt.loglog(dSNR_list*1e6, pk2_m, 'ko-')
# plt.title('band = {} - {} MHz, TEC = {}'.format(int(f1),int(f2), TEC))
# plt.xlabel('BField error (uT)')
# plt.ylabel('measured snr')
# plt.show()
# for TECu in [1., 100.]:
#    pk1, pk2 = bfield_search(f1=f1*1.e6, f2=f2*1.e6, TEC=TECu, Bcos_theta=42e-6,
#            dBF=1e-7,
#            Nscan=101,
#            SNRa=100,
#            polang=-999,
#            )
#
# for TECu in [1., 10.,100.]:
#    pk1, pk2 = TEC_search(f1=f1*1.e6, f2=f2*1.e6, TEC=TECu, Bcos_theta=42e-6,
#            dTEC=0.05,
#            Nscan=101,
#            SNRa=100,
#            polang=-999,
#            )
# for SNR in SNR_list:
#    pk1_list = []
#    pk2_list = []
#    for k in range(0,3):
#        pk1, pk2 = TEC_search(f1=f1*1e6, f2=f2*1.e6, TEC=TEC+1e-3, Bcos_theta=42e-6,
#                dTEC=2*1e-3,
#                Nscan=3,
#                SNRa=SNR,
#                polang=-1,
#                )
#        pk1_list.append(pk1)
#        pk2_list.append(pk2)
#    pk1_m.append(np.mean(pk1_list))
#    pk2_m.append(np.mean(pk2_list))
#    pk1_s.append(np.std(pk1_list))
#    pk2_s.append(np.std(pk2_list))
#    print(SNR, pk1_m[-1], pk2_m[-1])
# plt.loglog(SNR_list, pk2_m, 'ko-')
# plt.title('band = {} - {} MHz'.format(int(f1),int(f2)))
# plt.xlabel('input snr')
# plt.ylabel('measured snr')
# plt.show()
