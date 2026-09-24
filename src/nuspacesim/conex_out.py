import numpy as np
import uproot
from scipy.optimize import curve_fit, least_squares

from .simulation.eas_optical.propagation import lexpr, slant_depth


def conex_out(data, RN, z_nodes, X_to_node, output_file):  # noqa: C901
    def GH(X, X0, Xmax, Nmax, p3, p2, p1):
        return (
            Nmax
            * ((X - X0) / (Xmax - X0)) ** ((Xmax - X0) / (p3 * X**2 + p2 * X + p1))
            * np.exp((Xmax - X) / (p3 * X**2 + p2 * X + p1))
        )

    def alpha(s):
        c1 = 47.9511
        c2 = 0.971315
        c3 = 9.23001
        c4 = 2.29587
        c5 = 0.285196
        return (c1 / (c2 + s) ** c3 + c4 + c5 * s) * 0.001

    def showerage(x, xmax):
        return 3 * x / (x + 2 * xmax)

    Zfirst = np.asarray(data["altDec"], dtype=np.float64)
    TauEnergy = np.log10(np.asarray(data["tauEnergy"], dtype=np.float64)) + 9
    RN = np.asarray(RN, dtype=np.float64)
    Zmax = 20
    Zmin = 0
    # TauEnergyMax=18
    # Useful masks (Zfirst masks are necessary between 0 and 20km). To introduce masks you must also change them in simulation/eas_optical/eas.py
    z_mask = np.asarray((Zfirst >= Zmin) & (Zfirst <= Zmax), dtype=bool)
    beta = np.asarray(data["beta_rad"], dtype=np.float64)[z_mask]
    Zfirst = Zfirst[z_mask]
    TauEnergy = TauEnergy[z_mask]
    rn_max_not_reached = RN[:, -1] < np.max(RN[:, :-1], axis=1)
    mask = rn_max_not_reached
    beta = beta[mask]
    Zfirst = Zfirst[mask]
    TauEnergy = TauEnergy[mask]
    beta = np.asarray(beta, dtype=np.float64)
    RN = RN[mask]
    z_nodes = np.asarray(z_nodes, dtype=np.float64)[mask]
    X_to_node = np.asarray(X_to_node, dtype=np.float64)[mask]

    # The optical pipeline's X_to_node starts at the decay point. CONEX X must
    # start at ground, so add the slant depth from ground to each decay point.
    ground_length = lexpr(0.0, beta)
    decay_length = lexpr(Zfirst, beta)
    Xfirst = slant_depth(ground_length, decay_length, beta)

    n = np.size(Zfirst)
    X = np.asarray(X_to_node + Xfirst[:, None], dtype=np.float32)
    Z = np.asarray(z_nodes, dtype=np.float32)
    RN = np.asarray(RN, dtype=np.float32)
    print(f"Number of valid events with Xmax inside atmosphere: {n}")
    # CONEX D is propagation distance measured from ground, in km here.
    D = lexpr(z_nodes, beta[:, None]) - ground_length[:, None]
    profile_size = X_to_node.shape[-1]

    azim = 360 * np.random.rand(n)  # Random azimuth
    zenith = 90 + np.degrees(beta)
    # Force file name to end in .root and start with conex
    if not output_file.endswith(".root"):
        # strip any existing extension and add .root
        output_file = ".".join(output_file.split(".")[:-1]) + ".root"
    if not output_file.startswith("conex"):
        output_file = "conex_" + output_file
    # Useful variables to fill the conex File
    PID = np.array([100], dtype="i4")  # Proton type for Conex
    zmin = np.array([90], dtype="i4")
    zmax = np.array([132], dtype="i4")
    nan4 = np.array([np.nan], dtype="f4")
    nan8 = np.array([np.nan], dtype="f8")
    int4 = np.array([-1], dtype="i4")
    intn = np.full(n, -1, dtype="i4")
    nan4n = np.full(n, np.nan, dtype="f4")
    nan8n = np.full(n, np.nan, dtype="f8")
    Xempty = np.zeros_like(X)
    OutputVersion = np.array([2.51], dtype="f4")
    a = np.full((1, 31), np.nan, dtype="f8")
    Eground = np.full((n, 3), nan4)
    Eg = Eground

    # Initialize some variables for GH fit
    Xmax = np.empty(n, dtype="f4")
    Nmax = np.empty(n, dtype="f4")
    X0 = np.empty(n, dtype="f4")
    p1 = np.empty(n, dtype="f4")
    p2 = np.empty(n, dtype="f4")
    p3 = np.empty(n, dtype="f4")
    chi2 = np.zeros(n, dtype="f4")
    Xmx = np.empty(n, dtype="f4")
    Nmx = np.empty(n, dtype="f4")
    dEdXmx = np.empty(n, dtype="f4")
    dEdX_profiles = []

    for i in range(n):
        x_profile = np.array(X[i])
        # Shift profile in X and reduce magnitude in N to simplify the fits.
        x0 = x_profile[0]
        x = x_profile - x0
        rn = np.array(RN[i])
        y = rn / 1e5

        init = np.empty(6)
        max_pos = np.argmax(y)
        y = y[
            0 : max_pos * 2
        ]  # Only interested in profile around the maximum, disregard the tail
        x = x[0 : max_pos * 2]

        # Best initial values for a good, fast and reliable fit.
        init[1] = x[max_pos]
        init[0] = -0.30943336 * init[1]
        init[2] = y[max_pos]
        init[3:] = [1e-7, 4e-4, 44]

        fit_succeeded = True
        try:
            popt, pcov = curve_fit(GH, x, y, p0=init, maxfev=1000000)
            if not np.all(np.isfinite(popt)):
                raise RuntimeError("curve_fit returned non-finite parameters")
        except (RuntimeError, ValueError, FloatingPointError) as error:
            print(f"Warning: fit failed for event {i}: {error}")
            try:
                fallback = least_squares(
                    lambda parameters: GH(x, *parameters) - y,
                    init,
                    max_nfev=1000000,
                )
                popt = fallback.x
                if not np.all(np.isfinite(popt)):
                    raise RuntimeError("least_squares returned non-finite parameters")
                print(f"Using best-effort least-squares parameters for event {i}.")
            except (RuntimeError, ValueError, FloatingPointError) as fallback_error:
                print(f"Warning: no fit found for event {i}: {fallback_error}")
                fit_succeeded = False
                popt = np.full(6, np.nan)

        yfit = GH(x, *popt) if fit_succeeded else np.full_like(y, np.nan)
        fit_xmax = popt[1] if fit_succeeded else x[max_pos]
        dedx = alpha(showerage(x_profile, fit_xmax + x0)) * rn
        dEdX_profiles.append(dedx)

        # Calculate chi**2
        if fit_succeeded:
            for j in range(yfit.size):
                if y[j] > 0:
                    chi2[i] += (y[j] - yfit[j]) ** 2 / y[j]

            chi2[i] = chi2[i] / (y.size - 6) / (np.sqrt(popt[2] * 1e5))
        else:
            chi2[i] = np.nan

        g3 = popt[3]
        g2 = popt[4]
        g1 = popt[5]
        if np.isnan(chi2[i]):
            print(f"Warning: chi2 is NaN for event {i}.")
        elif chi2[i] < 0:
            print(f"Warning: chi2 is negative for event {i}: {chi2[i]:.3g}.")
        elif chi2[i] > 0.1:
            print(
                f"Warning: chi2 for event {i} is too high ({chi2[i]:.3f}). "
                "This may indicate a poor fit."
            )
        # Undo the variable change. For p1, p2, p3 this involves shifting the parabola coefficients.
        X0[i] = popt[0] + x0
        Xmax[i] = popt[1] + x0
        Nmax[i] = popt[2] * 1e5
        p3[i] = popt[3]
        p2[i] = g2 - 2 * g3 * x0
        p1[i] = g1 - g2 * x0 + g3 * x0**2
        Xmx[i] = x[max_pos] + x0
        Nmx[i] = rn[max_pos]
        dEdXmx[i] = dedx[max_pos]
        chi2[i] = chi2[i] * 1e5

    nan_chi2 = np.flatnonzero(np.isnan(chi2))
    negative_chi2 = np.flatnonzero(chi2 < 0)
    if nan_chi2.size:
        print(f"Warning: NaN chi2 values found for events {nan_chi2.tolist()}.")
    if negative_chi2.size:
        print(
            f"Warning: negative chi2 values found for events "
            f"{negative_chi2.tolist()}."
        )

    branches_header = {
        "Seed1": np.dtype("i4"),
        "Particle": np.dtype("i4"),
        "Alpha": np.dtype("f8"),
        "lgEmin": np.dtype("f8"),
        "lgEmax": np.dtype("f8"),
        "zMin": np.dtype("f8"),
        "zMax": np.dtype("f8"),
        "SvnRevision": np.dtype("b"),
        "Version": np.dtype("f4"),
        "OutputVersion": np.dtype("f4"),
        "HEModel": np.dtype("i4"),
        "LEModel": np.dtype("i4"),
        "HiLowEgy": np.dtype("f4"),
        "hadCut": np.dtype("f4"),
        "emCut": np.dtype("f4"),
        "hadThr": np.dtype("f4"),
        "muThr": np.dtype("f4"),
        "emThr": np.dtype("f4"),
        "haCut": np.dtype("f4"),
        "muCut": np.dtype("f4"),
        "elCut": np.dtype("f4"),
        "gaCut": np.dtype("f4"),
        "lambdaLgE": ("f8", (31,)),
        "lambdaProton": ("f8", (31,)),
        "lambdaPion": ("f8", (31,)),
        "lambdaHelium": ("f8", (31,)),
        "lambdaNitrogen": ("f8", (31,)),
        "lambdaIron": ("f8", (31,)),
    }

    branches_shower = {
        "lgE": np.dtype("f4"),
        "zenith": np.dtype("f4"),
        "azimuth": np.dtype("f4"),
        "Seed2": np.dtype("i4"),
        "Seed3": np.dtype("i4"),
        "Xfirst": np.dtype("f4"),
        "Hfirst": np.dtype("f4"),
        "XfirstIn": np.dtype("f4"),
        "altitude": np.dtype("f8"),
        "X0": np.dtype("f4"),
        "Xmax": np.dtype("f4"),
        "Nmax": np.dtype("f4"),
        "p1": np.dtype("f4"),
        "p2": np.dtype("f4"),
        "p3": np.dtype("f4"),
        "chi2": np.dtype("f4"),
        "Xmx": np.dtype("f4"),
        "Nmx": np.dtype("f4"),
        "XmxdEdX": np.dtype("f4"),
        "dEdXmx": np.dtype("f4"),
        "cpuTime": np.dtype("f4"),
        "X": ("f4", (profile_size,)),
        "N": ("f4", (profile_size,)),
        "H": ("f4", (profile_size,)),
        "D": ("f4", (profile_size,)),
        "dEdX": ("f4", (profile_size,)),
        "Mu": ("f4", (profile_size,)),
        "Gamma": ("f4", (profile_size,)),
        "Electrons": ("f4", (profile_size,)),
        "Hadrons": ("f4", (profile_size,)),
        "dMu": ("f4", (profile_size,)),
        "nX": np.dtype("i4"),
        "nN": np.dtype("i4"),
        "nH": np.dtype("i4"),
        "nD": np.dtype("i4"),
        "ndEdX": np.dtype("i4"),
        "nMu": np.dtype("i4"),
        "nGamma": np.dtype("i4"),
        "nElectrons": np.dtype("i4"),
        "nHadrons": np.dtype("i4"),
        "ndMu": np.dtype("i4"),
        "EGround": ("f4", (3,)),
    }

    f = uproot.recreate(output_file)
    f.mktree("Header", branches_header, title="run header")
    f.mktree("Shower", branches_shower, title="shower info")
    f["Header"].extend(
        {
            "Seed1": int4,
            "Particle": PID,  # Proton type (no specific ID for tau)
            "Alpha": nan8,
            "lgEmin": nan8,
            "lgEmax": nan8,
            "zMin": zmin,
            "zMax": zmax,
            "SvnRevision": [0],
            "Version": [64],
            "OutputVersion": OutputVersion,
            "HEModel": int4,
            "LEModel": int4,
            "HiLowEgy": nan4,
            "hadCut": nan4,
            "emCut": nan4,
            "hadThr": nan4,
            "muThr": nan4,
            "emThr": nan4,
            "haCut": nan4,
            "muCut": nan4,
            "elCut": nan4,
            "gaCut": nan4,
            "lambdaLgE": a,
            "lambdaProton": a,
            "lambdaPion": a,
            "lambdaHelium": a,
            "lambdaNitrogen": a,
            "lambdaIron": a,
        }
    )
    f["Shower"].extend(
        {
            "lgE": TauEnergy,
            "zenith": zenith,  # 90+np.degree(beta_rad)
            "azimuth": azim,
            "Seed2": intn,
            "Seed3": intn,
            "Xfirst": Xfirst,
            "Hfirst": Zfirst * 1000,
            "XfirstIn": np.full(n, 0.5, dtype="f4"),
            "altitude": nan8n,
            "X0": X0,
            "Xmax": Xmax,
            "Nmax": Nmax,
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "chi2": chi2,
            "Xmx": Xmx,
            "Nmx": Nmx,
            "XmxdEdX": Xmx,
            "dEdXmx": dEdXmx,
            "cpuTime": nan4n,
            "X": X,
            "N": RN,
            "H": Z * 1000,  # in meters
            "D": np.asarray(D, dtype=np.float32) * 1000,
            "dEdX": np.asarray(dEdX_profiles, dtype=np.float32),
            "Mu": Xempty,  # Xempty
            "Gamma": Xempty,  # Xempty
            "Electrons": RN,
            "Hadrons": Xempty,  # Xempty
            "dMu": Xempty,  # Xempty
            "nX": np.full(n, profile_size, dtype="i4"),
            "nN": np.full(n, profile_size, dtype="i4"),
            "nH": np.full(n, profile_size, dtype="i4"),
            "nD": np.full(n, profile_size, dtype="i4"),
            "ndEdX": np.full(n, profile_size, dtype="i4"),
            "nMu": np.full(n, profile_size, dtype="i4"),
            "nGamma": np.full(n, profile_size, dtype="i4"),
            "nElectrons": np.full(n, profile_size, dtype="i4"),
            "nHadrons": np.full(n, profile_size, dtype="i4"),
            "ndMu": np.full(n, profile_size, dtype="i4"),
            "EGround": Eg,
        }
    )
    f.close()
    print(f"Output file saved to {output_file}")
