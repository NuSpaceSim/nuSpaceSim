# The Clear BSD License
#
# Copyright (c) 2021 Alexander Reustle and the NuSpaceSim Team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#      * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
r"""The main proceedure for performaing a full simulation in nuspacesim.

*********************
NuSpaceSim Simulation
*********************

.. currentmodule:: nuspacesim


.. autosummary::
   :toctree:

   compute

"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from astropy.table import Table as AstropyTable
from numpy.typing import ArrayLike
from rich.console import Console

from . import results_table
from .config import NssConfig
from .simulation.atmosphere.clouds import CloudTopHeight
from .simulation.eas_optical.eas import EAS
from .simulation.eas_radio.radio import EASRadio
from .simulation.eas_radio.radio_antenna import calculate_snr
from .simulation.geometry.region_geometry import RegionGeom, RegionGeomToO

# from .simulation.geometry.too import *

from .simulation.spectra.spectra import Spectra
from .simulation.taus.taus import Taus
from .augermc import *
#from .augermc_seniordesign import *

from .conex_out import conex_out
from .full_root_out import full_root_out
#from .testingfuncs import *

#from .testrcut import *
import matplotlib.pyplot as plt
from .simulation.eas_optical.shower_properties import slant_depth_trig_approx, particle_count_fluctuated_gaisser_hillas, gaisser_hillas_particle_count_exp_form
try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files
import awkward as ak

__all__ = ["compute"]


def compute(
    config: NssConfig,
    verbose: bool = False,
    output_file: str | None = None,
    to_plot: list = [],
    write_stages=False,
) -> AstropyTable:
    r"""Simulate an upward going shower.

    The main proceedure for performaing a full simulation in nuspacesim.
    Given a valid NssConfig object, :func:`compute`, will perform the simulation as
    follows:

    #. Initialize the AstropyTable object.
    #. Initialize the appropritate :ref:`simulation modules<simulation>`.
    #. Compute array of valid beta angle trajectories: beta_tr from :class:`RegionGeom`.
    #. Compute tau interaction attributes componentwise for each element of beta_tr.

       #. tauBeta
       #. tauLorentz
       #. showerEnergy
       #. tauExitProb

    #. Compute Extensive Air Shower attributes componentwise

       #. Decay Altitude
       #. Photon Density
       #. Cherenkov Angle

    #. Compute the Monte Carlo integral for the resulting shower geometries.

    At each stage of the simulation, array results are stored as contiguous columns,
    and scalar results are stored as attributes, both in the :class:`AstropyTable`
    object.


    Parameters
    ----------
    config: NssConfig
        Configuration object.
    verbose: bool, optional
        Flag enabling verbose output.
    output_file: str, optional
        Name of file to write intermediate stages
    to_plot: list, optional
        Call the listed plotting functions as appropritate.
    write_stages: bool, optional
        Enable writing intermediate results to the output_file.

    Returns
    -------
    AstropyTable
        The Table of result values from each stage of the simulation.
    """

    """
    betaangles=np.linspace(1.3*np.pi/180,30*np.pi/180,100)
    h=1416
    b2=(6356752.314245)**2
    a2 = (6378137.0)**2
    corepoint=latlongtoECEF(mean_lat,mean_long,h)
    normal = np.column_stack((corepoint[:,0]/a2,
                          corepoint[:,1]/a2,
                          corepoint[ :,2]/b2))
    normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)

    # 2. Compute local North vector in ECEF
    # Cross product of z-axis with normal gives East; East × Normal gives North
    z_axis = np.array([0, 0, 1])
    east = np.cross(z_axis, normal)
    east = east / np.linalg.norm(east, axis=1, keepdims=True)
    north = np.cross(normal, east)
    north = north / np.linalg.norm(north, axis=1, keepdims=True)

    # 3. Combine normal and north with emergence angle beta
    # vcoord = sin(beta) * north + cos(beta) * normal
    sin_beta = np.sin(betaangles)
    cos_beta = np.cos(betaangles)
    vcoord = cos_beta[:, np.newaxis] * north + sin_beta[:, np.newaxis] * normal


    c2=vcoord[:,0]**2/a2+vcoord[:,1]**2/a2+vcoord[:,2]**2/b2
    c1=2*(vcoord[:,0]*corepoint[:,0]/a2+vcoord[:,1]*corepoint[:,1]/a2+vcoord[:,2]*corepoint[:,2]/b2)
    c0=corepoint[:,0]**2/a2+corepoint[:,1]**2/a2+corepoint[:,2]**2/b2-1
    D=c1**2-4*c2*c0
    mask=(D>0)
    print(len(mask),mask.sum())
    #corepoint=corepoint[mask]
    #betaangles=betaangles[mask]
    #vcoord=vcoord[mask]
    #always take bigger t because traj is upward -> starting point is lower and moves in correct direction
    t=(-c1+np.sqrt(c1**2-4*c2*c0))/(2*c2)
    groundecef=corepoint+t[:,np.newaxis]*vcoord
    coreauger=np.zeros_like(groundecef)
    print(corepoint)
    coreauger[:,0]=corepoint[:,0]
    coreauger[:,1]=corepoint[:,1]
    coreauger[:,2]=corepoint[:,2]
    delta=10
    slant_depth_offline=integrated_grammage(groundecef,coreauger,delta)
    slant_depth_old=np.zeros_like(betaangles)
    for i in range(len(betaangles)):
        slant_depth_old[i]=slant_depth(0,h/1000,betaangles[i])[0]

    plt.figure(figsize=(10,6),dpi=100)
    plt.plot(np.degrees(betaangles),slant_depth_old,'x',markersize=3,label='Slant depth currently in nss')
    plt.plot(np.degrees(betaangles),slant_depth_offline,'+',markersize=3,label='Slant depth offline')
    plt.xlabel('emergence angle at core (degrees)')
    plt.ylabel('slant depth from sea level to core (g/cm2)')
    plt.grid()
    #plt.yscale('log')
    plt.legend()
    plt.savefig('slantdepthxfirstcalc.png')
    betaangles=np.linspace(1*np.pi/180,30*np.pi/180,100)
    thetaangles=np.pi/2-betaangles
    zstart=0
    zend=3  #km
    slant_depth_scipy=np.zeros_like(betaangles)
    slant_depth_trig_approximation=np.zeros_like(betaangles)
    integ_grammage_auger=np.zeros_like(betaangles)
    pathlen=np.zeros_like(betaangles)

    origpoint=latlongtoECEF(mean_lat,mean_long,0)
    origpoint=origpoint[0]
    b2=(6356752.314245)**2
    a2 = (6378137.0)**2
    normal = np.column_stack((origpoint[ 0]/a2,
                          origpoint[1]/a2,
                          origpoint[ 2]/b2))
    normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)

    # 2. Compute local North vector in ECEF
    # Cross product of z-axis with normal gives East; East × Normal gives North
    z_axis = np.array([0, 0, 1])
    east = np.cross(z_axis, normal)
    east = east / np.linalg.norm(east, axis=1, keepdims=True)
    north = np.cross(normal, east)
    north = north / np.linalg.norm(north, axis=1, keepdims=True)

    # 3. Combine normal and north with emergence angle beta
    # vcoord = sin(beta) * north + cos(beta) * normal
    sin_beta = np.sin(betaangles)
    cos_beta = np.cos(betaangles)
    vcoord = cos_beta[:, np.newaxis] * north + sin_beta[:, np.newaxis] * normal
    endpoints,pathlengths=find_trajectory_points_to_height(origpoint,vcoord,zend*1000)
    pathlen_computed=np.linalg.norm(endpoints-origpoint,axis=1)
    origauger=np.zeros_like(endpoints)
    origauger[:,0]=origpoint[0]
    origauger[:,1]=origpoint[1]
    origauger[:,2]=origpoint[2]
    delta=10
    for i in range(len(betaangles)):    
        slant_depth_scipy[i]=slant_depth(zstart,zend,betaangles[i])[0]
        slant_depth_trig_approximation[i]=slant_depth_trig_approx(zstart,zend,thetaangles[i])
        pathlen[i]=path_length_tau_atm(zend,betaangles[i])
    slant_depth_offline=integrated_grammage(origauger,endpoints,delta)
    plt.figure(figsize=(14,10),dpi=250)
    plt.plot(np.degrees(betaangles),pathlen*1000,'.',label='Path length function')
    plt.plot(np.degrees(betaangles),pathlengths,'x',label='Path length from ECEF')

    plt.grid()
    plt.yscale('log')
    plt.legend()
    plt.savefig('pathlength.png')

    plt.figure(figsize=(14,10),dpi=250)
    plt.plot(np.degrees(betaangles),slant_depth_scipy,'o',markersize=3,label='Slant depth scipy')
    plt.plot(np.degrees(betaangles),slant_depth_trig_approximation,'x',markersize=3,label='Slant depth trig approximation')
    plt.plot(np.degrees(betaangles),slant_depth_offline,'+',markersize=3,label='Slant depth offline')

    plt.grid()
    #plt.yscale('log')
    plt.legend()
    plt.savefig('slantdepth.png')"""

    console = Console(width=80, log_path=False)


    def logv(*args):
        """optionally print descriptive messages."""
        if verbose:
            console.log(*args)

    if verbose:
        console.rule("[bold blue] NuSpaceSim")

    def mc_logv(mcint, mcintgeo, numEvPass, mcunc, method):
        logv(f"\t[blue]Monte Carlo Integral [/][magenta][{method}][/]:", mcint)
        logv(
            f"\t[blue]Monte Carlo Integral, GEO Only [/][magenta][{method}][/]:",
            mcintgeo,
        )
        logv(f"\t[blue]Number of Passing Events [/][magenta][{method}][/]:", numEvPass)
        logv(f"\t[blue]Stat uncert of MC Integral [/][magenta][{method}][/]:", mcunc)

    sim = results_table.init(config)
    geom = RegionGeom(config)
    cloud = CloudTopHeight(config)
    spec = Spectra(config)
    tau = Taus(config)
    eas = EAS(config)
    #eas_radio = EASRadio(config)

    geom = (
        RegionGeomToO(config)
        if config.simulation.mode == "Target"
        else RegionGeom(config)
    )

    class StagedWriter:
        """Optionally write intermediate values to file"""

        def __call__(
            self,
            col_names: Iterable[str],
            columns: Iterable[ArrayLike],
            *args,
            **kwargs,
        ):
            sim.add_columns(columns, names=col_names, *args, **kwargs)
            if write_stages:
                sim.write(output_file, format="fits", overwrite=True)

        def add_meta(self, name: str, value: Any, comment: str):
            sim.meta[name] = (value, comment)
            if write_stages:
                sim.write(output_file, format="fits", overwrite=True)

    sw = StagedWriter()

    logv(f"Running NuSpaceSim with Energy Spectrum ({config.simulation.spectrum})")

    logv("Computing [green] Geometries.[/]")
    logv(
        f"\t[blue]Threw {config.simulation.thrown_events} neutrinos. All were valid.[/]"
    )
    logv("Computing [green] Energy Spectra.[/]")

    log_e_nu, mc_spec_norm, spec_weights_sum = spec(
        config.simulation.thrown_events, store=sw, plot=to_plot
    )
    maxE = np.array(9 + np.max(log_e_nu))
    nuE = np.array(9 + log_e_nu)
    n = config.simulation.thrown_events
    print('N=', n)

    # PARAMETERS
    maxangle = np.radians(30)
    radiusfactor = 2
    energy_threshold = 16
    gpstime = 1261872018  # Time at 1 Jan 2020 00:00:00 UTC
    ntels = 4
    # PARAMETERS
    radius = roundcalcradius(maxE, radiusfactor)
    groundecef, vecef, beta_tr, azimuth = gen_points(n, radius, maxang=maxangle)
    if beta_tr.size == 0:
        console.log("\t[red] WARNING: No valid events thrown! Exiting early! Check geometry![/]")
        return sim

    logv("Computing [green] Taus.[/]")
    tauBeta, tauLorentz, tauEnergy, showerEnergy, tauExitProb = tau(
        beta_tr, log_e_nu, store=sw, plot=to_plot
    )
    energies = np.log10(tauEnergy) + 9

    logv("Computing [green] Decay Altitudes.[/]")
    decayecef, altDec = decay(groundecef, vecef, energies)
    gpsarray = np.arange(gpstime, gpstime + n)

    # Initialize full trackers
    cumulative_indices = np.arange(n)
    id_full = np.full(n, 1, dtype=int)

    # originals: full copies
    originals = dict(
        nuE=nuE.copy(),
        energies=energies.copy(),
        groundecef=groundecef.copy(),
        vecef=vecef.copy(),
        beta_tr=beta_tr.copy(),
        showerEnergy=showerEnergy.copy(),
        decayecef=decayecef.copy(),
        altDec=altDec.copy(),
        azimuth=azimuth.copy(),
        gpsarray=gpsarray.copy(),
        tauExitProb=tauExitProb.copy(),
    )

    # variables: working copies (initially full)
    variables = {
        k: v.copy() for k, v in originals.items()
    }

    # Helper to apply cut
    def apply_cut(variables, mask):
        return {k: v[mask] for k, v in variables.items()}

    print('Calculating Shower Development cuts')

    # Cut 1: energy threshold + altitude
    maxalt=35
    cut1 = (variables["energies"] > energy_threshold) & (variables["altDec"] < maxalt)
    cumulative_indices = cumulative_indices[cut1]
    variables = apply_cut(variables, cut1)
    print(variables["energies"].size, f' Valid events over 10^16 eV and decay altitude < {maxalt} km')

    # Compute Xnuclearcollision
    meancollisionlength = 61.3
    Xnuclearcollision = np.random.exponential(scale=meancollisionlength, size=len(variables["energies"]))
    variables['Xnuclearcollision'] = Xnuclearcollision
    delta=10


    # Cut 2: trajectory inside telescope sphere
    id_temp, int1, int2, min_distance, rcut = trajectory_inside_tel_sphere(
       variables["energies"], variables["groundecef"], variables["vecef"], ntels, radiusfactor=radiusfactor
    )
    id_full[cumulative_indices] = id_temp
    cut2 = id_temp != 1
    cumulative_indices = cumulative_indices[cut2]
    variables = apply_cut(variables, cut2)

    # Post-cut2 computations
    dist2EarthCenter = np.sqrt(variables['groundecef'][:, 0]**2 + variables['groundecef'][:, 1]**2 + variables['groundecef'][:, 2]**2)
    init_lat = np.arcsin(variables['groundecef'][:, 2] / dist2EarthCenter)
    init_lon = np.arctan2(variables['groundecef'][:, 1], variables['groundecef'][:, 0])

    #Cut 2.1: remove showers with no intersection at 1415m (virtually no Earth travelled)
    startatm=1414

    startingecef, mask_start = starting_point(variables['groundecef'], variables['vecef'],startatm)
    variables['startingecef'] = startingecef

    cut2_1=mask_start
    cumulative_indices = cumulative_indices[cut2_1]
    variables = apply_cut(variables, cut2_1)

    Xfirst_offline = integrated_grammage_opt(variables['startingecef'], variables['decayecef'], delta)
    variables['Xfirst_offline'] = Xfirst_offline

    Xfirstinteract = variables['Xfirst_offline'] + variables['Xnuclearcollision']
    variables['Xfirstinteract'] = Xfirstinteract

    remainingn = len(variables['Xfirst_offline'])
    xlimfactor = 5
    deltaX=10
    startoutcounter=0
    xmaxecef = np.full((remainingn, 3), 0.0)
    n_e_array = np.zeros(remainingn)
    all_ghparams = np.zeros((remainingn, 6))
    xmax_outside_atm_counter = 0
    xmax_outside_atm_and_trigg = 0
    finalmask = np.zeros(remainingn, dtype=bool)
    min_e=0.05
    variables['xstartecef']=np.zeros((remainingn,3))
    xfinal=np.zeros(remainingn)
    height_compare1=np.zeros(remainingn)
    height_compare2=np.zeros(remainingn)
    dist_compare1=np.zeros(remainingn)
    dist_compare2=np.zeros(remainingn)

    X_builder = ak.ArrayBuilder()
    RN_builder = ak.ArrayBuilder()
    dEdX_builder = ak.ArrayBuilder()


    print('Start loop')

    for i in range(remainingn):
        height,slant,dist=auger_atm_table(      variables['startingecef'][i,:],
                                                variables['vecef'][i,:],
                                                variables['groundecef'][i,:], 
                                                deltaX,                      # [g/cm^2] slant-depth step (pass positive)
                                                depth_spline,             # X(h_km) -> depth [g/cm^2]
                                                height_spline,             # h_km(X) -> height [km]
                                                altitude_from_ecef,          # alt_m(ECEF) -> meters
                                                startatm,   )
        start_slant=Xfirstinteract[i]
        if start_slant+100>slant[-1]:
            startoutcounter+=1
            #print('Skip event, start slant outside atmosphere',start_slant,slant[-1])
            continue
        start_dist=np.interp(start_slant,slant,dist)
        variables['xstartecef'][i,:]=variables['groundecef'][i,:]+start_dist*variables['vecef'][i,:]

        showerEnergy = variables['showerEnergy']
        EshowGeV = (showerEnergy * 1e8)  # GeV
        variables['EshowGeV'] = EshowGeV

        with as_file(
            files("nuspacesim.data.CONEX_table") / "dumpGH_conex_pi_E17_95deg_0km_eposlhc_1394249052_211.dat"
        ) as file:
            CONEX_table = np.loadtxt(file, usecols=(4, 5, 6, 7, 8, 9))

        available_grammage = slant[-1]-Xfirstinteract[i]

        idx = np.random.randint(low=0, high=CONEX_table.shape[0])
        Nm, Xm, X0, p1, p2, p3 = CONEX_table[idx]
        shiftedX0 = 0
        shiftedXm = Xm - X0
        shiftedp3 = p3
        shiftedp2 = p2 + 2 * p3 * X0
        shiftedp1 = p1 + p2 * X0 + p3 * X0 ** 2

        Nmax100 = 6.99e7
        NmaxE = 0.045 * (1.0 + 0.0217 * np.log(variables['EshowGeV'][i] / 1.0e5)) * variables['EshowGeV'][i] / 0.074
        Nmax = Nm * NmaxE / Nmax100

        XmaxOff = 58.0 * np.log10(variables['EshowGeV'][i] / 1.0e8)
        shiftedXmax = shiftedXm + XmaxOff

        maxgramm = min(xlimfactor * shiftedXmax, available_grammage)
        x = np.arange(0, maxgramm, deltaX)
        xfinal[i]=x[-1]
        shiftedgh_lam = shiftedp1 + shiftedp2 * x + shiftedp3 * x * x
        shiftedghparams = np.array([shiftedX0, shiftedXmax, Nmax, shiftedp3, shiftedp2, shiftedp1])
        all_ghparams[i] = shiftedghparams
        distances_along_shower=np.interp(x+start_slant,slant,dist)
        rn = gaisser_hillas_particle_count_exp_form(x, shiftedX0, shiftedXmax, Nmax, shiftedgh_lam)
        code, n_e_per_m2, dedx = energy_at_tel(variables['groundecef'][i,:], variables['vecef'][i,:], x, rn, min_e,shiftedXmax,distances_along_shower,ntels)
        """plt.figure(figsize=(12,10),dpi=200)
        plt.plot(x,rn,label='No. of particles')
        plt.plot(x,rn*0.0025935,label='dEdX constant')
        plt.plot(x,dedx,label='dE/dX')
        plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.savefig(f'dedx_over_rn_example_{i}.png')
        exit()"""
        global_i = cumulative_indices[i]
        #print(id_full[global_i], ' ID before', code, ' ID after')
        id_full[global_i] = code

        if available_grammage < shiftedXmax:
            xmax_outside_atm_counter += 1
            if code != 1:
                xmax_outside_atm_and_trigg += 1
                #print('ATTENTION Xmax outside atmosphere', available_grammage, shiftedXmax)
                #print(code, n_e_per_m2)

        if i % 1000 == 0:
            print(f"Done {i}/{remainingn} ({remainingn - i} left)")
        if code == 1:
            continue
        disttoxmax=np.interp(shiftedXmax+start_slant,slant,dist)
        xmaxecef[i, :] = variables['groundecef'][i, :] + disttoxmax * variables['vecef'][i, :]
        n_e_array[i] = n_e_per_m2
        finalmask[i] = True
        mask = (rn > 0)
        x_values = x[mask] + start_slant
        rn_values = rn[mask]
        dedx_values = dedx[mask]
        X_builder.append(x_values)
        RN_builder.append(rn_values)
        dEdX_builder.append(dedx_values)


        height_compare1[i]=np.interp((x_values[1]+x_values[0])/2,slant,height)
        height_compare2[i]=np.interp(x_values[-1],slant,height)
        dist_compare1[i]=np.interp((x_values[1]+x_values[0])/2,slant,dist)
        dist_compare2[i]=np.interp(x_values[-1],slant,dist)

    #print('Height 1 and 2', height_compare1[finalmask], height_compare2[finalmask])
    #print('Dist 1 and 2', dist_compare1[finalmask], dist_compare2[finalmask])

    print(startoutcounter, ' Start shower outside atmosphere')

    print(np.sum(finalmask), ' Events with enough shower development in view of detector')
    #print(xmax_outside_atm_counter, ' Events with Xmax outside atmosphere')
    #print(xmax_outside_atm_and_trigg, ' Events with Xmax outside atmosphere that still triggered')

    final_local = np.flatnonzero(finalmask)
    final_global = cumulative_indices[final_local]
    print(id_full[final_global], 'FINAL IDs')
    plot_telescope_hist(id_full[final_global])
    plot_telescope_multiplicity(id_full[final_global])
    #Re run GEOM cut to check if its a redundant cut after e_at_tel cut. Does it keep the zenith, azim distribution the same? 
    # radiusfactor=1.5
    # id_temp, int1, int2, min_distance, rcut = trajectory_inside_tel_sphere(
    #     variables["energies"][final_local], variables["groundecef"][final_local], variables["vecef"][final_local], ntels, radiusfactor=radiusfactor)
    # # = trajectory_inside_tel_sphere(
    # #    variables["energies"], variables["groundecef"], variables["vecef"], ntels, radiusfactor=radiusfactor
    # #)
    # #id_full[cumulative_indices] = id_temp
    # cut2 = id_temp != 1
    # final_global_geom=final_global[cut2]
    # #cumulative_indices_geomcut = cumulative_indices[cut2]
    # def generate_hists():

    #     # Variable histograms
    #     variableshists = {
    #         'lgE': {'data': originals["energies"], 'label': 'Log Energy (lgE)', 'xlabel': 'lgE'},
    #         'zenith': {'data': np.degrees(originals["beta_tr"]), 'label': 'Zenith Angle (degrees)', 'xlabel': 'Zenith (degrees)'},
    #         'azimuth': {'data': np.degrees(originals["azimuth"]), 'label': 'Azimuth Angle (degrees)', 'xlabel': 'Azimuth (degrees)'},

    #     }

    #     for var_name, var_info in variableshists.items():
    #         plt.figure(figsize=(8, 10))
    #         data_range = (np.min(var_info['data']), np.max(var_info['data']))
    #         if var_name=='lgE':
    #             data_range=(16,19)
    #         bins = np.linspace(data_range[0], data_range[1], 51)
    #         ax1 = plt.subplot(3, 1, 1)
    #         n_all = len(var_info['data'])
    #         mean_all = np.mean(var_info['data'])
    #         min_all = np.min(var_info['data'])
    #         max_all = np.max(var_info['data'])
    #         plt.hist(var_info['data'], bins=bins, color='blue', alpha=0.7, label=(
    #             f'All Events\nMean: {mean_all:.2f}\nMin: {min_all:.2f}\nMax: {max_all:.2f}'
    #         ))
    #         plt.ylabel('Number of Events')
    #         plt.title(f'Histogram of {var_info["label"]} (Simulation start Events, n={n_all})')
    #         plt.legend()
    #         plt.grid(True, alpha=0.3)
    #         plt.subplot(3, 1, 2, sharex=ax1)
    #         triggered_data = var_info['data'][final_global]
    #         n_triggered = len(triggered_data)
    #         if n_triggered > 0:
    #             mean_triggered = np.mean(triggered_data)
    #             min_triggered = np.min(triggered_data)
    #             max_triggered = np.max(triggered_data)
    #             plt.hist(triggered_data, bins=bins, color='green', alpha=0.7, label=(
    #                 f'e_at_tel >0.05 events\nMean: {mean_triggered:.2f}\nMin: {min_triggered:.2f}\nMax: {max_triggered:.2f}'
    #             ))
    #         else:
    #             plt.text(0.5, 0.5, 'No Triggered Events', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
    #         plt.xlabel(var_info['xlabel'])
    #         plt.ylabel('Number of Events')
    #         plt.title(f'Histogram of {var_info["label"]} (e_at_tel>0.05 Events, n={n_triggered})')
    #         plt.legend()
    #         plt.grid(True, alpha=0.3)


    #         plt.subplot(3, 1, 3, sharex=ax1)
    #         triggered_data = var_info['data'][final_global_geom]
    #         n_triggered = len(triggered_data)
    #         if n_triggered > 0:
    #             mean_triggered = np.mean(triggered_data)
    #             min_triggered = np.min(triggered_data)
    #             max_triggered = np.max(triggered_data)
    #             plt.hist(triggered_data, bins=bins, color='green', alpha=0.7, label=(
    #                 f'Passing Geometry events\nMean: {mean_triggered:.2f}\nMin: {min_triggered:.2f}\nMax: {max_triggered:.2f}'
    #             ))
    #         else:
    #             plt.text(0.5, 0.5, 'No Triggered Events', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
    #         plt.xlabel(var_info['xlabel'])
    #         plt.ylabel('Number of Events')
    #         plt.title(f'Histogram of {var_info["label"]} (Passing Geometry Events, n={n_triggered})')
    #         plt.legend()
    #         plt.grid(True, alpha=0.3)


    #         plt.tight_layout()
    #         plt.savefig(f'{var_name}_hist.png')
    #         plt.close()

    #         plt.figure(figsize=(8, 10))


    #         # --- Second: larger dataset (triggered) ---
    #         triggered_data = var_info['data'][final_global]
    #         n_triggered = len(triggered_data)
    #         if n_triggered > 0:
    #             mean_triggered = np.mean(triggered_data)
    #             min_triggered = np.min(triggered_data)
    #             max_triggered = np.max(triggered_data)
    #             plt.hist(
    #                 triggered_data, bins=bins, color='green', alpha=0.5,
    #                 label=(f'e_at_tel > 0.05 events n={n_triggered}\nMean: {mean_triggered:.2f}\nMin: {min_triggered:.2f}\nMax: {max_triggered:.2f}')
    #             )
    #         else:
    #             plt.text(0.5, 0.4, 'No Triggered Events', ha='center', va='center',
    #                     transform=plt.gca().transAxes, fontsize=12)

    #         geom_data = var_info['data'][final_global_geom]
    #         n_geom = len(geom_data)
    #         if n_geom > 0:
    #             mean_geom = np.mean(geom_data)
    #             min_geom = np.min(geom_data)
    #             max_geom = np.max(geom_data)
    #             plt.hist(
    #                 geom_data, bins=bins, color='blue', alpha=0.5, 
    #                 label=(f'Passing Geometry events n={n_geom} \nMean: {mean_geom:.2f}\nMin: {min_geom:.2f}\nMax: {max_geom:.2f}')
    #             )
    #         else:
    #             plt.text(0.5, 0.5, 'No Geometry Events', ha='center', va='center',
    #                     transform=plt.gca().transAxes, fontsize=12)


    #         # --- Labels, legend, formatting ---
    #         plt.xlabel(var_info['xlabel'])
    #         plt.ylabel('Number of Events')
    #         plt.title(f'Histogram of {var_info["label"]} (Overlayed)')
    #         plt.legend()
    #         plt.grid(True, alpha=0.3)
    #         plt.tight_layout()
    #         plt.savefig(f'{var_name}_overlapped_hist.png')
    #         plt.close()

    # #generate_hists()

    # remaining_indices = np.arange(len(final_local))
    # remaining_indices = remaining_indices[~cut2]
    # print(len(remaining_indices), ' Events passing geometry cut')
    # print(len(cut2))
    # print(np.sum(cut2))
    # print('E_at_tel of events not passing geom', n_e_array[final_local][~cut2])
    # coreenu=eceftoenu(LLecef,originals["groundecef"][final_global],lat=LLlat,lon=LLlong)
    # xstartenu=eceftoenu(LLecef,variables["xstartecef"][final_local],lat=LLlat,lon=LLlong)
    # xmaxenu=eceftoenu(LLecef,xmaxecef[final_local],lat=LLlat,lon=LLlong)
    # plot_trajectories(coreenu,xstartenu,xmaxenu,remaining_indices,50, "passing_e_tel_but_not_geom",originals["energies"][final_global],np.degrees(originals["beta_tr"][final_global]),n_e_array[final_local])



    print('Maximum Alt Decay:', np.max(variables['altDec'][final_local]))
    conex_out(
        X_builder, RN_builder, dEdX_builder,
        id_full[final_global],
        originals["groundecef"][final_global],
        originals["beta_tr"][final_global],
        originals["energies"][final_global],
        originals["altDec"][final_global],
        originals["azimuth"][final_global],
        originals["gpsarray"][final_global],
        originals["nuE"][final_global],
        originals["tauExitProb"][final_global],
        all_ghparams[final_local, :],
        variables['Xfirstinteract'][final_local],
        output_file,
        xmaxecef[final_local, :],
        variables['xstartecef'][final_local],
        n_e_array[final_local],
        height_compare1[final_local],
        height_compare2[final_local],
        dist_compare1[final_local],
        dist_compare2[final_local]
    )
    """
    # Full root out with full originals and final id (rejects=1, passers !=1)
    full_root_out(
        n, maxangle, originals["nuE"], originals["energies"], energy_threshold,
        originals["groundecef"], originals["vecef"], originals["decayecef"],
        originals["altDec"], originals["beta_tr"], originals["azimuth"],
        originals["gpsarray"], originals["tauExitProb"],
        id_full
    )
    """


    """numPEs, costhetaChEff, profilesOut, ghparams = eas(
        beta_tr[valid_evs],
        altDec[valid_evs],
        showerEnergy[valid_evs],
        init_lat,
        init_long,
        Conex,
        cloudf=cloud,
        #store=sw,
        plot=to_plot,
    )

    
    plt.figure(figsize=(12, 12),dpi=300)
    size=40e3

    x_bins = np.linspace(0-size,0+size, 60)
    y_bins = np.linspace(0-size,0+size, 60)
    xmaxenu= eceftoenu(telposecef[0, :], xmaxecef)

    # Create 2D histogram
    hist, xedges, yedges, im = plt.hist2d(xmaxenu[finalmask,0], xmaxenu[finalmask,1], bins=[x_bins, y_bins], cmap='viridis', cmin=1)

    # Add colorbar
    plt.colorbar(im, label='Number of Events')# Add colorbar
    plt.plot(0,0,color='red',marker='v',markersize=7,zorder=10)
    #plt.plot(centerarray_respLL[0],centerarray_respLL[1],color='black',marker='x',markersize=5)
    #plt.plot(cornereastenu, cornernorthenu, color='red', linewidth=0.5, label="UTM Zone 19H")
    # Set labels and title
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'scatter plot of all cores (n={len(xmaxecef[finalmask,0])})')
    #plt.plot(LLangx_vals, LLangy_vals, 'r--', linewidth=0.5, label=f'LL line of sight')

    plt.legend()
    plt.gca().set_aspect('equal')

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'allxmax_hist2d_energydep.png')
    """


    """
        logv("Computing [green] Optical Monte Carlo Integral.[/]")
        mcint, mcintgeo, passEV, mcunc = geom.mcintegral(
            numPEs,
            costhetaChEff,
            tauExitProb,
            config.detector.optical.photo_electron_threshold,
            mc_spec_norm,
            spec_weights_sum,
            lenDec=lenDec,
            method="Optical",
            store=sw,
        )

        sw.add_meta("OMCINT", mcint, "Optical MonteCarlo Integral")
        sw.add_meta("OMCINTGO", mcintgeo, "Optical MonteCarlo Integral, GEO Only")
        sw.add_meta("ONEVPASS", passEV, "Optical Number of Passing Events")
        sw.add_meta("OMCINTUN", mcunc, "Stat unc of MonteCarlo Integral")

        mc_logv(mcint, mcintgeo, passEV, mcunc, "Optical")

    if config.detector.radio.enable:
        logv("Computing [green] EAS Radio signal.[/]")

        eFields = eas_radio(
            beta_tr, altDec, lenDec, thetaArr, pathLenArr, showerEnergy, store=sw
        )

        snrs = calculate_snr(
            eFields,
            freqRange,
            config.detector.initial_position.altitude,
            config.detector.radio.nantennas,
            config.detector.radio.gain,
        )

        logv("Computing [green] Radio Monte Carlo Integral.[/]")
        mcint, mcintgeo, passEV, mcunc = geom.mcintegral(
            snrs,
            np.cos(config.simulation.max_cherenkov_angle),
            tauExitProb,
            config.detector.radio.snr_threshold,
            mc_spec_norm,
            spec_weights_sum,
            lenDec=lenDec,
            method="Radio",
            store=sw,
        )

        sw.add_meta("RMCINT", mcint, "Radio MonteCarlo Integral")
        sw.add_meta("RMCINTGO", mcintgeo, "Radio MonteCarlo Integral, GEO Only")
        sw.add_meta("RNEVPASS", passEV, "Radio Number of Passing Events")
        sw.add_meta("RMCINTUN", mcunc, "Stat unc of MonteCarlo Integral")

        mc_logv(mcint, mcintgeo, passEV, mcunc, "Radio")"""

    logv("\n :sparkles: [cyan]Done[/] :sparkles:")

    return sim
