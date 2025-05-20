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
from .conex_out_withZprofile import conex_out
from .full_root_out import full_root_out
from .testrcut import *

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
    maxE=np.array(9+np.max(log_e_nu))
    nuE=np.array(9+log_e_nu)
    n=config.simulation.thrown_events
    print('N=',n)
    
    #PARAMETERS
    maxangle=np.radians(30)
    radiusfactor=0.5
    minshowerpct=50
    energy_threshold=16
    gpstime=1261872018  #Time at 1 Jan 2020 00:00:00 UTC
    ntels=1
    #PARAMETERS

    radius=roundcalcradius(maxE,radiusfactor)
    groundecef, vecef,beta_tr, azimuth=gen_points(n,radius,maxang=maxangle)
    #groundenu=eceftoenu(centerecef,groundecef)
    if beta_tr.size == 0:
        console.log(
            "\t[red] WARNING: No valid events thrown! Exiting early! Check geometry![/]"
        )
        return sim

    logv("Computing [green] Taus.[/]")
    tauBeta, tauLorentz, tauEnergy, showerEnergy, tauExitProb = tau(
        beta_tr, log_e_nu, store=sw, plot=to_plot
    )
    energies=np.log10(tauEnergy)+9

    logv("Computing [green] Decay Altitudes.[/]")






    decayecef,altDec=decay(groundecef,vecef, energies)
    #Make .root file of ALL events. Add all simulation parameters
    
    #Mask events with energies below 10^16 eV
    gpsarray=np.arange(gpstime,gpstime+n)


    #ADD THIS FOR REAL SIMS
    #full_root_out(n,maxangle,nuE,energies,energy_threshold,groundecef,vecef,decayecef,altDec,beta_tr,azimuth,gpsarray,tauExitProb)

    valid_energies=(energies>energy_threshold)
    energies=energies[valid_energies]
    print(energies.size,' Valid events over 10^16 eV')
    groundecef=groundecef[valid_energies]
    vecef=vecef[valid_energies]
    beta_tr=beta_tr[valid_energies]
    showerEnergy=showerEnergy[valid_energies]
    decayecef=decayecef[valid_energies]
    altDec=altDec[valid_energies]
    azimuth=azimuth[valid_energies]
    #sw(
    #    ("init_lat", "init_lon"),
    #    (init_lat, init_long),
    #)



    id,int1,int2=trajectory_inside_tel_sphere(energies,groundecef,vecef,ntels,radiusfactor=radiusfactor)
    idfinal=decay_inside_fov(energies,groundecef,vecef,beta_tr,decayecef,altDec, id,int1,int2,ntels
                             ,diststep=200,radiusfactor=radiusfactor,minshowerpct=minshowerpct)
    valid_evs=(idfinal!=1)

    dist2EarthCenter = np.sqrt(groundecef[valid_evs,0]**2 + groundecef[valid_evs,1]**2 + groundecef[valid_evs,2]**2)
    init_lat = np.arcsin(groundecef[valid_evs,2] / dist2EarthCenter)
    init_long = np.arctan2(groundecef[valid_evs,1], groundecef[valid_evs,0])


    vecef=vecef[valid_evs]

    groundecef=groundecef[valid_evs]

    startingecef=starting_point(groundecef,vecef)
    delta=10
    Xfirst_offline=integrated_grammage(startingecef,decayecef[valid_evs],delta)


    logv("Computing [green] EAS Optical Cherenkov light.[/]")
    Conex=config.simulation.conex_output
    numPEs, costhetaChEff, profilesOut, ghparams = eas(
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
    if Conex:
        conex_out(profilesOut,idfinal[valid_evs],groundecef,vecef
                    ,beta_tr[valid_evs],energies[valid_evs],altDec[valid_evs]
                    ,azimuth[valid_evs],gpsarray[valid_energies][valid_evs]
                    ,nuE[valid_energies][valid_evs],tauExitProb[valid_energies][valid_evs],h,ghparams,Xfirst_offline)
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
