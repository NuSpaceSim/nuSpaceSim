conex2r7.50
===========

Last modifications : 27.05.2020 <tanguy.pierog@kit.edu>

 Output version 2.5x
 Conex v7.50 (trunk -r7548)
 by T. Pierog, N.N. Kalmykov, S. Ostapchenko and K. Werner
 with the collaboration of R. Engel and D. Heck.
 Paper to be cited if you use this program : [1] ([2])
 Ref. :
[1]
@Article{Bergmann:2006yz,
     author    = "Bergmann, T. and others",
     title     = "One-dimensional hybrid approach to extensive air shower
                  simulation",
     journal   = "Astropart. Phys.",
     volume    = "26",
     year      = "2007",
     pages     = "420-432",
     eprint    = "astro-ph/0606564",
     SLACcitation  = "%%CITATION = ASTRO-PH/0606564;%%"
}
[2]
@Article{Pierog:2004re,
     author    = "Pierog, T. and others",
     title     = "First Results of Fast One-dimensional Hybrid Simulation of
                  EAS Using CONEX",
     journal   = "Nucl. Phys. Proc. Suppl.",
     volume    = "151",
     year      = "2006",
     pages     = "159-162",
     eprint    = "astro-ph/0411260",
     SLACcitation  = "%%CITATION = ASTRO-PH/0411260;%%"
}
 Original work : CONEX 1.0 (2003)
 by the Nantes-Moscow collaboration
 (H.J. Drescher, N.N. Kalmykov, S. Ostapchenko, and K. Werner)
[3]
@Article{Bossard:2000jh,
     author    = "Bossard, G. and others",
     title     = "Cosmic ray air shower characteristics in the framework of
                  the  parton-based Gribov-Regge model NEXUS",
     journal   = "Phys. Rev.",
     volume    = "D63",
     year      = "2001",
     pages     = "054030",
     eprint    = "hep-ph/0009119",
     SLACcitation  = "%%CITATION = HEP-PH/0009119;%%"
}

 ROOT interface by M. Unger, R. Ulrich, and T. Pierog

 CONEX_EXTENSION by R. Ulrich


IMPORTANT NOTE : the default energy threshold for the MC to cascade
equation transition is now 0.005 instead of 0.05 to get a more realistic
description of shower to shower fluctuactions (with high statistic).


INTRODUCTION

CONEX is a hybrid simulation code that is suited for fast
one-dimensional simulations of shower profiles, including  fluctuations.
It combines Monte Carlo (MC) simulation of high energy interactions with
a fast numerical solution of cascade equations (CE) for the resulting
distributions of secondary particles.  For a given primary mass, energy,
and zenith angle, the energy deposit profile as well as charged particle
and muon longitudinal profiles are calculated. Furthermore an extended
Gaisser-Hillas (GH) is performed for each shower profile similar to what
is implemented in CORSIKA. The shower simulation parameters, profiles and fit
results are written to a ROOT file which is briefly described below (see
also macros/plotProfile.C for technical details).

High energy hadronic interaction model are EPOS LHC, QGSJETII-04
QGSJET01 and SIBYLL 2.3d. Default low energy hadronic interaction model is now 
UrQMD1.3. A shared library is compiled for each hadronic interaction model and
only the needed one is loaded during execution.

CONEX to ROOT interface version 5 :

- environment variables:

              CONEX_ROOT:  directory where steering files (described below)
                           reside (default $PWD/cfg)
              ROOT_OUT:    directory where to put the root output
                           file (default $PWD)
              URQMD13_TAB: full file name (with path) for a table produced by UrQMD
                           (default $PWD/tables.dat)


- for installation

    a) make sure that ROOT is set up properly, especially
       that $ROOTSYS/bin is in your your search PATH

    b) type "make [opt]" to create the libaries in subdirectory "lib"
       (where "opt" is "qgsjet", "epos", "qgsjetII" or "sibyll".
       type "make all" to create all the libaries.

    c) Use CONEX_PREFIX environment, if you want to chose an alternative
       installation location. By default $PWD is used.

    d) If you want to rely on the model tables in a central place since
       they are big and don't change often, use CONEXTABROOT
       envirnoment variable to point there. By default the local "tabs"
       directory is used.

    e) Have fun!
    

- for running

    a) If you don't set the environment variable CONEX_ROOT,
       the steering files (*.param) will be read from $PWD/cfg.
       If you don't set the environment variable ROOT_OUT,
       the output file will be written to $PWD
       If you don't set the environment variable URQMD13_TAB
       (with something like $CONEX_ROOT/tabs/urqmd.dat) a file
       tables.dat will be created in your $PWD. 

    b) Have more fun!


- program options (can be obtained by typing "bin/conex2r -h")

  a more detailed description of available options:

     flag         description           default value        Notes
   --------+------------------------+------------------+---------------
     -a       spectral index alpha:         3                 [2]
              dN/dE ~ E^(-alpha)

     -e       log10(Emin/eV)                16.5

     -E       log10(Emax/eV)                21

     -i       azimuth angle (degree)        0                 [4]

     -m       high energy interaction       4
              model:

              2=QGSJET01
              4=EPOS LHC
              5=SIBYLL 2.3d
              6=QGSJETII-04
              9=DMPJETIII.2017-1

     -n       number of Showers             1

     -o       minimum impact parameter [m]  0
     -O       maximum impact parameter [m]  0

     -p       particle type: 100=proton     100               [5]
                            5600=iron
                               0=gamma

     -s       random seed                   0                 [1]

     -S       auto-save after n showers    10

     -x       file name prefixe            conex              [6]

     -z       min zenith angle (degree)     60                [3]

     -Z       max zenith angle (degree)     60                [3]

     -H       1 = no electromagnetic shower  0

if LEADING_INTERACTIONS_TREE is defined in conexConfig.h (default) :

     -K       maximum detail Level         0                  [7]


Comments :
[1] if s <= 0, first seed for random number generator is automatically
    generated using "/dev/urandom" and the date (if possible, only
    under Linux) otherwise a fixed one is used (123).
    if s > 0, the first seed is the value of s (if you want to control
    the seed generation)
    The seed is used in the ouput file name (see [4]), so if the seed
    is the same for 2 different runs, you will generate the same showers
    (if parameters are equivalent) and the output file will be overwritten.
    Showers can only be reproduced with the same seed by using
    QGSJet. This limitation is due to a problem of single to double precision
    conversion.

[2] if a > 1 , primary energy E follows :  E^(-a) with Emin < E < Emax
    if 0 <= a <= 1 , primary energy E follows : 1/E  with Emin < E < Emax
    if a < 0 , log of primary energy log10(E) follows
               log10(E) = log10(Emin) + n * abs(a)
               where n is an integer and Emin <= E <= Emax.
    if Emin=Emax, a doesn't play any role : E=Emin=Emax.

[3] 0 <= angle <= 180. If angle > 90, the shower is upward-going and
    comes from the ground
    if theta1!=theta2 zenith angle is drawn from isotropic flux
    on flat surface, i.e.

        dN/dcos(theta) ~ cos(theta)


    Obviously, this makes not much sense for horizontal showers, so please
    make shure to set theta1=theta2 for this purpose

[4] 0 <= angle <= 360. AUGER definition of "phi" angle is used : 0 = East
    if angle < 0 it is drawn uniform from [0,360] deg.

[5] particle code = A*100 for Nuclei, gamma-ray : A=0
    (i.e. proton : A=1, iron : A=56)
    or PDG Monte-Carlo code can be used for any other particle
    (.e 11=electron, 22=photon, ...)
    Rmq: only 'standard' particles can be used (nuclei, long live time
    hadrons + gamma, electron and mu) => neutrino or tau induced showers
    are not available.

[6] output file name is automatically generated :
                         <prefix>_<MC model name>_<seed>_<particle>.root
    default one is then : conex_eposlhc_??????????_100.root
    Description of this file : see below.

[7] Number of hadronic Monte-Carlo interactions (<100) saved in LeadingInteractions tree
    For K=1, only the secondary particles of the first interaction are saved.
    For K=2, after the first interaction, the secondary particles of the next
    interaction of the leading (most energetic) particle are recorded too.
    And so on ... the secondary particles produced by the leading particle of the
    previously saved interaction are recorded.

- steering files: conex_sibyll.param
                  conex_qgsjet.param
                  conex_qgsjetII.param
                  conex_epos.param
                  (after compilation in subdirectory cfg)
  These files are written to have realistic and fast results.
  Options available here are described in the files.
  Notes :
                  * Cuts define the minimum energy for hadrons (muons)
                    (not less then 0.3 GeV) and
                    electromagnetic (e/m) particles (not less than 1 MeV).
                  * Threshold should not be changed if you don't want to
                    use special cases (1=only cascade equations
                    (no fluctuations) (Emax=10^10GeV) and 0=only MC).
                  * If "ixmax" is set to 0, the GH fit is not done in
                    CONEX (save time). Function is :

   N(X) = NMAX * ((X-X0)/(XMAX-X0))**((XMAX-X0)/(P1+P2*X+P3*X**2))
               * EXP((XMAX-X)/(P1+P2*X+P3*X***2))

                  * "xmaxp" is the maximum slant depth point for the
                    simulation. If nothing is specified, the simulation
                    is done until the ground (technical max is 37000 g/cm2).

- example macros to read the ROOT files is located in directory "macros".

  In the root file "Header" tree, branches are :
    "Seed1" : random seed1
    "Particle" : particle ID
    "Alpha"    : spectral index
    "lgEmin"   : log10 of the minimum energy in eV
    "lgEmax"   : log10 of the maximum energy in eV
    "zMin"     : minimum zenith angle in degree
    "zMax"     : maximum zenith angle in degree
    "SVNRevision"  : svn revision
    "Version"  : conex version
    "OutputVersion"  : cxroot output version
    "HEModel"  : High Energy interaction model flag
                 (2=QGSJET01, 4=EPOS LHC, 5=SIBYLL 2.3d, 6=QGSJETII-04)
    "LEModel"  : Low Energy (E < HiLowEgy GeV) model flag (3=GHEISHA, 8=URQMD 1.3.1)
    "HiLowEgy" : Transition energy from low to high hadronic interaction model (80 GeV) 
    "hadCut"   : hadron and muon cutoff (minimum energy) for charged particle profiles
    "emCut"    : e/m particles cutoff for charged particle profiles
                 (minimum energy for electrons, positrons and gammas)
    "hadThr"   : Emax(CE)/Emax(MC) for hadrons (threshold)
    "muThr"    : Emax(CE)/Emax(MC) for muon (threshold)
    "emThr"    : Emax(CE)/Emax(MC) for e/m particles (threshold)
    "haCut"    : cutoff for hadron profile
    "muCut"    : cutoff for muons profile
    "elCut"    : cutoff for electron+positron profile
    "gaCut"    : cutoff for photon profile

    "lambdaProton" : proton-air interaction length [g/cm^2]
    "lambdaPion" : pion-air interaction length [g/cm^2]
    "lambdaHelium" : He-air interaction length [g/cm^2]
    "lambdaNitrogen" : N-air interaction length [g/cm^2]
    "lambdaIron" : Fe-air interaction length [g/cm^2]
    all as a function of energy, "lambdaLgE"

  For each generated shower, branches in "Shower" tree are :
  > parameters :
    "lgE"      : log10 of the primary energy in eV
    "zenith"   : zenith angle in degree
    "azimuth"  : azimuth angle in degree (0 = East)
    "Seed2"    : random seed2 (number of random number generator calls
                 below 1 billion)
    "Seed3"    : random seed3 (number of billions of random number
                 generator calls)
    "XfirstIn" : inelasticity of first interaction ([0,1])
    "Xfirst"   : real first interaction point in slant depth (g/cm^2)
    "Hfirst"   : altitude of real first interaction point (m)
    "altitude" : altitude of impact parameter [m]
    "X0"       : "pseudo" first interaction point for GH fit
    "Xmax"     : GH fit result for slant depth of the shower maximum (g/cm^2)
    "Nmax"     : Number of charged particles above cut-off at the shower maximum
    "p1"       : first parameter for the polynomial function of the GH fit
    "p2"       : second parameter for the polynomial function of the GH fit
    "p3"       : third parameter for the polynomial function of the GH fit
    "chi2"     : Chi squared / number of degree of freedom / sqrt (Nmax) for the fit
                 (small number not realistic because it's divided by sqrt (Nmax) )
    "Xmx"      : X-position of maximum of quadratic fit of N(X) profile [g/cm^2]
    "Nmx"      : Maximum of quadratic fit on N(X) profile
    "XmxdEdX"  : X-position of maximum of quadratic fit of dE/dX(X) profile [g/cm^2]
    "dEdXmx"   : Maximum of quadratic fit on dE/dX(X) profile [GeV/(g/cm^2)]
    "cpuTime"  : CPU time to calculate this shower [s]

  > profiles :
    "nX"       : number of points of "X", "N", "H", "D", "dEdX", "Mu", "dMu",
                 "Hadrons", "Gamma", and "Electrons" array
    "X"        : slant depth array (g/cm^2)
    "H"        : height array      (m)
    "D"        : distance array    (m)
    "N"        : array of number of charged particles above "hadCut" and "emCut" cut-off
                 crossing each X plane
    "dEdX"     : array of energy deposit (GeV/(g/cm^2)) in the MIDDLE of
                 each X bin, i.e. dE/dX[i] correspond to (X[i]+X[i+1])/2
    "Mu"       : array of number of muons above "muCut" cut-off crossing each X plane
    "dMu"      : array of number of muons produce above cut-off in each bin
    "Electrons": array of number of e^+ + e^- above "elCut" cut-off crossing each X plane
    "Gamma"    : array of number of gammas above "gaCut" cut-off crossing each X plane
    "Hadrons"  : array of number of hadrons above "haCut" cut-off crossing each X plane
    "EGround"  : Energy of particles at maximum X (EGround[0]=e+gamma;
                 EGround[1]=hadrons; EGround[2]=muons)

  If LEADING_INTERACTIONS_TREE is defined in conexConfig.h and K>0,
  branches of "LeadingInteractions" tree are :
    "nInt"     : Number of saved Interactions (size of array "kinel", "pId", "pEnergy",
                 "mult", "matg", "depth") per shower
    "kinel"    : Inelasticity for each saved interactions
    "pId"      : Id of parent particle (CONEX Id)
                 (CORSIKA Id if LEADING_INTERACTIONS_CORSIKA is defined in conexConfig.h )
    "pEnergy"  : Energy of parent particle for each saved interactions
    "mult"     : Multiplicity for each saved interactions
    "matg"     : Mass of target nucleus  for each saved interactions
    "depth"    : Slant depth (g/cm2) of each saved interactions
    "height"   : altitude above see level (m) of each saved interactions
    "nPart"    : total number of recorded secondary  particles per shower (nPart=sum[nInt](mult))
                 (size of array "Energy", "px", "py", "pz", "Type", and "idInt")
    "idInt"    : Interaction number
    "Type"     : Id of secondary particles (CONEX Id)
                 (CORSIKA Id if -DLEADING_INTERACTIONS_CORSIKA is selected in src/conexConfig.h)
    "Energy"   : total energy (GeV) of secondary particles
    "px","py","pz" : momentum (GeV/c) of secondary particles
                     (if -DLEADING_INTERACTIONS_TREE_EXT is used in src/conexConfig.h)



If CONEX_EXTENSIONS is used (uncomment the 8th line (#CONEX_EXTENSIONS+=-DCONEX_EXTENSIONS  #) 
in the Makefile), extra functionality becomes available. This is the list of options:

     flag         description           default value        Notes
   --------+------------------------+------------------+---------------
     -X       Cross-section parameter f19    1                 [#]

     -P       Extra factor for mesons fmeson 1                 [###]

     -R       Particle resampling mode       0                 [#]

              0 = OFF [default]
              1 = Multiplicity
              2 = Elasticity
              3 = EMRatio
	      4 = Charge Ratio
	      5 = Pi0 Spectrum                                 [##]
 
     -M       Resampling parameter f19       1                 [#]

     -T       Modification threshold, E_th  15                 [#]

     -L       Particle list mode             0                 [*]

              0  = OFF [default]
              >0 = write
              <0 = read

     -F       Particle list output file     ""                 [*]


These options are documented in details elsewhere:

[#] R. Ulrich et al., Phys.Rev. D83 (2011) 054026

[##] Technically this is implemented as in [#], but the spectral 
     power law index of the pi0 spectrum is reweighted. This might
     be useful for LHCf very forward data. Be careful with this
     option, just take it as a "suggestion".

[##] Thus for mesons the following effective f19_eff is used: 
     f19_eff = f19 * fmeson

[*] Here you can write special root output files with CONEX, which
    you can also read in with CONEX to proceed the air shower 
    simulation after e.g. changing parameters, etc. 
    This is compatible (and very similar in the idea) to the 
    STACKIN option of CORSIKA. See also CORSIKA User Guide. 

