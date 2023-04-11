### The module: eas_composite
___
This module started off in Summer 2021, work by Fred Angelo Garcia at UMD under guidance of John, Toni, and Alex at GSFC.

Our initial approach was using CONEX GH Fits, located at the GDrive
> nuSpaceSim/CONEXresults/old_runs_from_john

```
dumpGH_conex_e_E17_95deg_15km_eposlhc_1576174973_11.dat
...
dumpGH_conex_pi_E17_95deg_5km_eposlhc_1131563074_211.dat
```

These have been processed into
> nuSpaceSim/src/nuspacesim/data/conex_gh_params
```
gh_00_km.h5  gh_05_km.h5  gh_15_km.h5  gh_20_km.h5
```
the name of which tells at what altitude the upward EAS is started (maybe observed), each has a key ("pion", "gamma", "electron", "proton") indicating the primary (initiating) particle of the shower. Recall, GH only fits the charged component of each shower. 

Within each key, there is a table with layout, left to right

``` 
# Primary
"Particle" : particle ID (100:proton, 11:electron, 13:mu, 211:pi, 22:gamma)
"lgE"      : log10 of the primary energy in eV
"zenith"   : zenith angle in degree
"azimuth"  : azimuth angle in degree (0 = East)
# GH-function
"Nmax"     : Number of charged particles above cut-off at the shower max
"Xmax"     : result for slant depth of the shower maximum (g/cm^2)
"X0"       : "pseudo" first interaction point for GH fit
"p1"       : first parameter for the polynomial 
"p2"       : second parameter for the polynomial 
"p3"       : third parameter for the polynomial 
"chi2"     : Chi squared/number of degree of freedom/sqrt (Nmax)
```

We also processed PYTHIA8 Runs for tau decays. 
