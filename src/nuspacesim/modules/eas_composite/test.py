import matplotlib.pyplot as plt
import numpy as np
from composite_ea_showers import CompositeShowers
from fitting_composite_eas import FitCompositeShowers

make_composites = CompositeShowers(shower_end=4000, grammage=1)
comp_showers, comp_depths, broken_event =  make_composites(filter_errors=False)  

# make_composites = CompositeShowers(shower_end=2000, grammage=1)

# electron_gh, pion_gh, gamma_gh= make_composites.conex_params()
# electron_e, pion_e, gamma_e = make_composites.tau_daughter_energies()

# gamm_showers, gamm_depths = make_composites.single_particle_showers(
#     tau_energies=gamma_e, gh_params=gamma_gh
#     )
# elec_showers, elec_depths = make_composites.single_particle_showers(
#     tau_energies=electron_e, gh_params=electron_gh
#     )

# pion_showers, pion_depths = make_composites.single_particle_showers(
#     tau_energies=pion_e, gh_params=pion_gh
#     )
#%%
# get_fits = FitCompositeShowers(comp_showers, comp_depths,)
# fits = get_fits()
#
for depths,showers  in zip( comp_depths[0:10,], comp_showers[0:10,]):
    event_num = depths[0]
    decay_code = depths[1]
    plt.plot(depths[2:], showers[2:],'--', label = str(event_num)+"|"+ str(decay_code) )
plt.yscale('log')

plt.legend()