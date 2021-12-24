import matplotlib.pyplot as plt
import numpy as np
from composite_ea_showers import CompositeShowers
from fitting_composite_eas import FitCompositeShowers
#%% 


make_composites = CompositeShowers(shower_end=3000, grammage=1)
comp_showers, depths, broken_event =  make_composites(
    filter_errors=False) 
#%%
# def shower_end_cuts (composite_showers, composite_depths):
#     #rom nuspacesim.utils.eas_cher_gen.composite_showers.composite_macros import bin_nmax_xmax
nmax_positions = np.argmax(comp_showers[:,2:], axis=1) + 2 
nmax_vals = np.take_along_axis(comp_showers, nmax_positions[:,None], axis=1)

xmax_vals = np.take_along_axis(depths, nmax_positions[:,None], axis=1)  

rebound_values = nmax_vals * 0.01  
diff_bw_rebound = comp_showers - rebound_values 
rebound_positions = np.argmin(np.abs(diff_bw_rebound), axis=1)
#%%
make_composites = CompositeShowers(shower_end=2000, grammage=1)
comp_showers, depths, broken_event =  make_composites(
    filter_errors=False)  
electron_gh, pion_gh, gamma_gh= make_composites.conex_params()
electron_e, pion_e, gamma_e = make_composites.tau_daughter_energies()

gamm_showers, gamm_depths = make_composites.single_particle_showers(
    tau_energies=gamma_e, gh_params=gamma_gh
    )
elec_showers, elec_depths = make_composites.single_particle_showers(
    tau_energies=electron_e, gh_params=electron_gh
    )

pion_showers, pion_depths = make_composites.single_particle_showers(
    tau_energies=pion_e, gh_params=pion_gh
    )
#%%




plt.figure(figsize=(8, 5), dpi= 120)


plt.plot(broken_gamm_depths[51, 2:3000],
         broken_gamm_showers[51, 2:3000],   
            c= 'tab:orange')

# plt.title('% Difference b/w Mean of Generated Composites and Mean of Composite Fits')
# plt.ylabel('% Difference')
# plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
# plt.xlim(left = 0 )         
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.yscale('log')

   #%%
   
def modified_gh(x, n_max, x_max, x_0, p1, p2, p3): 
 
 particles = (n_max * np.nan_to_num ( ((x - x_0) / (x_max - x_0))                  \
                                 **( (x_max - x_0)/(p1 + p2*x + p3*(x**2)) )  ) )   \
         *                                                                           \
         ( np.exp((x_max - x)/(p1 + p2*x + p3*(x**2))) )
         
 return particles

test_bin = np.linspace(0,30000,3000)
test_shower = modified_gh (test_bin, 
                           8.0913e+07,
                           7.2860e+02,
                           -8.9570e+01, 
                           4.7265e+01, -5.7572e-03,  2.8993e-06,
                           )
plt.plot(test_bin ,
         test_shower,   
            c= 'tab:red')
#%%
#no_nans =  comp_showers[np.isin (comp_showers [:,0], rebounding_evts)] 

#%% get rebound x
cut_off_depths = np.empty([np.shape(comp_showers)[0]])
for row,(shower,bins) in enumerate(zip(comp_showers,depths)):
    max_val = np.max(shower) 
    threshold = max_val * 0.01
    # max_pos = int(np.argwhere (shower == max_val))
    
    # shower =  shower[max_pos:]
    # bins = bins[max_pos:]
    
    # shower_mask = (shower < threshold)
    # masked_shower = shower[shower_mask]
    # masked_depths = bins[shower_mask]
    # #print(masked_depths)
    places_to_cut = np.argwhere(shower[2:] <= threshold) 
    #highest index where it is still less than (after or before peak, depends on x_limit) 
    cut_here = np.amax(places_to_cut) 
    #trims the arrays from 0 to where number of  particles < n_max_fraction
    trimmed_f = np.array (shower[0:cut_here] )
    trimmed_x = np.array (bins[0:cut_here] )
    #get the slant depth when f exceeds n_max_fraction
    cutoff_depth = np.amax(trimmed_x[2:]) 
    cut_off_depths[row] = cutoff_depth
    #print(cutoff_depth )
#%%
plt.figure(figsize=(8, 5), dpi= 120)  
mask = cut_off_depths < 10000 
mask_array = np.tile(mask, (comp_showers.shape[1], 1)).T 
#tile(array([[1,2,3]]).transpose(), (1, 3))
plt.hist (cut_off_depths[mask], bins = 30) 
plt.title ('cutoff depths for showers that rebound within 10,000 gcm^2')
plt.ylabel('Counts')
plt.xlabel('Depth (g/cm^2)')

rebounding_evts = comp_showers[:,0] [mask]
#filtered_comp_showers = np.ma.masked_array(comp_showers, mask=mask_array)
filtered_comp_showers  = comp_showers[np.isin (comp_showers [:,0], rebounding_evts)] 
#filtered_comp_showers = comp_showers[~np.array(mask_array)]
#%%
reference_make_composites = CompositeShowers(shower_end=2000, grammage=1)
reference_comp_showers, reference_depths =  reference_make_composites()    
#comp_showers= filtered_comp_showers 
# mask = reference_comp_showers != np.inf
get_fits = FitCompositeShowers(reference_comp_showers, reference_depths)
fits = get_fits()

# filtered_fits = fits[np.isin (fits[:,0], rebounding_evts)] 

#%%

# def giasser_hillas( x, n_max, x_max, x_0, gh_lambda): 
    
#     particles = (n_max * np.nan_to_num ( ((x - x_0) / (x_max - x_0))  \
#                                     **((x_max - x_0)/gh_lambda) )  )   \
#             *                                                           \
#             ( np.exp((x_max - x)/gh_lambda) )    
    
#     return particles

plt.figure(figsize=(8, 5), dpi= 120)
plt.plot(depths[2,2:], comp_showers[2,2:], label = 'generated composite shower')
# plt.plot(depths[4,2:],giasser_hillas(depths[4,2:], *fits[284,2:]), '--',
#          label ='shower fit, const. lambda, event 337 decay id 300001')
plt.yscale('log')
#plt.ylim(bottom = 1)
plt.legend()
#plt.draw()

#%% Plot Mean and RMS of Generated Showers


#comp_showers= filtered_comp_showers
plt.figure(figsize=(8, 5), dpi= 120)
average_composites = np.mean(comp_showers[:,2:], axis=0) #Takes the mean of each row (each bin)

square = np.square (comp_showers.T) #squares the 2D matrix  
mean = square.mean(axis = 1) #gets mean along each bin
rms =  np.sqrt(mean) #gets root of each mean

rms_error = np.sqrt(np.mean( ( average_composites  - comp_showers[:,2:])**2, axis = 0 ))


plt.figure(figsize=(8, 5), dpi= 120)
plt.plot(depths[1,2:], average_composites,  '--k', label= 'Mean Generated Showers') 

plt.fill_between(depths[1,2:], average_composites - rms_error, average_composites+ rms_error,
                  alpha = 0.5, edgecolor='red', facecolor='crimson', 
                  label = 'Error')

# #plt.title('G-H Plots | ' + str(file_name.split('/')[-1]) +'/'+ str(data_name))
# plt.ylabel('Number of Particles N')
# plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
# plt.xlim(left = 0 )         
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  
#plt.yscale('log')
plt.xlim(0,2000) 
plt.legend()

#%% Plot Mean and RMS of Fitted Showers
#comp_showers= filtered_comp_showers 

def giasser_hillas( x, n_max, x_max, x_0, gh_lambda): 
    
    particles = (n_max * np.nan_to_num ( ((x - x_0) / (x_max - x_0))  \
                                    **((x_max - x_0)/gh_lambda) )  )   \
            *                                                           \
            ( np.exp((x_max - x)/gh_lambda) )    
    
    return particles

reference_make_composites = CompositeShowers(shower_end=2000, grammage=1)
reference_comp_showers, reference_depths, broken =  reference_make_composites( filter_errors=False)    
#comp_showers= filtered_comp_showers 
# mask = reference_comp_showers != np.inf
get_fits = FitCompositeShowers(reference_comp_showers, reference_depths)
fits = get_fits()
#%%

fitted_showers = np.ones([reference_comp_showers.shape[0], 2000]) 

for row,fit in enumerate(fits):
    fitted_showers[row,:] = giasser_hillas(reference_depths[row,2:], *fits[row,2:])
    


plt.figure(figsize=(8, 5), dpi= 120)
fit_average_composites = np.mean(fitted_showers, axis=0) #Takes the mean of each row (each bin)
# #to get RMS
square = (fitted_showers .T)**2 #squares the 2D matrix  
mean = np.mean(square, axis = 1) #gets mean along each bin
rms =  np.sqrt(mean) #gets root of each mean

rms_error = np.sqrt(np.nanmean( ( fit_average_composites  - fitted_showers )**2, axis = 0 ))



plt.plot(reference_depths[1,2:], fit_average_composites,  '--k', label= 'Fitted Showers Mean') 

# plt.fill_between(np.linspace(0,2000,2000), 
#                  fit_average_composites - rms_error, 
#                  fit_average_composites+ rms_error,
#                   alpha = 0.5, edgecolor='green', facecolor='green', 
#                   label = 'Error')

# #plt.title('G-H Plots | ' + str(file_name.split('/')[-1]) +'/'+ str(data_name))
plt.ylabel('Shower Content')
plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
plt.xlim(left = 0 )         
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  
#plt.yscale('log')
plt.legend() 
#plt.xlim(0,2000)
#plt.ylim(1,1e8)
#%%
plt.figure(figsize=(8, 5), dpi= 120)
percent_diff = np.abs(average_composites - fit_average_composites) / \
               ( (average_composites + fit_average_composites)/2 )

bins  = depths[1,2:]

plt.scatter(bins[1::20], 
            percent_diff[1::20],  
            c= 'tab:orange', 
            alpha = 0.5,
            label= ' Percentage Difference Between Shower Means')

plt.title('% Difference b/w Mean of Generated Composites and Mean of Composite Fits')
plt.ylabel('% Difference')
plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
plt.xlim(left = 0 )         
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))