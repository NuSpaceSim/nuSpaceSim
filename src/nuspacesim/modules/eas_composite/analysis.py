import matplotlib.pyplot as plt
import numpy as np
from .composite_ea_showers import CompositeShowers
from .fitting_composite_eas import FitCompositeShowers
#%% 

make_composites = CompositeShowers(shower_end=20000, grammage=1)
comp_showers, comp_depths, broken_event =  make_composites(
    filter_errors=False) 
#%%
trimmed_showers, _ = make_composites.shower_end_cuts(
    composite_showers=comp_showers, composite_depths=comp_depths, separate_showers=False)

#%%


print("Trimming {} showers.".format(np.shape(comp_showers)[0]))
# get the idx of the maxiumum particle content, skip the event number and decay code and offset
nmax_idxs = np.argmax(comp_showers[:,2:], axis=1) + 2 
# given the idxs of the max values, get the max values
nmax_vals = np.take_along_axis(comp_showers, nmax_idxs[:,None], axis=1)
# given the idxs of the max values, get the x max
xmax_vals = np.take_along_axis(comp_depths, nmax_idxs[:,None], axis=1) 
# set the rebound threshold
rebound_values = nmax_vals * 0.01 
print("Cutting shower rebounds past {}% of Nmax".format(0.01 *100))
# get rebound idxs 
x, y = np.where(comp_showers < rebound_values)
s = np.flatnonzero(np.append([False], x[1:] != x[:-1]))
less_than_thresh_per_evt = np.split(y, s)
# checking each event and getting the last idx where it is still less than the threshold
rebound_idxs = map(lambda x: x[-1], less_than_thresh_per_evt)
rebound_idxs = np.array(list(rebound_idxs))
# check for showers not going below the threshold and rebounding up into it
non_rebounding = rebound_idxs < nmax_idxs 
went_past_thresh = rebound_values < comp_showers[:,-1][:, None]
did_not_go_below_rebound_thresh = non_rebounding[:, None] & went_past_thresh
# for showers not going low enough, continue them till the end without any cuts,changes mask above
rebound_idxs[:, None][did_not_go_below_rebound_thresh] = np.shape(comp_showers)[1] - 1 
# from the rebound idxs on cutoff the shower
cut_off_mask = rebound_idxs[:,None] < np.arange(np.shape(comp_showers)[1])
# check for showers not reaching up into the threshold and were never cut short
full_shower_mask = rebound_idxs == np.shape(comp_showers)[1] - 1

comp_showers[cut_off_mask] = np.nan
full_showers = comp_showers[full_shower_mask, :]
trimmed_showers = comp_showers[~full_shower_mask, :] 
shallow_showers = comp_showers[did_not_go_below_rebound_thresh.flatten(), :]

full_depths = comp_depths[full_shower_mask, :]
trimmed_depths = comp_depths[~full_shower_mask, :] 
shallow_depths = comp_depths[did_not_go_below_rebound_thresh.flatten(), :]

trimmed_showers_reb_grammage = np.take_along_axis(
    trimmed_depths, rebound_idxs[~full_shower_mask][:,None], axis=1
    )
#print(np.count_nonzero(did_not_go_below_rebound_thresh))
print("There are {} full showers.".format(np.shape(full_showers)[0]))
print("There are {} trimmed showers".format(np.shape(trimmed_showers)[0]))
print("With cutoffs happening at {} {}".format(
    np.mean(trimmed_showers_reb_grammage), np.std(trimmed_showers_reb_grammage)))
print("There are {} shallow showers.".format(np.shape(shallow_showers)[0]))


#%%
# def shower_end_cuts (composite_showers, composite_depths):
#     #rom nuspacesim.utils.eas_cher_gen.composite_showers.composite_macros import bin_nmax_xmax

# get the idx of the maxiumum particle content, skip the event number and decay code and offset
nmax_positions = np.argmax(comp_showers[:,2:], axis=1) + 2 
# given the idxs of the max values, get the max values
nmax_vals = np.take_along_axis(comp_showers, nmax_positions[:,None], axis=1)
# given the idxs of the max values, get the x max
xmax_vals = np.take_along_axis(depths, nmax_positions[:,None], axis=1)  
# get rebound idxs 
rebound_values = nmax_vals * 0.01  
diff_bw_rebound = comp_showers[:,2:] - rebound_values 
rebound_positions = np.argmin(np.abs(diff_bw_rebound), axis=1)
#get showers that do not rebound, avoid catching the rising edge
non_rebounding = rebound_positions < nmax_positions 
print(np.count_nonzero(non_rebounding))
rebound_positions[non_rebounding] = np.shape(comp_showers)[1]  
mask = rebound_positions[:,None] < np.arange(np.shape(comp_showers)[1])

#comp_showers[mask] = np.nan
#%%
print("Trimming {} showers using a re".format(np.shape(comp_showers)[0]))
# get the idx of the maxiumum particle content, skip the event number and decay code and offset
nmax_idxs = np.argmax(comp_showers[:,2:], axis=1) + 2 
# given the idxs of the max values, get the max values
nmax_vals = np.take_along_axis(comp_showers, nmax_idxs[:,None], axis=1)
# given the idxs of the max values, get the x max
xmax_vals = np.take_along_axis(depths, nmax_idxs[:,None], axis=1)  
rebound_values = nmax_vals * 0.01
# get rebound idxs 
x, y = np.where(comp_showers < rebound_values)
s = np.flatnonzero(np.append([False], x[1:] != x[:-1]))
less_than_thresh_per_evt = np.split(y, s)
# checking each event and getting the last idx where it is still less than the threshold
rebound_idxs = map(lambda x: x[-1], less_than_thresh_per_evt)
rebound_idxs = np.array(list(rebound_idxs))
# check for showers not going below the threshold and rebounding up into it
non_rebounding = rebound_idxs < nmax_idxs 
went_past_thresh = rebound_values < comp_showers[:,-1][:, None]
did_not_go_below_rebound_thresh = non_rebounding[:, None] & went_past_thresh
# for showers not going low enough, continue them till the end without any cuts,changes mask above
rebound_idxs[:, None][did_not_go_below_rebound_thresh] = np.shape(comp_showers)[1] - 1 
# from the rebound idxs on cutoff the shower
cut_off_mask = rebound_idxs[:,None] < np.arange(np.shape(comp_showers)[1])
# check for showers not reaching up into the threshold and were never cut short
full_shower_mask = rebound_idxs == np.shape(comp_showers)[1] - 1

comp_showers[cut_off_mask] = np.nan
full_showers = comp_showers[full_shower_mask, :]
trimmed_showers = comp_showers[~full_shower_mask, :] 
shallow_showers = comp_showers[did_not_go_below_rebound_thresh.flatten(), :]

print(np.count_nonzero(did_not_go_below_rebound_thresh))
#%%

for depth, showers  in zip( depths[0:10,:], comp_showers[0:10,:]):
    event_num = depth[0]
    decay_code = depth[1]
   # print(np.max(showers[2:]*.01))
    #plt.hlines(np.max(showers[2:]*.01), 0,20000)
    plt.plot(depth[2:], showers[2:],'--', label = str(event_num)+"|"+ str(decay_code) )
#plt.ylim(top = .25e7)
plt.yscale('log')

plt.legend()
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

#%% 
# import matplotlib.pyplot as plt
# mask = (fit_results[:,2] != np.inf) 
# #filtered_test = fit_results[:,2][mask]
# histogram = (fit_results[:,2][mask])
# histogram = histogram [histogram < 10] 

# #%%
# mask1 = (fit_results[:,3] ==1  ) 
# masked_event_nums = fit_results[:,0][mask1]
# masked_decaycodes = fit_results[:,1][mask1]
# masked_chi = fit_results[:,2][mask1]
# masked_p_vals = fit_results[:,3][mask1]

# mask1 = np.array([masked_event_nums, masked_decaycodes, masked_chi, masked_p_vals]).T
# #%%
# plt.figure(figsize=(8, 5), dpi= 120)  
# plt.hist (histogram, bins = 30, edgecolor='black') 
# plt.title ('Distribution of Reduced Chisquare for the Fits')
# plt.ylabel('Counts')
# plt.xlabel('Reduced ChiSquare')
# #plt.xlim(0,10)