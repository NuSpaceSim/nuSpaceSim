import h5py 
import numpy as np 
import time  
from shower_long_profiles import ShowerParameterization
from fitting_composite_eas import FitCompositeShowers

np.seterr(all='ignore')
try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files
    
def numpy_argmax_reduceat(arr, group_idxs):
    r"""Get the indexes of maximum values within a grouping. 
    
    Parameters
    ----------
    arr: array
        flattened array of all the values to find the maximum of each group
    group_idxs: int
        indeces of the start of each group, can be found by np.unique

    Returns
    -------
    max_in_grp_idx: array
        indeces of the maximum of each grouping 
    
    Based on: https://stackoverflow.com/a/41835843
    """
    n = arr.max()+1  
    id_arr = np.zeros(arr.size,dtype=int)
    id_arr[group_idxs[1:]] = 1
    shift = n*id_arr.cumsum()
    sortidx = (arr+shift).argsort()
    grp_shifted_argmax = np.append(group_idxs[1:],arr.size)-1
    max_in_grp_idx = sortidx[grp_shifted_argmax] 
    return max_in_grp_idx

class CompositeShowers():    
    r""" Make composite showers with constituent electrons, gamma, and pions, 
    contributions scaled by sampled tau energies.  
    
    Parameters
    ----------
    shower_end: int
        where the shower ends, default:2000
    grammage: int
        size of slant depth bins in g/cm^2, default: 1

    Returns
    -------
    composite_shower: array 
        Shower content for each generated shower.
    slant_depths: array
        Corresponding slant depths to shower contents. 
    """
    def __init__(self, shower_end: int = 2000, grammage: int = 1):  
        
        with as_file(
            files('nuspacesim.data.conex_gh_params') / 'electron_EAS_table.h5'
        ) as path:
            data = h5py.File(path, 'r')
            electron_gh = np.array(data.get('EASdata_11'))
            
        with as_file(
            files('nuspacesim.data.conex_gh_params') / 'gamma_EAS_table.h5'
        ) as path:
            data = h5py.File(path, 'r')
            gamma_gh = np.array(data.get('EASdata_22'))
            
        with as_file(
            files('nuspacesim.data.conex_gh_params') / 'pion_EAS_table.h5'
        ) as path:
            data = h5py.File(path, 'r')
            pion_gh = np.array(data.get('EASdata_211'))        
            
        with as_file(
            files('nuspacesim.data.pythia_tau_decays') / 'new_tau_100_PeV.h5'
        ) as path:
            data = h5py.File(path, 'r')
            tau_decays = np.array(data.get('tau_data'))  
        
        self.electron_showers = electron_gh
        self.gamma_showers = gamma_gh
        self.pion_showers = pion_gh
        
        self.tau_tables = tau_decays
        
        # shower development characterisitics
        self.shower_end = shower_end
        self.grammage = grammage
        
    def conex_params(self):
        r""" Return sampled GH paramters for electrons, pions, and gammas.
        """
        return self.electron_showers, self.pion_showers, self.gamma_showers
        
    
    def tau_daughter_energies(self):
        r"""Isolate energy contributions of pion, kaon, electron, 
        and gamma daughters.
        
        Used to scale Nmax of respective GH showers.  
        
        """
        electron_mask = self.tau_tables[:,2] == 11
        electron_energies = self.tau_tables[electron_mask] [:,[0,1,-1]] 
    
        gamma_mask = self.tau_tables[:,2] == 22
        gamma_energies = self.tau_tables[gamma_mask ] [:,[0,1,-1]] 
        
        # kaons and pions treated the same 
        pion_kaon_mask = ( (self.tau_tables[:,2] == 211) | (
            self.tau_tables[:,2] == -211)
            ) | ( (self.tau_tables[:,2] == 321) | (
                self.tau_tables[:,2] == -321)
            )
        pion_energies = self.tau_tables[pion_kaon_mask] [:,[0,1,-1]]
        
        # each row has [event_num, energy ] 
        return electron_energies, pion_energies, gamma_energies 
        
    def single_particle_showers(self, tau_energies, gh_params, left_pad:int = 500): 
        r""" Create single a particle shower w/ Nmax scaled by pythia energy from same PID.
        Enables variable-- allowing negative-- shower starting points, left padded to uniform length.
    
        Parameters
        ----------
        tau_energies: float
            shower_energy, default table energy is 100 PeV
        gh_params: array
            CONEX GH output for one shower, for sample table layout see
            nuSpaceSim/src/nuspacesim/data/conex_gh_params/dat_data \
            /dumpGH_conex_gamma_E17_95deg_0km_eposlhc_830250265_22.dat
    
        Returns
        -------
        showers: array 
            shower content for one, non-composite shower
        depths: array
            corresponding slant depths
        """
        #padded_vec_len = self.shower_end/self.grammage + 200
        # pre-allocate arrays, make room for event tag and decay tag 
        showers = np.ones([gh_params.shape[0], int((self.shower_end/ self.grammage) + left_pad + 2)]
                            ) 
        depths = np.ones([gh_params.shape[0], int((self.shower_end/ self.grammage) + left_pad + 2)]
                            ) 
        # showers = []
        # depths = []
        for row,(shower_params,tau_dec_e) in enumerate(zip(gh_params, tau_energies)):
            
            shower = ShowerParameterization (
                table_decay_e=tau_dec_e[-1], event_tag=tau_dec_e[0], decay_tag=tau_dec_e[1]
            )
            
            depth, shower_content= shower.gaisser_hillas(
                                    n_max = shower_params[4],
                                    x_max = shower_params[5],
                                    x_0 = shower_params[6],
                                    p1 = shower_params[7],
                                    p2 = shower_params[8],
                                    p3 = shower_params[9],
                                    shower_end = self.shower_end,
                                    grammage = self.grammage)
            
            showers[row,:] = shower_content
            depths[row,:] = depth 
            # showers.append(shower_content)
            # depths.append(depth)
            
        return showers, depths
    
    
    def composite_showers(self, rebound_cut= None, **kwargs):
        r""" From single particle showers, create composite showers by summing each event. 
        For depths, selects the longest grammage vector. 
        # of particles with columns 0 and 1 being the event number and decay ID, repsectively.
        Uniform padded bins of the showers set by x_0, end_shower, and grammage.
    
        Parameters
        ----------
        single_showers: arrays
            uniform grammage arrays for each shower component
        shower_bins: array
            binds for each shower componenet for the composite
    
        Returns
        -------
        composite_showers: array 
            composite showers 
        composite_depths : array
            composite shower slant depths, padded on the left to uniform lengths   
        """
        
        # read in all arrays and get a main array containing all of the, sorted 
        single_showers = kwargs.get('single_showers')
        single_shower_bins = kwargs.get('shower_bins')
        
        single_showers = np.concatenate((single_showers), axis=0) 
        single_shower_bins = np.concatenate((single_shower_bins), axis=0) 
        
        single_showers = single_showers[single_showers[:,0].argsort()]
        single_shower_bins  = single_shower_bins[single_shower_bins[:,0].argsort()]
       
        # get unique event numbers, the index at which each event group starts 
        # and number of showers in each event
        grps, idx, num_showers_in_evt = np.unique(
            single_showers[:,0], return_index=True, return_counts=True, axis=0)
        unique_decay_codes = np.take(single_showers[:,1], idx)
        
        # sum each column up until the row index of where the new group starts and tack on the codes
        composite_showers = np.column_stack(
            (grps, unique_decay_codes, np.add.reduceat(single_showers[:, 2:], idx))
            )
        
        # per event, find the longest slant depth and use that since grammage is identical 
        sum_of_each_row = np.nansum(np.abs(single_shower_bins[:, 2:]), axis = 1)
        longest_shower_in_event_idxs = numpy_argmax_reduceat(sum_of_each_row, idx)

        composite_depths = np.take(single_shower_bins, longest_shower_in_event_idxs, axis=0)
        
        return  composite_showers, composite_depths 
    
    def shower_end_cuts (self, composite_showers, composite_depths):
        #rom nuspacesim.utils.eas_cher_gen.composite_showers.composite_macros import bin_nmax_xmax
        nmax_positions = np.argmax(composite_showers)
        
    
    def __call__ (self, filter_errors = False):
        r"""Loads CONEX GH parametrizations for electron, pion, and gamma.
        Makes single particle showers. Takes these showers and summs them per slant depth bin.
        Returns the paritlce showers and the slant depth for each composite in an array.
        Each padded to the smae length with nans due to variable starting points.
        
        Currently supports 100 PeV only. 
        """
        # get CONEX params and pythia tables.
        electron_gh, pion_gh, gamma_gh= self.conex_params()
        electron_e, pion_e, gamma_e = self.tau_daughter_energies()
        # make single particle showers
        elec_showers, elec_depths = self.single_particle_showers(
            tau_energies=electron_e, gh_params=electron_gh
            )
        pion_showers, pion_depths = self.single_particle_showers(
            tau_energies=pion_e, gh_params=pion_gh
            )
        gamm_showers, gamm_depths = self.single_particle_showers(
            tau_energies=gamma_e, gh_params=gamma_gh
            )
        # make composite showers 
        comp_showers, depths = self.composite_showers( 
            single_showers = (elec_showers, pion_showers, gamm_showers), 
            shower_bins = (elec_depths, pion_depths, gamm_depths)
        )
        
        # filter out showers where the parameterization fails; i.e. > np.inf or 1e20
        err_threshold = 1e100
        broken_showers_row_idx = np.where( np.any(comp_showers >  err_threshold , axis = 1) )
        broken_events = comp_showers[:,0:2][broken_showers_row_idx]

        
        print('{} Broken Showers out to {} g/cm^2'.format(len(broken_events), self.shower_end) )
        np.savetxt('discontinous_events.txt', broken_events, fmt='%1.0f')
        
        if filter_errors is True:
            comp_showers = np.delete(comp_showers, broken_showers_row_idx, axis=0)
            depths = np.delete(depths, broken_showers_row_idx, axis=0)
           
 
        
        return comp_showers, depths, broken_events, #pion_showers, pion_depths



if __name__ == '__main__': 
    t0 = time.time()
    make_composites = CompositeShowers()
    comp_showers, depths, broke =  make_composites() 
    
    get_fits = FitCompositeShowers(comp_showers, depths)
    fits = get_fits()
    
    # do next: lambda with a one percent cut.
    # show distributions of chi squares. 
    # constant lambda plots rebounds 
    reco_showers = np.full([comp_showers.shape[0], comp_showers.shape[1]], fill_value = -1) 
    fit_results = np.full([comp_showers.shape[0], 4], fill_value = np.nan) 
    
    for row,(params, depth) in enumerate(zip(fits, depths)):
            
        reconstructed = get_fits.reco_showers(fit_params=params, depth=depth)
        reco_showers[row,:] = reconstructed
        
    for row,(shower, shower_thoery) in enumerate(zip(comp_showers, reco_showers)):
        
        fit_chi = get_fits.reco_chi(shower, shower_thoery)
        fit_results[row,:] = fit_chi
    
   
    
    t1 = time.time()
    total = t1-t0 
    
    print(total)


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

    