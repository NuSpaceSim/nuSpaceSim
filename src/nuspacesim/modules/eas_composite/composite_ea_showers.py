import h5py 
import importlib_resources
import numpy as np 

#what is need to create composite
#shower parameters of each type
#isolated pythia energies to scale each shower Nmax
#make scaled showers 
#sum showers for each event

try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files
 
class ShowerParameterization: 
    def __init__(self, table_decay_e, event_tag):
        
        # pythia related params
        self.table_decay_e = table_decay_e
        self.event_tag = event_tag
        
    def greisen(self, 
                shower_end = 2000, 
                grammage = 1, 
                table_energy = 100e15):
    
        #scaled_gh_n_max = gh_n_max * self.table_decay_e
        
        x = np.linspace( 1, shower_end, int(shower_end / grammage))  #slant depths 
        
        # source: https://pdg.lbl.gov/2015/AtomicNuclearProperties/HTML/air_dry_1_atm.html
       
        # x0 is the radiation length in air ~ 37 g/cm^2
        x_naught = 36.62 
        
        #critical energy for electromagnetic cascades in the air
        e_c = 87.92e6 
        
        # energy of photon initiated shower (eV) Gaisser 303 
        e_0 = self.table_decay_e * table_energy
        beta_0 = np.log(e_0 / e_c)
        t = x / x_naught 
        
        # age parameter, n = 0
        s = (0 + 3*t)/ (t + 2*beta_0 ) 
       
        x_max = x_naught * np.log(e_0 / e_c)
        n_max = ( 0.31 / pow(beta_0, 1/2) ) * (e_0 / e_c)
    
        term1 = ( 0.31 / pow(beta_0, 1/2) )
        term2 = np.exp ( t - (t * (3/2)  * np.log (s) ) )
        
        content = term1 * term2 # particle content 
         
        return x, content, self.event_tag
        
    def gaisser_hillas(self, 
                       n_max, 
                       x_max, 
                       x_0, 
                       p1, 
                       p2, 
                       p3, 
                       shower_end = 2000, 
                       grammage = 1):
        
        scaled_n_max = n_max * self.table_decay_e
        
        # allows negative starting depths
        # bins = int ( (shower_end + np.abs(x_0) ) / grammage)
        # x = np.linspace( int(x_0), shower_end, bins, endpoint=False) #slant depths g/cm^2 
        
        # constrains starting depths 
        x = np.linspace( 1, shower_end, shower_end) #slant depths g/cm^2    
        
        #calculating gaisser-hillas function
        gh_lambda = p1 + p2*x + p3*(x**2) 
        
        exp1 = (x_max - x_0) / gh_lambda
       
        term1 = scaled_n_max * np.nan_to_num ( ((x - x_0) / (x_max - x_0)) **exp1 )
        
        exp2 = (x_max - x) / gh_lambda
        term2 = np.exp(exp2) 
        
        f = np.nan_to_num (term1 * term2) 
        
        #LambdaAtx_max = p1 + p2*x_max + p3*(x_max**2)
        #t = (x - x_max)/36.62 #shower stage
         
        return x, f, self.event_tag
    
    
class CompositeShowers():    
    r""" Make composite showers with constituent electrons, gamma, and pions, contributions scaled
    by sampled tau energies. 
    
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
    
    def tau_daughter_energies(self):
        r"""Isolate energy contributions of pion, kaon, electron, and gamma daughters.
        Used to scale Nmax of respective GH showers.  
        
        """
        electron_mask = self.tau_tables[:,2] == 11
        electron_energies = self.tau_tables[electron_mask] [:,[0,-1]] 
    
        gamma_mask = self.tau_tables[:,2] == 22
        gamma_energies = self.tau_tables[gamma_mask ] [:,[0,-1]] 
        
        # kaons and pions treated the same 
        pion_kaon_mask = ((self.tau_tables[:,2] == 211) | (self.tau_tables[:,2] == -211)) | \
                         ((self.tau_tables[:,2] == 321) | (self.tau_tables[:,2] == -321))
        pion_energies = self.tau_tables[pion_kaon_mask] [:,[0,-1]]
        
        # each row has [event_num, energy ] 
        return electron_energies, gamma_energies, pion_energies 
    
    def single_particle_showers(self, gh_params, tau_energies): 
        
        showers = np.empty([ gh_params.shape[0], self.shower_end/ self.grammage]) 
        depths =  np.empty([ gh_params.shape[0], self.shower_end/ self.grammage]) 
        
        for row in gh_params:
            shower = ShowerParameterization (
                table_decay_e =  tau_energies[row],event_tag = tau_energies[row] 
            )
            
            depth, shower_content, event_num = shower.gaisser_hillas(n_max = gh_params[row, 4],
                                                                      x_max = gh_params[row, 5],
                                                                      x_0 = gh_params[row, 6],
                                                                      p1 = gh_params[row, 7],
                                                                      p2 = gh_params[row, 8],
                                                                      p3 = gh_params[row, 9],
                                                                      shower_end = self.shower_end,
                                                                      grammage = self.shower_end)
            showers[row,:] = shower_content
            depths[row,:] = depth 
            
        return depth, shower_content, event_num
            
    
    
    
    
    # with importlib_resources.as_file(gamma) as path:
    #     data = h5py.File(path, 'r')
    #     gamma_gh = np.array(data.get('EASdata_22'))
        
    # with importlib_resources.as_file(pion) as path:
    #     data = h5py.File(path, 'r')
    #     pion_gh = np.array(data.get('EASdata_211'))
        
    # with importlib_resources.as_file(pythia_output) as path:
    #     data = h5py.File(path, 'r')
    #     tau_decays = np.array(data.get('tau_data'))
        
    #return electron_gh, gamma_gh, pion_gh, tau_decays

#test1, test2, test3, test4 = load_data()

if __name__ == '__main__': 
    x = CompositeShowers()
    electron,gamma,pion = x.tau_daughter_energies()