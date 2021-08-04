import matplotlib.pyplot as plt
from nuSpaceSim.EAScherGen.Conex import conex_macros



def conex_plotter(file_name, data_name): 
   
    #regular plot some samples
    conex_macros.gh_profile_plot(file_name = file_name, data_name = data_name,
                                  start_row=0, end_row = 10,  regular_plot = True)
 
    #average plot all 
    conex_macros.gh_profile_plot(file_name = file_name, data_name = data_name,
                                  start_row=0, end_row = 999,  average_plot = True)
 
    #plot samples until a certain n_max threshold
    conex_macros.gh_profile_plot(file_name = file_name, data_name = data_name,
                                  start_row = 20, end_row = 20,
                                  n_max_cut_threshold = 0.01, x_limit= 10000,
                                  bins = 10000, rebound_plot = True) 

    #generating histograms
    n_maxs  = [] 
    x_maxs = []
    x_0s = []
    lambda_at_x_maxs = []
    threshold_depths = []
    
    for row in range (0, 1000): 
        n_max, x_max, x_0, lambda_at_x_max = conex_macros.gh_param_reader(
                                                    file_name = file_name, 
                                                    data_name = data_name,
                                                    row = row,
                                                    x_limit = 2000)[2:6]
        cutoff_depth = conex_macros.gh_profile_plot(file_name = file_name, 
                                                     data_name = data_name,
                                                     start_row = row,
                                                     end_row = row,  
                                                     n_max_cut_threshold = 0.01,
                                                     bins = 20000,
                                                     x_limit = 30000)        
        n_maxs .append (n_max)
        x_maxs.append (x_max)
        x_0s.append (x_0)
        lambda_at_x_maxs.append (lambda_at_x_max) 
        threshold_depths.append (cutoff_depth)
    
    
    conex_macros.parameter_histogram(param = n_maxs, title = 'Nmax Values ', x_label = 'Value', 
                                     color = 'royalblue')  
 
    conex_macros.parameter_histogram(param = x_maxs, title =  'Xmax Values ', x_label = 'Value', 
                                     color = 'orange') 

    conex_macros.parameter_histogram(param = x_0s, title = r'$X_{0}$' + ' Values ', x_label = 'Value', 
                                     color = 'green')  
 
    conex_macros.parameter_histogram(param = lambda_at_x_maxs , title =  'Lambda (t = Xmax) Values ', 
                                     x_label = 'Value', color ='purple')
 
    conex_macros.parameter_histogram(param = threshold_depths , 
                                      title = 'Cutoff Depths X (N $ > 0.01 \; Nmax$) ', 
                                      x_label = 'Value ($g \; cm^{-2}$) ', 
                                      color = 'crimson')
    
    plt.show() 
 
    
    

if __name__ == '__main__': 
    conex_plotter()
    