import matplotlib.pyplot as plt
from ..conex_gh import conex_macros


def conex_plotter(file_name, data_name, plt_type, start_row, end_row, x_limit,
                  bins, n_max_cut_threshold): 
    r"""Generate GH showers and optionally output plots, part of nuspacesim-misc.
    """

    #strt
    if plt_type.lower() == 'regular':
        #regular plot some samples
        conex_macros.gh_profile_plot(file_name = file_name, data_name = data_name,
                                      start_row = start_row, 
                                      end_row = end_row,
                                      x_limit = x_limit, 
                                      regular_plot = True)
    elif plt_type.lower() == 'average':
        #average plot all 
        conex_macros.gh_profile_plot(file_name = file_name, data_name = data_name,
                                      start_row = start_row, 
                                      end_row = end_row,  
                                      average_plot = True)
    elif plt_type.lower() == 'rebound':
        #plot samples until a certain n_max threshold
        conex_macros.gh_profile_plot(file_name = file_name, data_name = data_name,
                                      start_row =  
                                      start_row, end_row = end_row,
                                      n_max_cut_threshold = n_max_cut_threshold, 
                                      x_limit = x_limit,
                                      bins = bins, rebound_plot = True) 
    elif plt_type.lower() == 'histograms':
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
                                                         n_max_cut_threshold = n_max_cut_threshold,
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
    else: 
        print(plt_type, "is not a valid plot type")
    
    plt.show() 
 
    
    

# if __name__ == '__main__': 
#     conex_plotter()
#     #conex_plotter('gamma_EAS_table.h5', 'EASdata_22', 'regular', 1,2,5000,5000,.01)
    