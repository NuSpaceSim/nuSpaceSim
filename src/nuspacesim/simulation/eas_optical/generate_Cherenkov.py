from importlib.resources import as_file, files
import numpy as np
from scipy.constants import value,nano

class MakeYield:
    '''This class interacts with the table of Cherenkov yield ratios'''

    lXs = np.arange(-6,0)

    def __init__(self, l_min: float, l_max: float, npzfile = 'y_t_delta.npz'):
        self.l_min = l_min
        self.l_max = l_max
        self.l_mid = np.mean([l_min, l_max])

    def __repr__(self):
        return "Yield(l_min={:.2f} nm, l_max={:.2f} nm)".format(
        self.l_min, self.l_max)

    def find_nearest_interval(self, lX: float) -> tuple:
        '''This method returns the start and end points of the lX interval that
        the mesh falls within.
        '''
        index = np.searchsorted(self.lXs[:-1], lX)
        if index == 0:
            return self.lXs[0], self.lXs[1]
        else:
            return self.lXs[index-1], self.lXs[index]

    def get_npz_file(self, lX: float) -> str:
        '''This method returns the gg array file for the axis' particular
        log(moliere) interval.
        '''
        start, end = self.find_nearest_interval(lX)
        return f'y_t_delta_lX_{start}_to_{end}.npz'

    def set_yield_attributes(self, y: dict[str,np.ndarray]):
        '''This method sets the yield (as a function of stage and delta)
        attributes from the specified file.
        '''

        self.y_delta_t = y['y_t_delta'] * self.lambda_interval(self.l_min, self.l_max)
        self.delta = y['ds']
        self.t = y['ts']

    def set_yield_at_lX(self, lX: float):
        self.set_yield_attributes(self.get_npz_file(lX))

    def lambda_interval(self, l_min, l_max):
        '''This method returns the factor that results from integrating the
        1/lambda^2 factor in the Frank Tamm formula
        '''
        return 1 / (l_min * nano) - 1 / (l_max * nano)

    def y_of_t(self, t: float):
        '''This method returns the array of yields (for each delta) of the
        tabulated stage nearest to the given t
        '''
        return self.y_delta_t[np.abs(t - self.t).argmin()]

    def y(self, d: float, t: float):
        '''This method returns the average Cherenkov photon yield per meter
        per charged particle at a given stage and delta.
        '''
        return np.interp(d, self.delta, self.y_of_t(t))

    def y_list(self, t_array: np.ndarray, delta_array: np.ndarray):
        '''This method returns a list of average Cherenkov photon yields
        corresponding to a list of stages and deltas.
        Parameters:
        t_array: numpy array of stages
        delta_array: numpy array of corresponding deltas

        Returns: numpy array of yields
        '''
        y_array = np.empty_like(t_array)
        for i, (t,d) in enumerate(zip(t_array,delta_array)):
            y_array[i] = self.y(delta_array[i], t)
        return y_array
