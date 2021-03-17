"""
sim_store: NuSpaceSim module for storing simulation intermediates in HDF5
Files.

Design requirements:
    1 file per simulation.
    1 top level attribute (metadata or header info) containing top level
        configuration parameters.
    1 group per simulation module. (heirarchical subdivision, like a folder or
        dictionary)
    1 dataset per result vector from module.
    Masks???
"""

import h5py
import datetime


class SimStore():

    def __init__(self, config, filename=None):

        now = datetime.datetime.now()

        if filename is None:
            filename = f'Simulation_run_{now:%Y%m%d%H%M%S}.hdf5'

        self.store = h5py.File(filename, 'w')
        self.store.attrs['earthRadius'] = config.EarthRadius
        self.store.attrs['N'] = config.N
        self.store.attrs['logNuTauEnergy'] = config.logNuTauEnergy
        self.store.attrs['nuTauEnergy'] = config.nuTauEnergy
        self.store.attrs['simStartTime'] = f'{now:%Y%m%d%H%M%S}'

    def __del__(self):
        self.close()

    def close(self):
        self.store.close()

    def __call__(self, group_name, dataset_name_list, *datasets):
        """
        Insert data into the named group, with the names corresponding to the
        values in dataset_name_list.
        """

        grp = self.store[group_name] if group_name in self.store else self.store.create_group(group_name)

        for n, d in zip(dataset_name_list, datasets):
            grp.create_dataset(n, data=d)
