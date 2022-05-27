import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy import stats

# relative imports are not for scripts, absolute imports here
from nuspacesim.simulation.eas_composite.composite_eas import CompositeShowers
from nuspacesim.simulation.eas_composite.comp_eas_utils import (
    bin_nmax_xmax,
    decay_channel_filter,
    separate_showers,
)
from nuspacesim.simulation.eas_composite.mc_mean_shwr import MCVariedMean
from nuspacesim.simulation.eas_composite.plt_routines import (
    mean_rms_plt,
    decay_channel_mult_plt,
    recursive_plt,
    get_decay_channel,
)

# initialize shower parameters
make_composites_00km = CompositeShowers(
    alt=0, shower_end=1e4, grammage=1, tau_table_start=1000
)

# call to make the composite showers and return particles as a function of depth
comp_showers_00km, comp_depths_00km = make_composites_00km(filter_errors=False)

# trim ends and seperate out by shower type: reached the threshold or not
# TODO: make grammage and slant depth x,y for all-- in same order,
# y goes first in the tuples
full, trimmed, shallow = make_composites_00km.shower_end_cuts(
    composite_showers=comp_showers_00km,
    composite_depths=comp_depths_00km,
    separate_showers=True,
)

dpths_300001, n_300001, not_300001_dpths, not_300001_n = decay_channel_filter(
    comp_depths_00km, comp_showers_00km, 300001, get_discarded=True
)
# intialize monte carlo sampler
mc = MCVariedMean(
    slant_depths=dpths_300001,
    composite_showers=n_300001,
    n_throws=400,
    hist_bins=30,
    sample_grammage=1,
)
# throw darts using intialized parameters, return a scaling factor,
# the sampled grammage, and the samples shower mean
mc_rms, sample_grm, sample_shwr, variability_dist = mc.sampling_nmax_once(
    return_rms_dist=True
)
