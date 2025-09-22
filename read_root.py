import argparse
import glob
import os
import uproot
import numpy as np
import sys
from pathlib import Path
from matplotlib.colors import LogNorm

sys.path.append(os.path.join(os.path.dirname(__file__), "src", "nuspacesim"))

from augermc import *

# --- Parse arguments ---

parser = argparse.ArgumentParser(description="Process ROOT files")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-d", "--directory", help="Directory containing ROOT files")
group.add_argument("-f", "--file", help="Single ROOT file")
parser.add_argument(
    "-p", "--plots",
    default="plots",
    help="Directory to save plots (default: ./plots)"
)

args = parser.parse_args()

# --- Create plots dir ---
plots_dir = Path(args.plots)
plots_dir.mkdir(parents=True, exist_ok=True)
args = parser.parse_args()

# --- Collect files ---
if args.directory:
    root_files = sorted(glob.glob(os.path.join(args.directory, "*.root")))
    if not root_files:
        raise FileNotFoundError(f"No .root files found in {args.directory}")
else:
    if not os.path.isfile(args.file):
        raise FileNotFoundError(f"File not found: {args.file}")
    root_files = [args.file]

# --- Check available branches ---
with uproot.open(f"{root_files[0]}:Shower") as tree:
    available = set(k.split(";")[0] for k in tree.keys())
branches = [
    "easting", "northing", "height", "zonenumber", "zoneletter",
    "telescopeid", "zenith", "azimuth", "lgE", "Hfirst",
    "Xfirst", "X", "H", "N", "ExitProb", "lgnuE", "n_tel",
    "xmaxecef0", "xmaxecef1", "xmaxecef2",
    "xstartecef0", "xstartecef1", "xstartecef2",   
]
data_root = uproot.concatenate(
    [f"{f}:Shower" for f in root_files],
    expressions=branches,
    library="np"
)

east = data_root.get("easting")
north = data_root.get("northing")
altitude = data_root.get("height")
zone = data_root.get("zonenumber")
lettercoded = data_root.get("zoneletter")
band = np.array([chr(c) for c in lettercoded]) if lettercoded is not None else None
telid = data_root.get("telescopeid")
zenith = data_root.get("zenith")
azimuth = data_root.get("azimuth")
if azimuth is not None:
    azimuth = (azimuth + 180) % 360
lgE = data_root.get("lgE")
Hfirst = data_root.get("Hfirst")
Xfirst = data_root.get("Xfirst")
X = data_root.get("X")
H = data_root.get("H")
RN = data_root.get("N")
exitprob = data_root.get("ExitProb")
lgnuE = data_root.get("lgnuE")
n_tel = data_root.get("n_tel")
xmaxecef0 = data_root.get("xmaxecef0")
xmaxecef1 = data_root.get("xmaxecef1")
xmaxecef2 = data_root.get("xmaxecef2")
xmaxecef = np.vstack([xmaxecef0, xmaxecef1, xmaxecef2]).T if all(x is not None for x in [xmaxecef0, xmaxecef1, xmaxecef2]) else None
xstart0 = data_root.get("xstartecef0")
xstart1 = data_root.get("xstartecef1")
xstart2 = data_root.get("xstartecef2")
xstartecef = np.vstack([xstart0, xstart1, xstart2]).T if all(x is not None for x in [xstart0, xstart1, xstart2]) else None
beta = zenith - 90


mean_lat = -35.20965731232134
mean_lon = -69.31811672662049
LLang = np.radians(330 - 360)
LLog = np.array([459208.3, 6071871.5, 1416.2])
LMog = np.array([498903.7, 6094570.2, 1416.4])
LAog = np.array([480743.1, 6134058.4, 1476.7])
COog = np.array([445343.8, 6114140.0, 1712.3])
easting_values = [LLog[0], LMog[0], LAog[0], COog[0]]
northing_values = [LLog[1], LMog[1], LAog[1], COog[1]]
z_values = [LLog[2], LMog[2], LAog[2], COog[2]]
mean_easting = np.mean(easting_values)
mean_northing = np.mean(northing_values)
mean_height = np.mean(z_values)
centerarray = np.array([mean_easting, mean_northing, mean_height])
center = np.array([LLog[0], LLog[1], LLog[2]])
centerarray_respLL = centerarray - center
LL = LLog - center
LM = LMog - center
LA = LAog - center
CO = COog - center
maskoutzone = (zone != 19)
maskoutletter = (lettercoded != 72)
inzoneandletter = (zone == 19) & (lettercoded == 72)

latcore, loncore, heightcore= utm_to_geodetic(east,north,zone,band, altitude)
LLlat,LLlong, LLheight=utm_to_geodetic(LLog[0], LLog[1], 19, 'H', LLog[2])
LLlat=LLlat[0]
LLlong=LLlong[0]
LLheight=LLheight[0]
coreecef = latlongtoECEF(latcore, loncore, heightcore)
LLecef = latlongtoECEF(LLlat, LLlong, LLheight)
m = np.tan(LLang)
LLangx_vals = np.linspace(-50000, 50000, 100)
LLangy_vals = m * LLangx_vals

vecef=(xstartecef-coreecef)/np.linalg.norm((xstartecef-coreecef),axis=1)[:,np.newaxis]

coreenu=eceftoenu(LLecef,coreecef,lat=LLlat,lon=LLlong)
xmaxenu=eceftoenu(LLecef,xmaxecef,LLlat,LLlong)
xstartenu=eceftoenu(LLecef,xstartecef,LLlat,LLlong)

coredist=np.linalg.norm(coreenu,axis=1)
xmaxdist=np.linalg.norm(xmaxenu,axis=1)
xstartdist=np.linalg.norm(xstartenu,axis=1)

xmaxdist[np.isnan(xmaxdist)] = 1.1 * np.nanmax(xmaxdist)
trigger=(n_tel>50)


def plot_trajectories(core,start,xmax,mask,idx,ntoplot, label):

    xxmaxtrig=xmax[mask,0]
    yxmaxtrig=xmax[mask,1]
    
    xcoretrig=core[mask,0]
    ycoretrig=core[mask,1]
    
    xstartrig=start[mask,0]
    ystartrig=start[mask,1]
    
    plt.figure(figsize=(12,12), dpi=300)

    plt.plot(0,0,color='red',marker='v',markersize=7,zorder=10)
    plt.plot(centerarray_respLL[0],centerarray_respLL[1],color='black',marker='x',markersize=5)

    plt.scatter(xxmaxtrig[idx], yxmaxtrig[idx], color='orange', marker='x', s=5, label='Xmax position')
    plt.scatter(xstartrig[idx], ystartrig[idx], color='green', marker='x', s=5, label='Shower start position')
 
    plt.plot(np.vstack([xcoretrig[idx], xxmaxtrig[idx]]),
             np.vstack([ycoretrig[idx], yxmaxtrig[idx]]),
             color="blue", linestyle='--', linewidth=0.5, alpha=0.2)

    plt.scatter(xcoretrig[idx], ycoretrig[idx], color='blue', marker='o', s=10, label='Core')

    for i in idx:
        plt.text(xcoretrig[i], ycoretrig[i] + 1000,
                 f"{lgE[mask][i]:.1f}, {zenith[mask][i]-90:.1f}",
                 ha='center', fontsize=6, zorder=10)

    plt.legend()
    plt.gca().set_aspect('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'map of {label} triggering events around Los Leones, n={ntoplot}')
    plt.plot(LLangx_vals, LLangy_vals, 'r--', linewidth=0.5, label=f'LL line of sight')

    plt.savefig(plots_dir / f'triggers_core+xmax_n{ntoplot}_{label}.png')
    plt.close()

def plot_colormaps_all_trigg(coords, label, nbins=20, size=40e3, log=False):
    """
    Generate two 2D histograms of core positions:
    1) All events
    2) Triggered events
    
    Parameters
    ----------
    coords : ndarray of shape (n, 3)
        Array containing xcorecentered, ycorecentered, zcorecentered.
    label : str
        Label used for plot titles and output filenames.
    nbins : int, optional (default=40)
        Number of bins on each axis for the histogram.
    size : float, optional (default=40e3)
        Half-size of the axis range around LL in meters.
    log : bool, optional (default=False)
        If True, use logarithmic color scale.
    """
    xcorecentered = coords[:, 0]
    ycorecentered = coords[:, 1]

    # Define bins
    x_bins = np.linspace(LL[0] - size, LL[0] + size, nbins)
    y_bins = np.linspace(LL[1] - size, LL[1] + size, nbins)

    # Pick normalization
    norm = LogNorm() if log else None

    # ---------- Plot 1: all cores ----------
    plt.figure(figsize=(12, 12))

    hist, xedges, yedges, im = plt.hist2d(
        xcorecentered, ycorecentered,
        bins=[x_bins, y_bins], cmap='viridis', cmin=1, norm=norm
    )

    plt.colorbar(im, label='Number of Events')
    plt.plot(LL[0], LL[1], color='red', marker='v', markersize=7, zorder=10, label="LL")
    plt.plot(centerarray_respLL[0], centerarray_respLL[1], color='black', marker='x', markersize=5, label="Center Array")
    plt.plot(LLangx_vals, LLangy_vals, 'r--', linewidth=0.5, label='LL line of sight')

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'{label} position colormap of all events (n={len(xcorecentered)})')
    plt.legend()
    plt.gca().set_aspect('equal')

    plt.tight_layout()
    plt.savefig(plots_dir / f'{label}_all_colormap{"_log" if log else ""}.png')
    plt.close()

    # ---------- Plot 2: triggered cores ----------
    plt.figure(figsize=(12, 12))

    hist, xedges, yedges, im = plt.hist2d(
        xcorecentered[trigger], ycorecentered[trigger],
        bins=[x_bins, y_bins], cmap='viridis', cmin=1, norm=norm
    )

    plt.colorbar(im, label='Number of Events')
    plt.plot(LL[0], LL[1], color='red', marker='v', markersize=7, zorder=10, label="LL")
    plt.plot(centerarray_respLL[0], centerarray_respLL[1], color='black', marker='x', markersize=5, label="Center Array")
    plt.plot(LLangx_vals, LLangy_vals, 'r--', linewidth=0.5, label='LL line of sight')

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'{label} position colormap of triggered events (n={len(xcorecentered[trigger])})')
    plt.legend()
    plt.gca().set_aspect('equal')

    plt.tight_layout()
    plt.savefig(plots_dir / f'{label}_triggered_colormap{"_log" if log else ""}.png')
    plt.close()


def generate_hists():
    # Variable histograms
    variables = {
        'lgE': {'data': lgE, 'label': 'Log Energy (lgE)', 'xlabel': 'lgE'},
        'zenith': {'data': zenith, 'label': 'Zenith Angle (degrees)', 'xlabel': 'Zenith (degrees)'},
        'azimuth': {'data': azimuth, 'label': 'Azimuth Angle (degrees)', 'xlabel': 'Azimuth (degrees)'},
        #'opening_angle': {'data': openingangle, 'label': 'Opening Angle from Xstart (degrees)', 'xlabel': 'Opening Angle (degrees)'},
        #'opening_angle_core': {'data': openinganglecore, 'label': 'Opening Angle from Core (degrees)', 'xlabel': 'Opening Angle (degrees)'},

    }

    for var_name, var_info in variables.items():
        plt.figure(figsize=(8, 10))
        data_range = (np.min(var_info['data']), np.max(var_info['data']))
        bins = np.linspace(data_range[0], data_range[1], 51)
        ax1 = plt.subplot(2, 1, 1)
        n_all = len(var_info['data'])
        mean_all = np.mean(var_info['data'])
        min_all = np.min(var_info['data'])
        max_all = np.max(var_info['data'])
        plt.hist(var_info['data'], bins=bins, color='blue', alpha=0.7, label=(
            f'All Events\nMean: {mean_all:.2f}\nMin: {min_all:.2f}\nMax: {max_all:.2f}'
        ))
        plt.ylabel('Number of Events')
        plt.title(f'Histogram of {var_info["label"]} (All Events, n={n_all})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 1, 2, sharex=ax1)
        triggered_data = var_info['data'][trigger]
        n_triggered = len(triggered_data)
        if n_triggered > 0:
            mean_triggered = np.mean(triggered_data)
            min_triggered = np.min(triggered_data)
            max_triggered = np.max(triggered_data)
            plt.hist(triggered_data, bins=bins, color='green', alpha=0.7, label=(
                f'Triggered Events\nMean: {mean_triggered:.2f}\nMin: {min_triggered:.2f}\nMax: {max_triggered:.2f}'
            ))
        else:
            plt.text(0.5, 0.5, 'No Triggered Events', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
        plt.xlabel(var_info['xlabel'])
        plt.ylabel('Number of Events')
        plt.title(f'Histogram of {var_info["label"]} (Triggered Events, n={n_triggered})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f'{var_name}_hist.png')
        plt.close()

def generate_dist_hists():
    # Variable histograms
    variables = {
        'xmaxdist': {'data': xmaxdist, 'label': 'Distance to Xmax', 'xlabel': 'Xmax distance (m)'},
        'coredist': {'data': coredist, 'label': 'Distance to core', 'xlabel': 'Core distance (m)'}
    }
    lim=np.max(coredist)
    for var_name, var_info in variables.items():
        plt.figure(figsize=(8, 10))
        data_range = (0, lim)
        bins = np.linspace(data_range[0], data_range[1], 51)
        ax1 = plt.subplot(2, 1, 1)
        n_all = len(var_info['data'])
        mean_all = np.mean(var_info['data'])
        min_all = np.min(var_info['data'])
        max_all = np.max(var_info['data'])
        plt.hist(var_info['data'], bins=bins, color='blue', alpha=0.7, label=(
            f"All Events\nMean: {mean_all:.2f}\nMin: {min_all:.2f}\nMax: {max_all:.2f}\n Events out of plot: {np.sum(var_info['data']>lim)}"
        ))
        plt.ylabel('Number of Events')
        plt.title(f'Histogram of {var_info["label"]} (All Events, n={n_all})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 1, 2, sharex=ax1)
        triggered_data = var_info['data'][trigger]
        n_triggered = len(triggered_data)
        if n_triggered > 0:
            mean_triggered = np.mean(triggered_data)
            min_triggered = np.min(triggered_data)
            max_triggered = np.max(triggered_data)
            plt.hist(triggered_data, bins=bins, color='green', alpha=0.7, label=(
                f"Triggered Events\nMean: {mean_triggered:.2f}\nMin: {min_triggered:.2f}\nMax: {max_triggered:.2f}\n Events out of plot: {np.sum(triggered_data>lim)}"
            ))
        else:
            plt.text(0.5, 0.5, 'No Triggered Events', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
        plt.xlabel(var_info['xlabel'])
        plt.ylabel('Number of Events')
        plt.title(f'Histogram of {var_info["label"]} (Triggered Events, n={n_triggered})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f'{var_name}_hist.png')
        plt.close()

ntoplot = 30

# Sort indices of triggered events by core distance
sorted_idx = np.argsort(coredist[trigger])

# Closest and farthest subsets
idx_random=np.arange(ntoplot)
idx_close = sorted_idx[:ntoplot]
idx_far   = sorted_idx[-ntoplot:]

plot_trajectories(coreenu,xstartenu,xmaxenu,trigger,idx_random,ntoplot, "example")
plot_trajectories(coreenu,xstartenu,xmaxenu,trigger,idx_close,ntoplot, "close")
plot_trajectories(coreenu,xstartenu,xmaxenu,trigger,idx_far,ntoplot, "far")

plot_colormaps_all_trigg(coreenu,'Core')
plot_colormaps_all_trigg(coreenu,'Core', log=True)

plot_colormaps_all_trigg(xmaxenu,'Xmax')
plot_colormaps_all_trigg(xmaxenu,'Xmax', log=True)

generate_hists()
generate_dist_hists()