import src.hp_models as models 
import src.hp_utils as utils 
from collections import defaultdict
import matplotlib.pyplot as plt 
import src.plot_utils as p_utils 
import xarray as xr
import numpy as np
import time 
from pathlib import Path
import pickle 
from scipy.interpolate import interp1d
import warnings


with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)


BASE_PATH = '/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/'

models_dict = models.models_name_dict
MODEL_NAMES, REGION_CFG, COLORS = p_utils.plot_var_setup()


p_utils.apply_plot_style()


tracks_list = utils.get_tracks_list()

    



def _plot_lifecycle_mean_cr(n_bins=20): 

    fig, ax = plt.subplots()
    ax1 = ax.twinx()

    for mname, color, model_tracks in zip(MODEL_NAMES, COLORS, tracks_list): 
        model_pid = models_dict[mname]['path_id']
        
        dstracks = model_tracks

        PE_zarr = xr.open_zarr(f'/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/{model_pid}/mcs_condensation_rate_wam_stats.zarr')


        dstracks.track_duration.load()
        durations_hours = dstracks.track_duration.values  # (n_tracks,) in hours

        cr_values = PE_zarr.cr_mean.values 
        pr_values = PE_zarr.pr_mean.values # (tracks, times_3h)
        n_tracks, n_times = cr_values.shape  # (tracks, times_3h)


        
        

        binned_cr = np.full((n_tracks, n_bins), np.nan)
        binned_pr = np.full((n_tracks, n_bins), np.nan)

        lifecycle_positions = np.linspace(0, 1, n_bins)

        for tr in range(n_tracks):
            n_valid = (~np.isnan(cr_values[tr])).sum()
            
            if n_valid < 2:
                continue
            
            valid_cr = cr_values[tr, :n_valid]
            valid_pr = pr_values[tr, :n_valid]

            dur_hours = durations_hours[tr]
            frac_positions = (np.arange(n_valid) * 3) / dur_hours
            frac_positions = np.clip(frac_positions, 0, 1)

            f_cr = interp1d(frac_positions, valid_cr, bounds_error=False, fill_value=np.nan)
            f_pr = interp1d(frac_positions, valid_pr, bounds_error=False, fill_value=np.nan)

            binned_cr[tr] = f_cr(lifecycle_positions)
            binned_pr[tr] = f_pr(lifecycle_positions)

        lifecycle_mean_cr = np.nanmean(binned_cr, axis=0)
        lifecycle_mean_pr = np.nanmean(binned_pr, axis=0)
        lifecycle_pctg    = np.linspace(0, 100, n_bins)

        ax.plot(lifecycle_pctg, lifecycle_mean_cr, color=color, label=mname)
        ax1.plot(lifecycle_pctg, lifecycle_mean_pr, color=color, label=mname, linestyle='--')

    ax.plot([], [], color='grey', alpha=0.5, linestyle= '--', label='precip flux')
    ax.plot([], [], color='grey', alpha=0.5, linestyle= '-', label='condensation rate')
    ax.legend(bbox_to_anchor = (1.2, 1.3), ncols=3)
    ax.set_xlabel(r"$\%$ of MCS lifecyle")
    ax.set_ylabel(r"Condensation rate [kg m$^2$ s$^{-1}$]")
    ax1.set_ylabel(r"Precipitation flux [kg m$^2$ s$^{-1}$]")
    # ax.set_ylim(0.2, 1.)
    ax.spines['right'].set_visible(True)
    ax.grid(color='white')
    plt.savefig('figs/cr_pr_lifecycle.png', bbox_inches='tight')
    plt.close()


def _plot_cr_lifecycle_contribution(n_bins=20): 
    fig, ax = plt.subplots()
    # ax1 = ax.twinx()

    for mname, color, model_tracks in zip(MODEL_NAMES, COLORS, tracks_list): 
    
        model_pid = models_dict[mname]['path_id']
            
        dstracks = model_tracks

        PE_zarr = xr.open_zarr(f'/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/{model_pid}/mcs_condensation_rate_wam_stats.zarr')
        
        

        dstracks.track_duration.load()
        durations_hours = dstracks.track_duration.values  # (n_tracks,) in hours

        cr_values = PE_zarr.cr_mean.values 
        pr_values = PE_zarr.pr_mean.values # (tracks, times_3h)
        n_tracks, n_times = cr_values.shape  # (tracks, times_3h)


        

        bin_values = np.full((n_tracks, n_bins), np.nan)
        lifecycle_positions = np.linspace(0, 1, n_bins)

        for tr in range(n_tracks):
            n_valid = (~np.isnan(cr_values[tr])).sum()
            if n_valid < 2:
                continue
            
            valid_cr = cr_values[tr, :n_valid]
            total_cr = valid_cr.sum()
            pctg_contr = (valid_cr / total_cr) * 100
            
            dur_hours = durations_hours[tr]
            frac_positions = (np.arange(n_valid) * 3) / dur_hours
            frac_positions = np.clip(frac_positions, 0, 1)
            
            # interpolate onto fixed lifecycle grid
            f = interp1d(frac_positions, pctg_contr, bounds_error=False, fill_value=np.nan)
            bin_values[tr] = f(lifecycle_positions)

        lifecycle_mean_cr_contr = np.array([
                np.nanmean(bin_values[:, b]) if not np.all(np.isnan(bin_values[:, b])) else np.nan 
                for b in range(n_bins)
            ])
        lifecycle_pctg = np.linspace(0, 100, n_bins)

        ax.plot(lifecycle_pctg, lifecycle_mean_cr_contr, color=color, label=mname)

    ax.legend(bbox_to_anchor = (1.3, 1.2), ncols=4)
    ax.set_xlabel(r"$\%$ of MCS lifecyle")
    ax.set_ylabel(r"$\%$ Contribution to total condensation")
    ax.set_xticks(np.arange(0, 110, 10))
    # ax.set_ylim(0.2, 1.)
    ax.grid(color='white')
    plt.savefig('figs/lifecycle_cr_contribution.png', bbox_inches='tight')





########## 
## RUN FILES
##########

_plot_lifecycle_mean_cr(n_bins=10)
_plot_cr_lifecycle_contribution(n_bins=10)

###########




