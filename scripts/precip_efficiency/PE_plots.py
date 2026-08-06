from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 
import numpy as np
import src.hp_models as models 
import src.hp_utils as utils 
import src.plot_utils as p_utils 
import warnings
import xarray as xr



with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)


BASE_PATH = '/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/'

models_dict = models.models_name_dict
MODEL_NAMES, REGION_CFG, COLORS = p_utils.plot_var_setup()


p_utils.apply_plot_style()

cache_path = Path('../../data/tracks_list_cache.pkl')
tracks_list = utils.get_tracks_list(MODEL_NAMES, REGION_CFG, cache_path)

    



def _plot_lifecycle_mean_cr(n_bins=20): 

    fig, ax = plt.subplots()
    ax1 = ax.twinx()

    for mname, color, model_tracks in zip(MODEL_NAMES, COLORS, tracks_list): 
        model_pid = models_dict[mname]['path_id']
        
        dstracks = model_tracks

        PE_zarr = xr.open_zarr(f'{BASE_PATH}{model_pid}/mcs_condensation_rate_wam_stats.zarr')


        dstracks.track_duration.load()
        durations_hours = dstracks.track_duration.values  # (n_tracks,) in hours

        cr_values = PE_zarr.cr_sum.values 
        pr_values = PE_zarr.pr_sum.values # (tracks, times_3h)
        n_tracks, n_times = cr_values.shape  # (tracks, times_3h)


        
        

        binned_cr = np.full((n_tracks, n_bins), np.nan)
        binned_pr = np.full((n_tracks, n_bins), np.nan)

        lifecycle_positions = np.linspace(0, 1, n_bins)

        for tr in range(n_tracks):
            valid_mask = ~np.isnan(cr_values[tr])
            valid_cr = cr_values[tr, valid_mask]
            valid_pr = pr_values[tr, valid_mask]
            n_valid = valid_mask.sum()

            if n_valid < 2:
                continue

            dur_hours = durations_hours[tr]
            valid_indices = np.where(valid_mask)[0]
            frac_positions = (valid_indices * 3) / dur_hours
            frac_positions = np.clip(frac_positions, 0, 1)

            f_cr = interp1d(frac_positions, valid_cr, bounds_error=False, fill_value=(np.nan, valid_cr[-1]))
            f_pr = interp1d(frac_positions, valid_pr, bounds_error=False, fill_value=(np.nan, valid_pr[-1]))

            binned_cr[tr] = f_cr(lifecycle_positions)
            binned_pr[tr] = f_pr(lifecycle_positions)

        lifecycle_mean_cr = np.nanmean(binned_cr, axis=0)
        lifecycle_mean_pr = np.nanmean(binned_pr, axis=0)
        lifecycle_pctg    = np.linspace(0, 100, n_bins)

        ax.plot(lifecycle_pctg, lifecycle_mean_cr, color=color, label=mname)
        ax1.plot(lifecycle_pctg, lifecycle_mean_pr, color=color, linestyle='--')

    ax.plot([], [], color='grey', alpha=0.5, linestyle= '--', label='precipitation')
    ax.plot([], [], color='grey', alpha=0.5, linestyle= '-', label='condensation')
    # ax.legend(bbox_to_anchor = (1.2, 1.3), ncols=3)
    ax.set_xlabel(r"$\%$ MCS lifecyle")
    ax.set_ylabel(r"Condensation rate [kg m$^{-2}$ s$^{-1}$]")
    ax1.set_ylabel(r"Precipitation rate [kg m$^{-2}$ s$^{-1}$]")
    # ax.set_ylim(0.2, 1.)
    ax.spines['right'].set_visible(True)
    ax.grid(color='white')
    ax1.grid(False)
    p_utils.centre_legend_above(fig, ax, ncols=3)
    plt.savefig('figs/lifecycles/crSUM_prSUM_lifecycle')
    plt.close()


def _plot_cr_pr_lifecycle_contribution(n_bins=20): 
    fig, ax = plt.subplots()
    ax1 = ax.twinx()
    # ax1 = ax.twinx()

    for mname, color, model_tracks in zip(MODEL_NAMES, COLORS, tracks_list): 
    
        model_pid = models_dict[mname]['path_id']
            
        dstracks = model_tracks

        PE_zarr = xr.open_zarr(f'{BASE_PATH}{model_pid}/mcs_condensation_rate_wam_stats.zarr')
        
        

        dstracks.track_duration.load()
        durations_hours = dstracks.track_duration.values  # (n_tracks,) in hours

        cr_values = PE_zarr.cr_sum.values 
        pr_values = PE_zarr.pr_sum.values # (tracks, times_3h)
        n_tracks, n_times = cr_values.shape  # (tracks, times_3h)


        

        cr_bin_values = np.full((n_tracks, n_bins), np.nan)
        pr_bin_values = np.full((n_tracks, n_bins), np.nan)
        lifecycle_positions = np.linspace(0, 1, n_bins)

        for tr in range(n_tracks):
            valid_mask = ~np.isnan(cr_values[tr])
            valid_cr = cr_values[tr, valid_mask]
            valid_pr = pr_values[tr, valid_mask]
            n_valid = valid_mask.sum()

            if n_valid < 2:
                continue

            total_cr = valid_cr.sum()
            total_pr = valid_pr.sum()
            pctg_cr_contr = (valid_cr / total_cr) * 100
            pctg_pr_contr = (valid_pr / total_pr) * 100

            dur_hours = durations_hours[tr]
            valid_indices = np.where(valid_mask)[0]
            frac_positions = (valid_indices * 3) / dur_hours
            frac_positions = np.clip(frac_positions, 0, 1)

            f = interp1d(frac_positions, pctg_cr_contr, bounds_error=False, fill_value=(np.nan, pctg_cr_contr[-1]))
            f_pr = interp1d(frac_positions, pctg_pr_contr, bounds_error=False, fill_value=(np.nan, pctg_pr_contr[-1]))
            cr_bin_values[tr] = f(lifecycle_positions)
            pr_bin_values[tr] = f_pr(lifecycle_positions)
        
        lifecycle_mean_cr_contr = np.array([
                np.nanmean(cr_bin_values[:, b]) if not np.all(np.isnan(cr_bin_values[:, b])) else np.nan 
                for b in range(n_bins)
            ])
        lifecycle_mean_pr_contr = np.array([
                        np.nanmean(pr_bin_values[:, b]) if not np.all(np.isnan(pr_bin_values[:, b])) else np.nan 
                        for b in range(n_bins)
                    ])
        lifecycle_mean_cr_contr = lifecycle_mean_cr_contr / lifecycle_mean_cr_contr.sum() * 100  ## accounts for the interpolation error
        lifecycle_mean_pr_contr = lifecycle_mean_pr_contr / lifecycle_mean_pr_contr.sum() * 100  ## accounts for the interpolation error
        lifecycle_pctg = np.linspace(0, 100, n_bins)

        ax.plot(lifecycle_pctg, lifecycle_mean_cr_contr, color=color, label=mname)
        ax1.plot(lifecycle_pctg, lifecycle_mean_pr_contr, color=color, linestyle='--')

    # ax.legend(bbox_to_anchor = (1.3, 1.2), ncols=4)
    ax.plot([], [], color='grey', alpha=0.5, linestyle= '--', label='precipitation')
    ax.plot([], [], color='grey', alpha=0.5, linestyle= '-', label='condensation')
    ax.set_xlabel(r"$\%$ MCS lifecyle")
    ax.set_ylabel(r"$\%$ Contribution to total condensation")
    ax1.set_ylabel(r"$\%$ Contribution to total precipitation")
    ax.set_xticks(np.arange(0, 110, 10))
    for _ax in [ax, ax1]: 
        _ax.set_yticks(np.arange(2, 7, 1))
        _ax.set_ylim(1.5, 6)
    ax.spines['right'].set_visible(True)
    # ax.set_ylim(0.2, 1.)
    ax.grid(color='white')
    p_utils.centre_legend_above(fig, ax, y_offset=1.06, ncols=3)
    plt.savefig('figs/lifecycles/crprSUM_contribution_lifecycle')





########## 
## RUN FILES
##########

_plot_lifecycle_mean_cr(n_bins=24)
_plot_cr_pr_lifecycle_contribution(n_bins=24)

###########




