import xarray as xr 
import numpy as np
import src.hp_utils as utils 
import src.hp_models as models 
import src.microphysics as micro 
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import easygems.healpix as egh
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from collections import defaultdict


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 15,
    'axes.labelsize': 16,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.top': False,       # no top ticks (tmXTOn = False)
    'ytick.right': False,
    'ytick.left': True,     # no right ticks (tmYROn = False)
    'axes.linewidth': 1.5,
    'lines.linewidth': 2,
    'axes.spines.top': False, 
    'axes.spines.right': False,
    'axes.facecolor':  "#EAEAF2E6",
     
    
    'axes.grid': True, 
    'grid.color': '#DEDFE4',
    'grid.alpha': 0.5,
    'figure.labelsize': '15', 
    'font.weight': 'normal', 
    'legend.handlelength': 2, 
    'legend.handletextpad': 0.5, 
    'legend.frameon': False, 
})


models_dict = {
    'RAL3_z10_40k' : {'path_id':'z10/um_glm_n2560_RAL3p3_tuned_sahel_z10_t40k', 'color': '#AC2078' },
    'RAL3_z9' : {'path_id': 'z9/um_glm_n2560_RAL3p3_tuned_hk26', 'color': '#0D0C6E'},
    'COMORPH_n2560_z9'     : {'path_id': 'z9/um_glm_n2560_CoMA9_hk26', 'color': '#BC6263'}, 
    'COMORPH_n1280' : {'path_id': 'z9/um_glm_n1280_CoMA9_hk26', 'color':  '#B79394'},
    'GAL9' : {'path_id': 'z9/um_glm_n1280_GAL9_v2_hk26', 'color':'#BBCBDF'},
     
    'RAL3_z10_4k' : {'path_id':'z10/um_glm_n2560_RAL3p3_tuned_sahel_z10_t4k', 'color': '#6893DA'}
}



BASE_PATH = '/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/'
MNAMES = list(models_dict.keys())
COLORS = [models_dict[mname]['color'] for mname in MNAMES]




def plot_mcs_stats_PE():

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    ax1, ax2 = axs.flatten()

    for mname, color in zip(MNAMES, COLORS): 
        model_pid = models_dict[mname]['path_id']
        PE_zarr = xr.open_zarr(f'{BASE_PATH}{model_pid}/mcs_precip_efficiency_wam.zarr')
        pe_valid = (~np.isnan(PE_zarr.precip_eff.values)).sum(axis=0) 
        # print(pe_valid[hours > 60])
        n_tracks = PE_zarr.sizes['tracks']
        pe_valid_percentages = (pe_valid / n_tracks) * 100
        # (times_3h,) - count per time slot
        hours = np.arange(len(pe_valid)) * 3
        ax1.plot(hours, pe_valid, color=color, label=mname)
        ax2.plot(hours, pe_valid_percentages, color=color)


    fig.legend(bbox_to_anchor = (0.85
                                , 1.2), ncols=3)
    for _ax in axs.flatten():
        _ax.set_xlabel('Hours since storm initiation')
        _ax.grid(color='white')

    for _ax in [ax1, ax2]: 
        _ax.set_xlim(0, 120)
        _ax.set_yscale('log')

    ax1.set_ylabel('Number of MCSs')
    ax2.set_ylabel('Percentage of MCSs')

    plt.subplots_adjust(wspace=0.3)

    fig, ax = plt.subplots()

    for mname, color in zip(MNAMES, COLORS): 
        model_pid = models_dict[mname]['path_id']
        PE_zarr = xr.open_zarr(f'{BASE_PATH}{model_pid}/mcs_precip_efficiency_wam.zarr')
        storm_scale_PE = PE_zarr.precip_eff.groupby(PE_zarr.times_3h).mean(dim='tracks', skipna=True)
        ax.plot(PE_zarr.times_3h.values * 3, storm_scale_PE.values,  color=color, label=mname)

    ax.set_xlim(0, 60)
    ax.set_ylim(0.2, 1)
    ax.set_ylabel('Precip Efficiency')
    ax.set_xlabel('Hours since storm initiation')
    fig.legend(bbox_to_anchor = (1.1
                                , 1.1), ncols=3)
    ax.grid(color='white')

    plt.savefig('figs/mcs_PE_since_initiation.pdf', bbox_inches = 'tight', dpi=300)


"""
MAKE HASH-MAP OF DSTRACKS FOR EACH OF THE MODELS TO MAKE THE PLOTTING QUICKER
"""
print("making TRACKS_DICT")
TRACKS_DICT = defaultdict(lambda: defaultdict(dict))

for mname in MNAMES: 
    model_pid = models_dict[mname]['path_id']
    model_id  = model_pid.split('/')[1]


    stats_url = models.stats_url(model_id)
    dstracks  = utils.load_track_stats(stats_url)

    TRACKS_DICT[mname] = dstracks




def plot_PE_normalized_lifecycle(): 

    fig, ax = plt.subplots()

    for mname, color in zip(MNAMES, COLORS): 
        model_pid = models_dict[mname]['path_id']
        
        dstracks = TRACKS_DICT[mname]

        PE_zarr = xr.open_zarr(f'{BASE_PATH}{model_pid}/mcs_precip_efficiency_wam.zarr')


        track_ids = PE_zarr.tracks.values  # mask values
        track_indices = track_ids - 1       # convert to dstracks indices

        dstracks.track_duration.load()
        durations_hours = dstracks.track_duration.isel(tracks=track_indices).values  # (n_tracks,) in hours

        cr_values = PE_zarr.precip_eff.values  # (tracks, times_3h)
        n_tracks, n_times = cr_values.shape  # (tracks, times_3h)


        # build normalised lifecycle arrays
        n_bins = 20  # resolution of normalised lifecycle
        binned_cr_contrs = np.full((n_tracks, n_bins), np.nan)
        
        for tr in range(n_tracks):
            n_valid = (~np.isnan(cr_values[tr])).sum()  

            ## number of valid timesteps in times_3h
            if n_valid < 2:  ## doesn't compute if only 1 timestep
                continue
            valid_cr = cr_values[tr, :n_valid]  # (track, :3h duration) 
            
            dur_hours = durations_hours[tr]   ### for index tr, get the duration of the track in hours
            # frac_positions based on the actual 3-hourly steps, normalised by true duration
            frac_positions = (np.arange(n_valid) * 3) / dur_hours


            frac_positions = np.clip(frac_positions, 0, 1)  ## why is this necessary? shouldn't be 
            
            bin_edges = np.linspace(0, 1, n_bins + 1)  ## to define n_bins (widths), you need n + 1 edges
            bin_idx = np.digitize(frac_positions, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            for b in range(n_bins):
                vals = valid_cr[bin_idx == b]
                if len(vals) > 0:
                    binned_cr_contrs[tr, b] = np.nanmean(vals)

        lifecycle_mean_cr_contr = np.nanmean(binned_cr_contrs, axis=0)
        lifecycle_pctg = np.linspace(0, 100, n_bins)

        ax.plot(lifecycle_pctg, lifecycle_mean_cr_contr, color=color, label=mname)

    ax.legend(bbox_to_anchor = (1.2, 1.3), ncols=3)
    ax.set_xlabel(r"$\%$ of MCS lifecyle")
    ax.set_ylabel(r"Precip Efficiency")
    ax.set_ylim(0.2, 1.)
    ax.grid(color='white')

    plt.savefig('figs/PE_normalized_lifecycle.pdf', bbox_inches = 'tight', dpi=300)



def plot_cr_pr_lifecycle():     
    fig, ax = plt.subplots()
    ax1 = ax.twinx()

    for mname, color in zip(MNAMES, COLORS): 
        model_pid = models_dict[mname]['path_id']
        
        dstracks = TRACKS_DICT[mname]

        PE_zarr = xr.open_zarr(f'{BASE_PATH}{model_pid}/mcs_precip_efficiency_wam.zarr')


        track_ids = PE_zarr.tracks.values  # mask values
        track_indices = track_ids - 1       # convert to dstracks indices

        dstracks.track_duration.load()
        durations_hours = dstracks.track_duration.isel(tracks=track_indices).values  # (n_tracks,) in hours

        cr_values = PE_zarr.condensation_rate.values 
        pr_values = PE_zarr.precip_flux.values # (tracks, times_3h)
        n_tracks, n_times = cr_values.shape  # (tracks, times_3h)


        # build normalised lifecycle arrays
        n_bins = 20  # resolution of normalised lifecycle
        binned_cr_contrs = np.full((n_tracks, n_bins), np.nan)
        binned_pr = np.full((n_tracks, n_bins), np.nan)
        
        for tr in range(n_tracks):
            n_valid = (~np.isnan(cr_values[tr])).sum()  
            
            ## number of valid timesteps in times_3h
            if n_valid < 2:  ## doesn't compute if only 1 timestep
                continue
            
            valid_cr = cr_values[tr, :n_valid]  # (track, :3h duration) 
            valid_pr = pr_values[tr, :n_valid]

            dur_hours = durations_hours[tr]   ### for index tr, get the duration of the track in hours
            # frac_positions based on the actual 3-hourly steps, normalised by true duration
            frac_positions = (np.arange(n_valid) * 3) / dur_hours
            

            frac_positions = np.clip(frac_positions, 0, 1)  ## why is this necessary? shouldn't be 
            

            bin_edges = np.linspace(0, 1, n_bins + 1) ## to define n_bins (widths), you need n + 1 edges
            
                        
            bin_idx = np.digitize(frac_positions, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            for b in range(n_bins):
                vals = valid_cr[bin_idx == b]
                vals_pr = valid_pr[bin_idx == b]
                if len(vals) > 0:
                    binned_cr_contrs[tr, b] = np.nanmean(vals)
                if len(vals_pr) > 0: 
                    binned_pr[tr, b] = np.nanmean(vals_pr)   

        lifecycle_mean_cr_contr = np.nanmean(binned_cr_contrs, axis=0)
        lifecycle_mean_pr = np.nanmean(binned_pr, axis=0)
        lifecycle_pctg = np.linspace(0, 100, n_bins)

        ax.plot(lifecycle_pctg, lifecycle_mean_cr_contr, color=color, label=mname)
        ax1.plot(lifecycle_pctg, lifecycle_mean_pr, color=color, label=mname, linestyle='--')

    ax.plot([], [], color='grey', alpha=0.5, linestyle= '--', label='precip flux')
    ax.plot([], [], color='grey', alpha=0.5, linestyle= '-', label='condensation rate')
    ax.legend(bbox_to_anchor = (1.5, 1.3), ncols=4)
    ax.set_xlabel(r"$\%$ of MCS lifecyle")
    ax.set_ylabel(r"Condensation rate [kg m$^2$ s$^{-1}$]")
    ax1.set_ylabel(r"Precipitation flux [kg m$^2$ s$^{-1}$]")
    # ax.set_ylim(0.2, 1.)
    ax.spines['right'].set_visible(True)
    ax.grid(color='white')

    plt.savefig('figs/cr_pr_MCS_lifecycle.pdf', bbox_inches = 'tight', dpi=300)


def plot_contribution_to_total_cr(): 

    fig, ax = plt.subplots()
    # ax1 = ax.twinx()

    for mname, color in zip(MNAMES, COLORS): 

        model_pid = models_dict[mname]['path_id']
            
        dstracks = TRACKS_DICT[mname]

        PE_zarr = xr.open_zarr(f'{BASE_PATH}{model_pid}/mcs_precip_efficiency_wam.zarr')


        track_ids = PE_zarr.tracks.values  # mask values
        track_indices = track_ids - 1       # convert to dstracks indices

        dstracks.track_duration.load()
        durations_hours = dstracks.track_duration.isel(tracks=track_indices).values  # (n_tracks,) in hours

        cr_values = PE_zarr.condensation_rate.values 
        n_tracks, n_times = cr_values.shape  # (tracks, times_3h)


        # build normalised lifecycle arrays
        n_bins = 20  # resolution of normalised lifecycle
        binned_cr_contrs = np.full((n_tracks, n_bins), np.nan)
        # binned_pr = np.full((n_tracks, n_bins), np.nan)

        for tr in range(n_tracks):
            n_valid = (~np.isnan(cr_values[tr])).sum() 
            if n_valid < 2:  ## doesn't compute if only 1 timestep
                    continue
                
            valid_cr = cr_values[tr, :n_valid] 
            
            total_cr = valid_cr.sum()

            pctg_contr = (valid_cr / total_cr) * 100 
            # print(pctg_contr)

            dur_hours = durations_hours[tr]   ### for index tr, get the duration of the track in hours
                # frac_positions based on the actual 3-hourly steps, normalised by true duration
            frac_positions = (np.arange(n_valid) * 3) / dur_hours


            frac_positions = np.clip(frac_positions, 0, 1) ## edge case
            
            bin_edges = np.linspace(0, 1, n_bins + 1)  ## to define n_bins (widths), you need n + 1 edges
            bin_idx = np.digitize(frac_positions, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            for b in range(n_bins):
                vals = pctg_contr[bin_idx == b]
                if len(vals) > 0:
                    binned_cr_contrs[tr, b] = np.nanmean(vals)

        lifecycle_mean_cr_contr = np.nanmean(binned_cr_contrs, axis=0)
        lifecycle_pctg = np.linspace(0, 100, n_bins)

        ax.plot(lifecycle_pctg, lifecycle_mean_cr_contr, color=color, label=mname)

    ax.legend(bbox_to_anchor = (1.2, 1.3), ncols=3)
    ax.set_xlabel(r"$\%$ of MCS lifecyle")
    ax.set_ylabel(r"$\%$ Contribution to total condensation")
    # ax.set_ylim(0.2, 1.)
    ax.grid(color='white')

    plt.savefig('figs/cr_contribution_lifecycle.pdf', bbox_inches = 'tight', dpi=300)



print("plot 1")

plot_mcs_stats_PE()


print("plot 2")


plot_PE_normalized_lifecycle()

print("plot 3")


plot_cr_pr_lifecycle()


print("plot 4")

plot_contribution_to_total_cr()






