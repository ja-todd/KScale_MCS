import xarray as xr 
import numpy as np
import src.hp_utils as utils 
import src.hp_models as models 
import src.plot_utils as p_utils
import src.microphysics as micro 
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import easygems.healpix as egh
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from collections import defaultdict
import matplotlib.patches as mpatches
import cartopy.feature as cf


p_utils.apply_plot_style()

models_dict = models.models_name_dict

BASE_PATH = '/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/'
MNAMES = list(models_dict.keys())
COLORS = [models_dict[mname]['color'] for mname in MNAMES]


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


def surface_duration_filtering(var_values, dstracks, track_ids, surface='all', duration='all'): 
    full_track_indices = track_ids - 1       # convert to dstracks indices
    valid_track_mask = full_track_indices < dstracks.sizes['tracks']
    full_track_indices = full_track_indices[valid_track_mask]

    dstracks.track_duration.load()
    dstracks.meanlon.load()
    dstracks.meanlat.load()

    durations_hours = dstracks.track_duration.isel(tracks=full_track_indices).values

    ## filter duration 
    if duration == 'all': 
        pass

    if duration == 'short': 
        dstracks = dstracks.isel(tracks=(durations_hours < 10))

    elif duration == 'long': 
        dstracks = dstracks.isel(tracks=(durations_hours >= 10))



    dstracks_surface = utils.filter_surface(dstracks, surface)
    surface_track_indices = dstracks_surface.tracks.values

    surface_mask = np.isin(full_track_indices, surface_track_indices )

    var_out = var_values[valid_track_mask][surface_mask] 
    durations_out = durations_hours[surface_mask]

    return var_out, durations_out


def binned_norm_lifecycle(n_tracks, var_values, durations_hours, n_bins=20): 
    # build normalised lifecycle arrays
        
    binned_var = np.full((n_tracks, n_bins), np.nan)

    for tr in range(n_tracks):
        n_valid = (~np.isnan(var_values[tr])).sum()  
        
        ## number of valid timesteps in times_3h
        if n_valid < 2:  ## doesn't compute if only 1 timestep
            continue
        
        valid_var = var_values[tr, :n_valid]  # (track, :3h duration) 

        dur_hours = durations_hours[tr]   ### for index tr, get the duration of the track in hours
        # frac_positions based on the actual 3-hourly steps, normalised by true duration
        frac_positions = (np.arange(n_valid) * 3) / dur_hours
        

        frac_positions = np.clip(frac_positions, 0, 1)  ## why is this necessary? shouldn't be 
        

        bin_edges = np.linspace(0, 1, n_bins + 1) ## to define n_bins (widths), you need n + 1 edges
        
                    
        bin_idx = np.digitize(frac_positions, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        for b in range(n_bins):
            vals = valid_var[bin_idx == b]
            if len(vals) > 0 and not np.all(np.isnan(vals)):
                binned_var[tr, b] = np.nanmean(vals)
            
    lifecycle_mean_var = np.nanmean(binned_var, axis=0)
    lifecycle_pctg = np.linspace(0, 100, n_bins)

    return lifecycle_mean_var, lifecycle_pctg


def plot_PE_normalized_lifecycle(surface='all', duration='all'): 

    fig, ax = plt.subplots()
    # ax1 = ax.twinx()

    for mname, color in zip(MNAMES, COLORS): 
        model_pid = models_dict[mname]['path_id']
        
        dstracks = TRACKS_DICT[mname]

        PE_zarr = xr.open_zarr(f'{BASE_PATH}{model_pid}/mcs_precip_efficiency_wam.zarr')
        entr_ds = xr.open_dataset(f'/gws/ssde/j25b/mcs_prime/jtodd/entrainment/data/{model_pid}/mcs_entrainment_wam.nc')

        track_ids = PE_zarr.tracks.values  # mask values
        
        pe_values = PE_zarr.precip_eff.values
        tbdiff_values = entr_ds.tb_diff_mean
        
        # print(tbdiff_values.shape)

        pe_values, durations_hours = surface_duration_filtering(pe_values, dstracks, track_ids, 
                                                       surface, duration)
        # tbdiff_values, _ = surface_duration_filtering(tbdiff_values, dstracks, track_ids, surface, duration)

                                        
        n_tracks, _ = pe_values.shape  # (tracks, times_3h)


        lifecycle_mean_pe, lifecycle_pctg            = binned_norm_lifecycle(n_tracks, pe_values, durations_hours)
        # lifecycle_mean_tbdiff, lifecycle_tbdiff_pctg = binned_norm_lifecycle(n_tracks, tbdiff_values, durations_hours)
    
        ax.plot(lifecycle_pctg, lifecycle_mean_pe, color=color, label=mname)
        # ax.plot(lifecycle_tbdiff_pctg, lifecycle_mean_tbdiff, color=color, linestyle='--')

    ax.legend(bbox_to_anchor = (1.2, 1.3), ncols=3)
    ax.set_xlabel(r"$\%$" + f" of {surface} MCS lifecycle")
    ax.set_ylabel(r"Precip Efficiency")
    # ax1.set_ylabel(r'T$_b$ diff [K]')
    ax.set_ylim(0.2, 1.)
    ax.grid(color='white')

    plt.savefig(f'figs/PE_normalized_lifecycle_{surface}.pdf', bbox_inches = 'tight', dpi=300)
    plt.savefig(f'figs/PE_normalized_lifecycle_{surface}.png', bbox_inches = 'tight')


def plot_cr_pr_lifecycle(surface='all'):     
    fig, ax = plt.subplots()
    ax1 = ax.twinx()

    for mname, color in zip(MNAMES, COLORS): 
        model_pid = models_dict[mname]['path_id']
        
        dstracks = TRACKS_DICT[mname]

        PE_zarr = xr.open_zarr(f'{BASE_PATH}{model_pid}/mcs_precip_efficiency_wam.zarr')


        track_ids = PE_zarr.tracks.values  # mask values
    
        cr_values = PE_zarr.condensation_rate.values 
        pr_values = PE_zarr.precip_flux.values
        
        cr_values, durations_hours = surface_duration_filtering(cr_values, dstracks, track_ids, 
                                                       surface)
        
        pr_values, durations_hours = surface_duration_filtering(pr_values, dstracks, track_ids, 
                                                       surface)
        
        
        n_tracks, _ = cr_values.shape  # (tracks, times_3h)


         

        lifecycle_mean_cr, lifecycle_pctg = binned_norm_lifecycle(n_tracks, cr_values, durations_hours)
        lifecycle_mean_pr, _ = binned_norm_lifecycle(n_tracks, pr_values, durations_hours)
    
        ax.plot(lifecycle_pctg, lifecycle_mean_cr, color=color, label=mname)
        ax1.plot(lifecycle_pctg, lifecycle_mean_pr, color=color, label=mname, linestyle='--')

    ax.plot([], [], color='grey', alpha=0.5, linestyle= '--', label='precip flux')
    ax.plot([], [], color='grey', alpha=0.5, linestyle= '-', label='condensation rate')
    ax.legend(bbox_to_anchor = (1.5, 1.3), ncols=4)
    ax.set_xlabel(r"$\%$" + f' of {surface} MCS lifecyle')
    ax.set_ylabel(r"Condensation rate [kg m$^{-2}$ s$^{-1}$]")
    ax1.set_ylabel(r"Precipitation flux [kg m$^{-2}$ s$^{-1}$]")
    # ax.set_ylim(0.2, 1.)
    ax.spines['right'].set_visible(True)
    ax.grid(color='white')

    plt.savefig('figs/cr_pr_MCS_lifecycle.pdf', bbox_inches = 'tight', dpi=300)
    plt.savefig(f'figs/cr_pr_MCS_lifecycle_{surface}.png', bbox_inches = 'tight')

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
                if len(vals) > 0 and not np.all(np.isnan(vals)):
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

BF_LAT_MIN, BF_LAT_MAX = 9, 15
BF_LON_MIN, BF_LON_MAX = -5, 5


def plot_region_map(ax, region_cfg, model_display, region_display):
    """Map showing the analysis region and Burkina Faso sub-box."""
    # Convert WAM lon to [-180, 180] for the map extent
    wam_lon_min = region_cfg['lon_min'] - 360 if region_cfg['lon_min'] > 180 else region_cfg['lon_min']
    wam_lon_max = region_cfg['lon_max']
    wam_lat_min = region_cfg['lat_min']
    wam_lat_max = region_cfg['lat_max']


    proj = ccrs.PlateCarree()

    ax.set_extent([wam_lon_min - 10, wam_lon_max + 10, wam_lat_min - 10, wam_lat_max + 10],
                  crs=proj)
    ax.coastlines(linewidth=0.8)
    ax.add_feature(cf.BORDERS, linewidth=0.5)
    ax.add_feature(cf.LAND, facecolor='lightgrey', alpha=0.4)

    crs = ccrs.PlateCarree()
    ax.add_patch(mpatches.Rectangle(
        (wam_lon_min, wam_lat_min),
        wam_lon_max - wam_lon_min, wam_lat_max - wam_lat_min,
        linewidth=1.5, edgecolor='steelblue', facecolor='steelblue',
        alpha=0.15, transform=proj, label=f'{region_display} region',
    ))
    ax.add_patch(mpatches.Rectangle(
        (BF_LON_MIN, BF_LAT_MIN),
        BF_LON_MAX - BF_LON_MIN, BF_LAT_MAX - BF_LAT_MIN,
        linewidth=2, edgecolor='tab:orange', facecolor='none',
        transform=proj, label='Burkina Faso box',
    ))
    ax.legend(loc='lower left')
    ax.set_title(f'Analysis regions')
    # plt.savefig('analysis_region.png', bbox_inches='tight')

def plot_map(region_cfg, model_display, region_display):
    fig_map, ax_map = plt.subplots(subplot_kw={'projection': ccrs.Robinson()}, figsize=(6, 5))
    plot_region_map(ax_map, region_cfg, model_display, region_display)
    fig_map.tight_layout()
    path = 'analysis_region.png'
    fig_map.savefig(path, bbox_inches = 'tight', dpi=300)
    print(f'Saved {path}')
    plt.close(fig_map)


# print("plot 1")

# plot_mcs_stats_PE()


# print("plot 2-4")


# plot_PE_normalized_lifecycle()
# plot_PE_normalized_lifecycle(surface='land')
# plot_PE_normalized_lifecycle(surface='ocean')

# print("plot 5-7")


# plot_cr_pr_lifecycle()
# plot_cr_pr_lifecycle(surface='land')
# plot_cr_pr_lifecycle(surface='ocean')

# print("plot 6")

# plot_contribution_to_total_cr()
region = 'wam'
model = 'um_glm_n2560_RAL3p3_tuned_hk26'
region_cfg     = models.REGIONS[region]
region_display = region_cfg['display']
model_display = models.MODELS[model]['display']



plot_map(region_cfg, model_display, region_display)





