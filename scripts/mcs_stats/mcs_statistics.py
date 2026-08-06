from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cf
import cartopy.crs as ccrs
import easygems.healpix as egh
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from pathlib import Path
import pickle
from matplotlib.colors import FuncNorm
from scipy.ndimage import zoom
from scipy.stats import gaussian_kde

import src.hp_models as models 
import src.hp_utils as utils
import src.plot_utils as p_utils
import time
import xarray as xr




p_utils.apply_plot_style()

MODEL_NAMES, REGION_CFG, COLORS = p_utils.plot_var_setup(region='wam')

models_dict = models.models_name_dict
cache_path = Path('../../data/tracks_list_cache.pkl')
tracks_list = utils.get_tracks_list(MODEL_NAMES, REGION_CFG, cache_path)



def mcs_spatial_dist(track_density=False, jja = False, land_only=False): 
    """
    Plots the percentage proportion of MCSs across the West African 
    Sahel region by track mean lat over lifetime OR track density 
    (by percentage proportion)
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 8),
                            subplot_kw={'projection': ccrs.PlateCarree()}, sharey=True, sharex=True)

    p_utils.label_subplots(axs)


    for model_tracks, ax, mname in zip(tracks_list, axs.flatten(), MODEL_NAMES):

        

        if land_only: 
            tag = '_land'
            dstracks_wam = utils.filter_surface(model_tracks, 'land')


        start_time = pd.DatetimeIndex(dstracks_wam.start_basetime.values)


        

        jja_mask = (start_time.month >= 6) & (start_time.month <= 8)

        if jja: 
            model_tracks = dstracks_wam.sel(tracks=jja_mask)

        lats = model_tracks.meanlat.values  # (n_tracks, n_times)
        lons = model_tracks.meanlon.values
        lons = (lons + 180) % 360 - 180

        lon_bins = np.arange(-25, 27.5, 2.5)
        lat_bins = np.arange(-3, 21, 2.5)

        if not track_density: 
            tag += '_mean_track_position'
            # one point per track — mean position over lifetime
            track_mean_lat = np.nanmean(lats, axis=1)  # (n_tracks,)
            track_mean_lon = np.nanmean(lons, axis=1)
    
            valid = ~np.isnan(track_mean_lat) & ~np.isnan(track_mean_lon)
            track_mean_lat = track_mean_lat[valid]
            track_mean_lon = track_mean_lon[valid]
    
            
            density, _, _ = np.histogram2d(track_mean_lat, track_mean_lon,
                                            bins=[lat_bins, lon_bins])

            density_pct = (density / model_tracks.sizes['tracks']) * 100

            
        if track_density: 
            tag += '_track_density'
            lats_flat = lats.ravel()
            lons_flat = lons.ravel()
    
            valid = ~np.isnan(lats_flat) & ~np.isnan(lons_flat)
            lats_flat = lats_flat[valid]
            lons_flat = lons_flat[valid]

            density, _, _ = np.histogram2d(lats_flat, lons_flat,
                                                    bins=[lat_bins, lon_bins])

            density_pct = (density / density.sum()) * 100

        

        lon_centres = (lon_bins[:-1] + lon_bins[1:]) / 2
        lat_centres = (lat_bins[:-1] + lat_bins[1:]) / 2

        ax.coastlines()
        ax.set_extent([-20, 40, -3, 20])
        ax.set_xticks(np.arange(-20, 41, 10), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(0, 21, 5), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        im = ax.pcolormesh(lon_centres, lat_centres, density_pct,
                            vmin=0, vmax=3, cmap='bone_r')
        ax.set_title(f'{mname}, n = {model_tracks.sizes["tracks"]}')

    axs = axs.flatten()

    axs[1].yaxis.set_tick_params(labelleft=False)   # top right - no left labels
    axs[0].xaxis.set_tick_params(labelbottom=False)
    axs[1].xaxis.set_tick_params(labelbottom=False) # bottom left - no bottom labels
    axs[3].yaxis.set_tick_params(labelleft=False)   # bottom right - no left labels

    cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])  # placeholder, will be resized
    cbar = fig.colorbar(im, cax=cbar_ax, label='Proportion of MCSs [%]')
    plt.subplots_adjust(hspace=-0.1, wspace=0.15)
    p_utils.match_colorbar_to_axes(fig, cbar, axs)
    print(tag)
    plt.savefig(f'figs/MCS_spatial_dist{tag}')
    plt.close()


def plot_mcs_durations(jja=False, land_only=False):
    base_path = '/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/'
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    p_utils.label_subplots(axs)
    ax1, ax2, ax3 = axs.flatten()
    

    for tracks, mname, color in zip(tracks_list, MODEL_NAMES, COLORS): 
        tag = ''
        start_times = tracks.start_basetime.values

        start_times_pd = pd.DatetimeIndex(start_times)


        if jja: 
            tag = '_jja'
            jja_mask = (start_times_pd.month >= 6) & (start_times_pd.month <= 8)
            
            tracks = tracks.sel(tracks=jja_mask)

        if land_only: 
            tag += '_land'
            tracks = utils.filter_surface(tracks, 'land')

        start_times = tracks.start_basetime.values


        ## ax1 
        model_pid = models_dict[mname]['path_id']
        PE_zarr = xr.open_zarr(f'{base_path}{model_pid}/mcs_condensation_rate_wam_stats.zarr')
        PE_zarr = PE_zarr.sel(track=tracks.tracks)
        num_valid_MCSs = (~np.isnan(PE_zarr.pe_mean.values)).sum(axis=0)
        hours = np.arange(len(num_valid_MCSs)) * 3


        ## ax2
        durations = tracks.track_duration.values 
        durations = durations[~np.isnan(durations)]
        kde = gaussian_kde(durations)
        x = np.linspace(0, durations.max(), 200)


        

        
        init_hours  = pd.DatetimeIndex(start_times).hour.astype(float)
        valid       = ~np.isnan(init_hours)
        init_hours  = init_hours[valid]

        bins = np.arange(0, 25, 1)
        counts, edges = np.histogram(init_hours, bins=bins)
        pct = (counts / len(init_hours)) * 100
        centres = 0.5 * (edges[:-1] + edges[1:])


        zoom_factor = 4
        pct_upsampled = zoom(np.nan_to_num(pct), zoom=zoom_factor, order=3)

        print(pct_upsampled)

        centres_fine = np.linspace(centres.min(), centres.max(), pct_upsampled.shape[0])
        
        ax1.plot(hours, num_valid_MCSs, color=color, label=mname, linewidth=2.5) 
        ax2.plot(x, kde(x), color=color, linewidth=2.5)
        # ax3.plot(centres, pct, color=color, linewidth=2.5)
        ax3.plot(centres_fine, pct_upsampled, color=color, linewidth=2.5)


    # fig.legend(bbox_to_anchor = (0.75
    #                             , 1.05), ncols=4)
    for _ax in axs.flatten():
        _ax.set_xlabel('MCS track duration [hrs]')
        _ax.grid(color='white')

    
    ax1.set_xlim(0, 120)
        # _ax.set_yscale('log')
    ax2.set_xlim(0, 60)
    ax1.set_yscale('log')

    ax1.set_ylabel('Number of MCSs')
    ax2.set_ylabel('Probability Density')
    ax3.set_xlabel('Initiation hour [UTC]')
    ax3.set_ylabel(r'$\%$ MCS initiations')
    ax3.set_xlim(0, 23)
    ax3.set_xticks(np.arange(0, 24, 3))

    plt.subplots_adjust(wspace=0.3)
    p_utils.centre_legend_above(fig, axs, ncols=4, y_offset=1.1)
    plt.savefig(f'figs/mcs_counts_durations{tag}')




def init_time_map():
    fig, axs = plt.subplots(2, 2, figsize=(15, 8),
                         subplot_kw={'projection': ccrs.PlateCarree()}, sharey=True, sharex=True)
    for model_tracks, ax, name in zip(tracks_list, axs.flatten(), MODEL_NAMES):

        lats = model_tracks.meanlat.values
        lons = model_tracks.meanlon.values
        lons = (lons + 180) % 360 - 180

        track_mean_lat = np.nanmean(lats, axis=1)
        track_mean_lon = np.nanmean(lons, axis=1)

        start_times = model_tracks.start_basetime.values
        init_hours  = pd.DatetimeIndex(start_times).hour.astype(float)

        valid = ~np.isnan(track_mean_lat) & ~np.isnan(track_mean_lon)
        track_mean_lat = track_mean_lat[valid]
        track_mean_lon = track_mean_lon[valid]
        init_hours     = init_hours[valid]

        lon_bins = np.arange(-25, 27.5, 2.5)
        lat_bins = np.arange(0, 21, 2.5)

        hour_sum, _, _ = np.histogram2d(track_mean_lat, track_mean_lon,
                                        bins=[lat_bins, lon_bins],
                                        weights=init_hours)
        count, _, _ = np.histogram2d(track_mean_lat, track_mean_lon,
                                    bins=[lat_bins, lon_bins])

        mean_hour = np.where(count > 0, hour_sum / count, np.nan)

        lon_centres = (lon_bins[:-1] + lon_bins[1:]) / 2
        lat_centres = (lat_bins[:-1] + lat_bins[1:]) / 2

        ax.coastlines()
        ax.set_extent([-20, 40, -3, 20])
        ax.set_xticks(np.arange(-20, 41, 10), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(0, 21, 5), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        im = ax.pcolormesh(lon_centres, lat_centres, mean_hour,
                            vmin=0, vmax=23, cmap='twilight_shifted')
        ax.set_title(f'{name}, n = {model_tracks.sizes["tracks"]}')

    axs = axs.flatten()
    axs[1].yaxis.set_tick_params(labelleft=False)
    axs[1].xaxis.set_tick_params(labelbottom=False)
    axs[3].yaxis.set_tick_params(labelleft=False)

    cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Mean initiation hour (UTC)')
    plt.subplots_adjust(hspace=-0.1, wspace=0.15)
    p_utils.match_colorbar_to_axes(fig, cbar, axs)
    plt.savefig('MCS_init_hour')
    plt.close()


def mcs_durations_map(resolution=2.5):
    fig, axs = plt.subplots(2, 2, figsize=(15, 8),
                         subplot_kw={'projection': ccrs.PlateCarree()}, sharey=True, sharex=True)
    p_utils.label_subplots(axs)
    for model_tracks, ax, name in zip(tracks_list, axs.flatten(), MODEL_NAMES):

        lats = model_tracks.meanlat.values
        lons = model_tracks.meanlon.values
        lons = (lons + 180) % 360 - 180

        track_mean_lat = np.nanmean(lats, axis=1)
        track_mean_lon = np.nanmean(lons, axis=1)

        
        durations   = model_tracks.track_duration.values
    
        valid = ~np.isnan(track_mean_lat) & ~np.isnan(track_mean_lon)
        track_mean_lat = track_mean_lat[valid]
        track_mean_lon = track_mean_lon[valid]
        durations      = durations[valid]

        lon_bins = np.arange(-25, 27.5, resolution)
        lat_bins = np.arange(-3, 21, resolution)

        duration_sum, _, _ = np.histogram2d(track_mean_lat, track_mean_lon,
                                        bins=[lat_bins, lon_bins],
                                        weights=durations)
        count, _, _ = np.histogram2d(track_mean_lat, track_mean_lon,
                                    bins=[lat_bins, lon_bins])

        mean_duration = np.where(count > 0, duration_sum / count, np.nan)

        lon_centres = (lon_bins[:-1] + lon_bins[1:]) / 2
        lat_centres = (lat_bins[:-1] + lat_bins[1:]) / 2

        xp = np.array([4, 10, 18, 24])
        fp = np.array([0.0, 0.2, 0.8, 1.0])
        forward = lambda x: np.interp(x, xp, fp)
        inverse = lambda y: np.interp(y, fp, xp)

        norm = FuncNorm((forward, inverse), vmin=4, vmax=24)

        ax.coastlines(linewidth=2.5)
        ax.set_extent([-25, 40, -3, 20])
        ax.set_xticks(np.arange(-25, 41, 10), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(0, 21, 5), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        im = ax.pcolormesh(lon_centres, lat_centres, mean_duration,
                             cmap='viridis', norm=norm)

        factor = 4

        # Interpolate the field
        mean_duration_zoom = zoom(np.nan_to_num(mean_duration), factor, order=3)

        print(mean_duration_zoom)

        # Create corresponding coordinates
        lon_zoom = np.linspace(lon_centres.min(), lon_centres.max(), mean_duration_zoom.shape[1])
        lat_zoom = np.linspace(lat_centres.min(), lat_centres.max(), mean_duration_zoom.shape[0])

        # Plot smooth contours
        # cs = ax.contour(
        #     lon_zoom,
        #     lat_zoom,
        #     mean_duration_zoom,
        #     levels=np.arange(4, 30, 6),
        #     colors="black", 
        #     transform=ccrs.PlateCarree()
        # )
        ax.set_title(f'{name}, n = {model_tracks.sizes["tracks"]}')

    axs = axs.flatten()
    axs[1].yaxis.set_tick_params(labelleft=False)
    axs[0].xaxis.set_tick_params(labelbottom=False)
    axs[1].xaxis.set_tick_params(labelbottom=False)
    axs[3].yaxis.set_tick_params(labelleft=False)

    cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Mean MCS track duration [hrs]')
    plt.subplots_adjust(hspace=-0.1, wspace=0.15)
    p_utils.match_colorbar_to_axes(fig, cbar, axs)
    plt.savefig('figs/mcs_spatial_dist_durations')



def temp_gradient(): 
        fig, axs = plt.subplots(2, 2, figsize=(15, 8),
                                    subplot_kw={'projection': ccrs.PlateCarree()}, sharey=True, sharex=True)
        
        for model_tracks, color, name, ax in zip(tracks_list, COLORS, MODEL_NAMES, axs.flatten()): 
            ds_name = models.models_name_dict[name]['path_id'].split('/')[1]  ## name used to access the catalog
            
            start_times = model_tracks.start_basetime.values

            # init_hours  = pd.DatetimeIndex(start_times).hour.astype(float)

            # needs_shifting = np.where(init_hours < 12)
            
            # needs_previous_day = start_times[needs_shifting]
            # new_start_times = needs_previous_day - pd.Timedelta(1, unit='D')
            # start_times[needs_shifting] = new_start_times
        
            # select_start_times = pd.DatetimeIndex(start_times).normalize() + pd.Timedelta(12, unit='hours')
            
            ### shift the start times of the 
            adjusted = np.where(
                pd.DatetimeIndex(start_times).hour < 12,
                start_times - pd.Timedelta(1, unit='D'),
                start_times
                )
            select_start_times = pd.DatetimeIndex(adjusted).normalize() + pd.Timedelta(12, unit='hours')

            ds3h = utils.open_region_dataset(ds_name, REGION_CFG)
            ds_pre_MCS = ds3h.sel(time=select_start_times)
            pre_MCS_temp = ds_pre_MCS.ta.sel(pressure=1000).mean(dim='time')
            pre_MCS_shear = (ds_pre_MCS.ua.sel(pressure=600) - ds_pre_MCS.ua.sel(pressure=850)).mean(dim='time')

            ax.set_extent(([-20, 20, 2.5, 14]))
            ax.coastlines()
            temp_plot = egh.healpix_contour(pre_MCS_temp, ax=ax, cmap='Reds')
            shear_plot = egh.healpix_contour(pre_MCS_shear, ax=ax, cmap='Greens')
            ax.clabel(temp_plot, fontsize=10, colors='black', inline=True)
            ax.clabel(shear_plot, fontsize=10, colors='green', inline=True)
            ax.set_xticks(np.arange(-20, 21, 10), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(0, 16, 5), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.set_title(f'{name}')

        axs = axs.flatten()

        axs[0].xaxis.set_tick_params(labelbottom=False)
        axs[1].yaxis.set_tick_params(labelleft=False)   # top right - no left labels
        axs[1].xaxis.set_tick_params(labelbottom=False) # bottom left - no bottom labels
        axs[3].yaxis.set_tick_params(labelleft=False)   # bottom right - no left labels

        plt.subplots_adjust(hspace=0.15)
        plt.savefig('figs/pre-MCS_LLtemp_gradient+shear')


BF_LAT_MIN, BF_LAT_MAX = 9, 15
BF_LON_MIN, BF_LON_MAX = -5, 5


def plot_region_map(ax, region_cfg, region_display):
    """Map showing the analysis region and Burkina Faso sub-box."""
    # Convert WAM lon to [-180, 180] for the map extent
    wam_lon_min = region_cfg['lon_min'] - 360 if region_cfg['lon_min'] > 180 else region_cfg['lon_min']
    wam_lon_max = region_cfg['lon_max']
    wam_lat_min = region_cfg['lat_min']
    wam_lat_max = region_cfg['lat_max']

    buf_lon_min = region_cfg['buf_lon_min']
    buf_lon_max = region_cfg['buf_lon_max']
    buf_lat_min = region_cfg['buf_lat_min']
    buf_lat_max = region_cfg['buf_lat_max']

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

    ax.add_patch(mpatches.Rectangle(
        (buf_lon_min, buf_lat_min), 
        buf_lon_max - buf_lon_min, buf_lat_max - buf_lat_min, 
        linewidth=1.5, edgecolor='black', facecolor='none', transform=proj, label=f'Tracking region', linestyle='--',

    ))



    # ax.legend(loc='lower left')
    # ax.set_title(f'Analysis regions')
    ax.set_xticks(np.arange(wam_lon_min - 5, wam_lon_max + 10, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(wam_lat_min - 11, wam_lat_max + 9, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    # plt.savefig('analysis_region.png', bbox_inches='tight')

def plot_map(region_cfg, region_display):
    fig_map, ax_map = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(6, 5))
    plot_region_map(ax_map, region_cfg, region_display)
    fig_map.tight_layout()
    p_utils.centre_legend_above(fig_map, ax_map)
    path = 'figs/analysis_region'
    fig_map.savefig(path)
    print(f'Saved {path}')
    plt.close(fig_map)


# plt.subplots_adjust(right=0.4, hspace=0.1)
# plt.savefig('MCS_spatial_dist.png')

# init_time_map() 

# init_time_dists() 

# temp_gradient()


# mcs_spatial_dist()
mcs_spatial_dist(track_density=True, jja=True, land_only=True)
# plot_mcs_durations(jja=True, land_only=True)

# region_display = REGION_CFG['display']
# plot_map(REGION_CFG, region_display)

# mcs_durations_map()