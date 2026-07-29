import time
t0 = time.time()
import src.hp_models as models 
print(f"hp_models: {time.time()-t0:.1f}s")
t0 = time.time()
import src.hp_utils as utils
print(f"hp_utils: {time.time()-t0:.1f}s")
t0 = time.time() 
import src.plot_utils as p_utils
print(f"plot_utils: {time.time()-t0:.1f}s")
t0 = time.time()
import numpy as np 
print(f"numpy: {time.time()-t0:.1f}s")
t0 = time.time()
import cartopy.crs as ccrs
print(f"ccrs: {time.time()-t0:.1f}s")
t0 = time.time()
import matplotlib.pyplot as plt 
print(f"plt: {time.time()-t0:.1f}s")
t0 = time.time()
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
print(f"axes_formatters: {time.time()-t0:.1f}s")
t0 = time.time()
import pickle
print(f"pickle: {time.time()-t0:.1f}s")
t0 = time.time()

from pathlib import Path
print(f"Path: {time.time()-t0:.1f}s")
print(f"hp_models: {time.time()-t0:.1f}s")

import pandas as pd
print(f"pandas: {time.time()-t0:.1f}s")

import easygems.healpix as egh

t0 = time.time()
p_utils.apply_plot_style()
print(f"time to apply plot style: {time.time()-t0:.1f}s")

model_names = list(models.models_name_dict.keys())
region_cfg = models.REGIONS['wam']
COLORS = [models.models_name_dict[mname]['color'] for mname in model_names]


t0 = time.time()
cache_path = Path('../../data/tracks_list_cache.pkl')

if cache_path.exists():
    with open(cache_path, 'rb') as f:
        tracks_list = pickle.load(f)

else: 
    tracks_list = []
    for mname in model_names: 
        m_id         = models.models_name_dict[mname]['path_id'].split('/')[1]
        m_maskurl    = models.mask_url(m_id)
        m_statsurl   = models.stats_url(m_id)
        dstracks     = utils.load_track_stats(m_statsurl)
        dstracks_wam = utils.filter_region_tracks(dstracks, region_cfg)
        tracks_list.append(dstracks_wam)
    with open(cache_path, 'wb') as f:
        pickle.dump(tracks_list, f)

print(f"data processing: {time.time()-t0:.1f}s")


def mcs_spatial_dist(): 

    fig, axs = plt.subplots(2, 2, figsize=(15, 8),
                            subplot_kw={'projection': ccrs.PlateCarree()}, sharey=True, sharex=True)

    for model_tracks, ax, name in zip(tracks_list, axs.flatten(), model_names):

        lats = model_tracks.meanlat.values  # (n_tracks, n_times)
        lons = model_tracks.meanlon.values
        lons = (lons + 180) % 360 - 180

        # one point per track — mean position over lifetime
        track_mean_lat = np.nanmean(lats, axis=1)  # (n_tracks,)
        track_mean_lon = np.nanmean(lons, axis=1)

        valid = ~np.isnan(track_mean_lat) & ~np.isnan(track_mean_lon)
        track_mean_lat = track_mean_lat[valid]
        track_mean_lon = track_mean_lon[valid]

        lon_bins = np.arange(-20, 41, 2.5)
        lat_bins = np.arange(0, 21, 2.5)

        density, _, _ = np.histogram2d(track_mean_lat, track_mean_lon,
                                        bins=[lat_bins, lon_bins])

        density_pct = (density / model_tracks.sizes['tracks']) * 100

        lon_centres = (lon_bins[:-1] + lon_bins[1:]) / 2
        lat_centres = (lat_bins[:-1] + lat_bins[1:]) / 2

        ax.coastlines()
        ax.set_extent([-20, 40, 0, 20])
        ax.set_xticks(np.arange(-20, 41, 10), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(0, 21, 5), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        im = ax.pcolormesh(lon_centres, lat_centres, density_pct,
                            vmin=0, vmax=3, cmap='bone_r')
        ax.set_title(f'{name}, n = {model_tracks.sizes["tracks"]}')

    axs = axs.flatten()

    axs[1].yaxis.set_tick_params(labelleft=False)   # top right - no left labels
    axs[1].xaxis.set_tick_params(labelbottom=False) # bottom left - no bottom labels
    axs[3].yaxis.set_tick_params(labelleft=False)   # bottom right - no left labels

    cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])  # placeholder, will be resized
    cbar = fig.colorbar(im, cax=cbar_ax, label='Proportion of MCSs [%]')
    p_utils.match_colorbar_to_axes(fig, cbar, axs)
    plt.savefig('MCS_spatial_dist.png', bbox_inches='tight')




def init_time_map():
    fig, axs = plt.subplots(2, 2, figsize=(15, 8),
                         subplot_kw={'projection': ccrs.PlateCarree()}, sharey=True, sharex=True)
    for model_tracks, ax, name in zip(tracks_list, axs.flatten(), model_names):

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

        lon_bins = np.arange(-20, 41, 2.5)
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
        ax.set_extent([-20, 40, 0, 20])
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
    p_utils.match_colorbar_to_axes(fig, cbar, axs)
    plt.savefig('MCS_init_hour.png', bbox_inches='tight')




def temp_gradient(): 
        fig, axs = plt.subplots(2, 2, figsize=(15, 8),
                                    subplot_kw={'projection': ccrs.PlateCarree()}, sharey=True, sharex=True)
        
        for model_tracks, color, name, ax in zip(tracks_list, COLORS, model_names, axs.flatten()): 
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

            ds3h = utils.open_region_dataset(ds_name, region_cfg)
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
        plt.savefig('pre-MCS_LLtemp_gradient+shear.png', bbox_inches='tight')

def init_time_dists(): 
    t0 = time.time()
    fig, ax = plt.subplots()

    for model_tracks, color, name in zip(tracks_list, COLORS, model_names):
        start_times = model_tracks.start_basetime.values
        init_hours  = pd.DatetimeIndex(start_times).hour.astype(float)
        valid       = ~np.isnan(init_hours)
        init_hours  = init_hours[valid]

        bins = np.arange(0, 25, 1)
        counts, edges = np.histogram(init_hours, bins=bins)
        pct = (counts / len(init_hours)) * 100
        centres = 0.5 * (edges[:-1] + edges[1:])

        ax.plot(centres, pct, color=color, label=name, linewidth=2.5)

    ax.set_xlabel('Initiation hour (UTC)')
    ax.set_ylabel('% of MCS initiations')
    ax.set_xlim(0, 23)
    ax.set_xticks(np.arange(0, 24, 3))
    ax.legend(bbox_to_anchor=(1.3, 1.15), frameon=False, ncols=4)
    plt.savefig('MCS_init_hour_dist.png', bbox_inches='tight')
    print(f"savefig: {time.time()-t0:.1f}s")
# plt.subplots_adjust(right=0.4, hspace=0.1)
# plt.savefig('MCS_spatial_dist.png')

# init_time_map() 

# init_time_dists() 

temp_gradient()