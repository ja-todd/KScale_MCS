import cmasher as cmr
import colormaps as cm
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
from scipy.stats import binned_statistic_2d
from scipy.stats import binned_statistic 
import src.hp_utils as utils 
import src.hp_models as models 
import xarray as xr 

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

BASE_PATH = '/gws/ssde/j25b/mcs_prime/jtodd/entrainment/data'
PE_BASE_PATH = '/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data'
MNAMES = list(models_dict.keys())
COLORS = [models_dict[mname]['color'] for mname in MNAMES]



model_choices = ['um_glm_n2560_RAL3p3_tuned_hk26', 'um_glm_n2560_CoMA9_hk26', 'um_glm_n1280_GAL9_v2_hk26']
model_display_names = ['RAL3 z9', 'CORMORPH n2560 z9', 'COMORPH n1280 z9']
durations = ['all', 'short', 'long']
seasons = ['all', 'jja', 'djf']
surfaces = ['all', 'land', 'ocean']
choice_colors = ['#0D0C6E', '#BC6263', '#BBCBDF' ]
new_cmap = cmr.get_sub_cmap('inferno', 0.3, 0.9)
### run from here




def binned_stats(x_axis, y_axis, vals, nx=20): 
    ny = nx
    xedges = np.linspace(-20, 20, nx + 1)
    yedges = np.linspace(-40, 40, ny + 1)
                    
    stat, _, _, _ = binned_statistic_2d(
        x_axis,
        y_axis,
        vals,
        statistic='mean',  
        bins=[xedges, yedges]
    )

    return xedges, yedges, stat



def shear_tbdiff_PE_plot(model_choices, model_display_names, seasons, durations, surfaces):
    
    for season in seasons: 
        for duration in durations: 
            for surface in surfaces: 
                fig, axs = plt.subplots(1, 3, figsize=(20, 5),  sharey=True)
                for ax, model, mname in zip(axs.flatten(), model_choices, model_display_names): 
                    
                    if season == 'all':
                        cmap = 'Greens'
                    elif season == 'djf': 
                        cmap = 'Blues'
                    elif season == 'jja': 
                        cmap = 'YlOrRd'

                    shear_entr_ds = xr.open_dataset(f'/gws/ssde/j25b/mcs_prime/jtodd/entrainment/data/z9/{model}/mcs_entrainment_wam.nc')
                    pe_ds = xr.open_zarr(f'/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/z9/{model}/mcs_condensation_rate_wam_stats.zarr')

                    track_nums = shear_entr_ds.tracks.values - 1 ### output (mask) idxs are offset by 1 from the true track idxs
                    


                    
                    MASK_URL = models.mask_url(model)
                    STATS_URL = models.stats_url(model)
                    mask_ds  = xr.open_zarr(MASK_URL, chunks={}, mask_and_scale=False)
                    dstracks = utils.load_track_stats(STATS_URL)
                    dstracks_wam = dstracks.isel(tracks=track_nums)
                    start_time = pd.DatetimeIndex(dstracks_wam.start_basetime.values)




                    jja_mask = (start_time.month >= 6) & (start_time.month <= 8)
                    djf_mask = (start_time.month <= 2) | (start_time.month == 12)



                    ## filter season 
                    if season == 'all': 
                        dstracks_wam = dstracks_wam
                        
                    if season == 'jja': 
                        season_mask = jja_mask
                        dstracks_wam = dstracks_wam.isel(tracks=season_mask)
                    elif season == 'djf': 
                        season_mask = djf_mask
                        dstracks_wam = dstracks_wam.isel(tracks=season_mask)

                    
                    track_durations = dstracks_wam.track_duration


                    ## filter duration 
                    if duration == 'all': 
                        pass

                    if duration == 'short': 
                        dstracks_wam = dstracks_wam.isel(tracks=(track_durations < 10))

                    elif duration == 'long': 
                        dstracks_wam = dstracks_wam.isel(tracks=(track_durations >= 10))


                    ## filter surface
                    dstracks_wam = utils.filter_surface(dstracks_wam, surface) ## default is all 



                    n_tracks = len(dstracks_wam.tracks)
                    mask_tracks = dstracks_wam.tracks.values + 1  ## get rid of the previous offset (probably not the right way of doing it but alright)


                    shear_entr_ds = shear_entr_ds.sel(tracks=mask_tracks)
                    pe_ds = pe_ds.sel(track=mask_tracks)



                    pe_vals = pe_ds.pe_mean.values
                    x_axis  = shear_entr_ds.shear_mean.values
                    y_axis  = shear_entr_ds.tb_diff_mean.values

                    valid = ~np.isnan(pe_vals) & ~np.isnan(x_axis) & ~np.isnan(y_axis) & (pe_vals <=1)

                    pe_vals = pe_vals[valid]
                    x_axis  = x_axis[valid]
                    y_axis  = y_axis[valid]

                    xedges, yedges, stat = binned_stats(x_axis, y_axis, pe_vals)
                
                    


                    plot = ax.pcolormesh(xedges, yedges, stat.T, cmap=cmap, vmin=0, vmax=1)

                    
                    ax.set_xlabel(r"Shear [m s$^{-1}$]")
                    
                
                    ax.set_xlim(-20, 20)
                    ax.set_ylim(-40, 40)
                    ax.invert_yaxis()
                    
                    ax.set_title(f'{mname.upper()} ' + r'n$_{tracks}$ = ' + f'{n_tracks}', pad=20)

                axs[0].set_ylabel(r"T$_b$ diff [K]")
                fig.colorbar(plot, ax=axs, label='Precip Efficiency')
                plt.suptitle(f'SEASON: {season.upper()}; SURFACE: {surface.upper()}; DURATIONS: {duration.upper()}', y=1.1)
                plt.savefig(f'figs/3Dhist_shear_tbdiff_PE_{mname}_{season}_{surface}_duration-{duration}.png', bbox_inches = 'tight')
                plt.close()


def plot_binned_line(x, y, bins, color, label, ax, stat='mean', spread='sem'):
    x = np.asarray(x)
    y = np.asarray(y)

    # bin edges
    bin_edges = np.linspace(np.nanmin(bins), np.nanmax(bins), len(bins))

    # mean (or median) of y in each bin
    means, edges, _ = binned_statistic(x, y, statistic=stat, bins=bin_edges)
    counts, _, _   = binned_statistic(x, y, statistic='count', bins=bin_edges)
    stds, _, _     = binned_statistic(x, y, statistic='std', bins=bin_edges)

    centers = 0.5 * (edges[:-1] + edges[1:])

    if spread == 'sem':
        err = stds / np.sqrt(np.maximum(counts, 1))
    else:  # 'std'
        err = stds

    # drop empty bins
    valid = counts > 0
    centers, means, err = centers[valid], means[valid], err[valid]

    ax.plot(centers, means, '-o', color=color, label=label)
    ax.fill_between(centers, means - err, means + err, color=color, alpha=0.25)

    # optional linear trend (dashed)
    coeffs = np.polyfit(centers, means, 1)
    trend = np.polyval(coeffs, centers)
    ax.plot(centers, trend, '--', color=color, linewidth=1)

    return centers, means, err

def shear_tbdiff_1Dhist(): 
    for season in seasons: 
        for duration in durations: 
            for surface in surfaces: 
                fig, axs = plt.subplots()
                for ax, model, mname, color in zip(axs.flatten(), model_choices, model_display_names, choice_colors): 
                    
                    if season == 'all':
                        cmap = 'Greens'
                    elif season == 'djf': 
                        cmap = 'Blues'
                    elif season == 'jja': 
                        cmap = 'YlOrRd'

                    shear_entr_ds = xr.open_dataset(f'/gws/ssde/j25b/mcs_prime/jtodd/entrainment/data/z9/{model}/mcs_entrainment_wam.nc')
                    pe_ds = xr.open_zarr(f'/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/z9/{model}/mcs_condensation_rate_wam_stats.zarr')

                    track_nums = shear_entr_ds.tracks.values - 1 ### output (mask) idxs are offset by 1 from the true track idxs
                    


                    
                    MASK_URL = models.mask_url(model)
                    STATS_URL = models.stats_url(model)
                    mask_ds  = xr.open_zarr(MASK_URL, chunks={}, mask_and_scale=False)
                    dstracks = utils.load_track_stats(STATS_URL)
                    dstracks_wam = dstracks.isel(tracks=track_nums)
                    start_time = pd.DatetimeIndex(dstracks_wam.start_basetime.values)




                    jja_mask = (start_time.month >= 6) & (start_time.month <= 8)
                    djf_mask = (start_time.month <= 2) | (start_time.month == 12)



                    ## filter season 
                    if season == 'all': 
                        dstracks_wam = dstracks_wam
                        
                    if season == 'jja': 
                        season_mask = jja_mask
                        dstracks_wam = dstracks_wam.isel(tracks=season_mask)
                    elif season == 'djf': 
                        season_mask = djf_mask
                        dstracks_wam = dstracks_wam.isel(tracks=season_mask)

                    
                    track_durations = dstracks_wam.track_duration


                    ## filter duration 
                    if duration == 'all': 
                        pass

                    if duration == 'short': 
                        dstracks_wam = dstracks_wam.isel(tracks=(track_durations < 10))

                    elif duration == 'long': 
                        dstracks_wam = dstracks_wam.isel(tracks=(track_durations >= 10))


                    ## filter surface
                    dstracks_wam = utils.filter_surface(dstracks_wam, surface) ## default is all 



                    n_tracks = len(dstracks_wam.tracks)
                    mask_tracks = dstracks_wam.tracks.values + 1  ## get rid of the previous offset (probably not the right way of doing it but alright)


                    shear_entr_ds = shear_entr_ds.sel(tracks=mask_tracks)
                    pe_ds = pe_ds.sel(track=mask_tracks)



                    pe_vals = pe_ds.pe_mean.values
                    x_axis  = shear_entr_ds.shear_mean.values
                    y_axis  = shear_entr_ds.tb_diff_mean.values

                    valid = ~np.isnan(pe_vals) & ~np.isnan(x_axis) & ~np.isnan(y_axis) & (pe_vals <=1)

                    pe_vals = pe_vals[valid]
                    x_axis  = x_axis[valid]
                    y_axis  = y_axis[valid]

                    

                    bin_edges = np.linspace(-20, 21, 20)
                    ax.set_xlabel(r"Shear [m s$^{-1}$]")
                    plot_binned_line(x_axis, y_axis, bin_edges, color=color, label=mname.upper(), ax=ax)
                    ax.set_ylabel(r"T$_b$ diff [K]")
                    ax.set_xlim(-20, 20)
                    ax.set_ylim(-40, 40)
                    ax.invert_yaxis()
                    
                plt.suptitle(f'SEASON: {season.upper()}; SURFACE: {surface.upper()}; DURATIONS: {duration.upper()}', y=1)
                plt.legend()
                plt.savefig(f'figs/1Dhist_shear_tbdiff_{season}_{surface}_duration-{duration}.png', bbox_inches = 'tight')
                plt.close()



def plot_1h_norm_lifecycle(): 
    fig, ax = plt.subplots()
    for mname, color in zip(MNAMES, COLORS): 
        m_id = models_dict[mname]['path_id']
        pe_ds = xr.open_dataset(f'/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/{m_id}/lifecycle_PE_wam.nc')

        ax.plot(pe_ds.lifecycle_pctg, pe_ds.lifecycle_mean_PE, color=color, label=mname)
        ax.set_xlabel(r'$\%$ of Lifecycle')
        ax.set_ylabel('Precip Efficiency')
    plt.legend()
    plt.savefig('figs/models_1h_norm_lifecycle.png', bbox_inches='tight')




# shear_tbdiff_PE_plot(model_choices, model_display_names, seasons, durations, surfaces)
# shear_tbdiff_1Dhist()
plot_1h_norm_lifecycle()