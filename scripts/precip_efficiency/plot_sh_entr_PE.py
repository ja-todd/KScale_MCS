import cmasher as cmr
import colormaps as cm
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
from scipy.stats import binned_statistic_2d
from scipy.stats import binned_statistic 
import seaborn as sns
import src.hp_utils as utils 
import src.hp_models as models 
import xarray as xr 
from scipy.ndimage import zoom
import src.plot_utils as p_utils 

p_utils.apply_plot_style()

# models_dict = {
#     'RAL3_z10_4k' : {'path_id':'z10/um_glm_n2560_RAL3p3_tuned_sahel_z10_t4k', 'color': '#6893DA'},
#     'RAL3_z10_40k' : {'path_id':'z10/um_glm_n2560_RAL3p3_tuned_sahel_z10_t40k', 'color': '#782078' },
#     'RAL3_z9' : {'path_id': 'z9/um_glm_n2560_RAL3p3_tuned_hk26', 'color': '#0D0C6E'},
#     'GAL9' : {'path_id': 'z9/um_glm_n1280_GAL9_v2_hk26', 'color':'#AC2078'},
#     'COMORPH_n2560_z9'     : {'path_id': 'z9/um_glm_n2560_CoMA9_hk26', 'color': '#BC6263'}, 
#     'COMORPH_n1280' : {'path_id': 'z9/um_glm_n1280_CoMA9_hk26', 'color':  '#B79394'},
# }


models_dict = models.models_name_dict
BASE_PATH = '/gws/ssde/j25b/mcs_prime/jtodd/entrainment/data'
PE_BASE_PATH = '/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data'
MNAMES = list(models_dict.keys())[2:]
COLORS = [models_dict[mname]['color'] for mname in MNAMES]


durations = ['all', 'short', 'long']
seasons = ['all', 'jja', 'djf']
surfaces = ['all', 'land', 'ocean']



def binned_stats(x_axis, y_axis, vals, nx=20, statistic='mean', clip=False):
    ny = nx

    if clip:
        xedges = np.linspace(np.percentile(x_axis, 5), np.percentile(x_axis, 95), nx + 1)
        yedges = np.linspace(np.percentile(y_axis, 5), np.percentile(y_axis, 95), ny + 1)
    else:
        xedges = np.linspace(x_axis.min(), x_axis.max(), nx + 1)
        yedges = np.linspace(y_axis.min(), y_axis.max(), ny + 1)

    stat, _, _, _ = binned_statistic_2d(
        x_axis,
        y_axis,
        vals,
        statistic=statistic,
        bins=[xedges, yedges]
    )

    return xedges, yedges, stat


def build_jointgrid_axes(subfig, ratio=5, space=0.15,
                          density_labels=False, cbar=False, **kwargs):
    gs = subfig.add_gridspec(
        2, 2,
        width_ratios=[ratio, 1], height_ratios=[1, ratio],
        wspace=space, hspace=space
    )
    ax_marg_x = subfig.add_subplot(gs[0, 0])
    ax_joint  = subfig.add_subplot(gs[1, 0])
    ax_marg_y = subfig.add_subplot(gs[1, 1])
    ax_marg_x.set_xticklabels([]); ax_marg_x.set_xlabel('')
    ax_marg_y.set_yticklabels([]); ax_marg_y.set_ylabel('')

    ax_marg_x.spines['left'].set_visible(True)
    ax_marg_y.spines['bottom'].set_visible(True)
    ax_marg_x.tick_params(axis='y', left=True, labelleft=True)
    ax_marg_y.tick_params(axis='x', bottom=True, labelbottom=True)
    ax_marg_x.yaxis.label.set_visible(True)
    ax_marg_y.xaxis.label.set_visible(True)

    if density_labels:
        ax_marg_x.set_ylabel("Density")
        ax_marg_y.set_xlabel("Density")

    if cbar:
        pos = ax_marg_y.get_position()  # use ax_marg_y so cbar sits just right of it
        joint_pos = ax_joint.get_position()
        cbar_ax = subfig.add_axes([
            pos.x1 + 0.02,       # a bit to the right of ax_marg_y
            joint_pos.y0,        # bottom aligned with ax_joint
            0.03,                 # width
            joint_pos.height      # exact same height as ax_joint
        ])
        
    return ax_joint, ax_marg_x, ax_marg_y, cbar_ax


def shear_tbdiff_PE_plot(seasons, durations, surfaces):
    
    for season in seasons: 
        for duration in durations: 
            for surface in surfaces: 
                fig, axs = plt.subplots(1, 6, figsize=(40, 5),  sharey=True)
                for ax, mname in zip(axs.flatten(), list(models_dict.keys())): 
                    
                    m_pid = models_dict[mname]['path_id']
                    model = m_pid.split('/')[1]

                    if season == 'all':
                        cmap = 'Greens'
                    elif season == 'djf': 
                        cmap = 'Blues'
                    elif season == 'jja': 
                        cmap = 'YlOrRd'

                    shear_entr_ds = xr.open_dataset(f'/gws/ssde/j25b/mcs_prime/jtodd/entrainment/data/{m_pid}/mcs_entrainment_wam.nc')
                    pe_ds = xr.open_zarr(f'/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/{m_pid}/mcs_condensation_rate_wam_stats.zarr')

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

                    xedges, yedges, stat = binned_stats(x_axis, y_axis, pe_vals, clip=True)
                
                    


                    plot = ax.pcolormesh(xedges, yedges, stat.T, cmap=cmap, vmin=0, vmax=1)

                    
                    ax.set_xlabel(r"Shear [m s$^{-1}$]")
                    
                
                    ax.set_xlim(-20, 20)
                    ax.set_ylim(-40, 40)
                    ax.invert_yaxis()
                    
                    ax.set_title(f'{mname.upper()} ' + r'n$_{tracks}$ = ' + f'{n_tracks}', pad=20)

                axs[0].set_ylabel(r"T$_b$ diff [K]")
                plt.subplots_adjust(wspace=0.7)
                fig.colorbar(plot, ax=axs, label='Precip Efficiency')
                plt.suptitle(f'SEASON: {season.upper()}; SURFACE: {surface.upper()}; DURATIONS: {duration.upper()}', y=1.1)
                
                plt.savefig(f'figs/2Dhist_shear_tbdiff_PE_{season}_{surface}_duration-{duration}.png', bbox_inches = 'tight')
                plt.close()




def joint_dists_tbdiff_shear(seasons, durations, surfaces, lifetime_mean=False):
    
    for season in seasons: 
        for duration in durations: 
            for surface in surfaces: 
                fig = plt.figure(figsize=(20, 10))
                subfigs = fig.subfigures(2, 3)
                for subfig, mname in zip(subfigs.flatten(), list(models_dict.keys())): 
                    
                    ax_joint, ax_marg_x, ax_marg_y, cbar_ax = build_jointgrid_axes(subfig, cbar='True')

                    m_pid = models_dict[mname]['path_id']
                    model = m_pid.split('/')[1]

                    if season == 'all':
                        cmap = 'Greens'
                    elif season == 'djf': 
                        cmap = 'Blues'
                    elif season == 'jja': 
                        cmap = 'YlOrRd'

                    shear_entr_ds = xr.open_dataset(f'/gws/ssde/j25b/mcs_prime/jtodd/entrainment/data/{m_pid}/mcs_entrainment_wam.nc')
                    pe_ds = xr.open_zarr(f'/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/{m_pid}/mcs_condensation_rate_wam_stats.zarr')

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



                    pe_vals = pe_ds.pe_mean
                    x_axis  = shear_entr_ds.shear_mean
                    y_axis  = shear_entr_ds.tb_diff_mean

                    if lifetime_mean: 
                        pe_vals = pe_vals.mean(dim='times_3h', skipna=True)
                        x_axis  = x_axis.mean(dim='times_3h', skipna=True)
                        y_axis  = y_axis.mean(dim='times_3h', skipna=True)

                    pe_vals = pe_vals.values

                    valid = ~np.isnan(pe_vals) & ~np.isnan(x_axis) & ~np.isnan(y_axis)

                    pe_vals = pe_vals[valid]
                    x_axis  = x_axis[valid]
                    y_axis  = y_axis[valid]

                    xedges, yedges, counts = binned_stats(x_axis, y_axis, pe_vals, statistic='count', clip=True)
                
                    
                    # g = make_jointgrid(space=0.5, density_labels=True, cbar=True)
                    x, y = x_axis, y_axis

                    # sns.histplot(x=x, y=y, ax=g.ax_joint, bins=20, cbar=True,
                                # cbar_ax=g.cbar_ax, cbar_kws=dict(label='Count'))

                    plot = ax_joint.pcolormesh(xedges, yedges, counts.T, cmap=cmap)
                    fig.colorbar(plot, cax=cbar_ax, label='Number of MCSs')
                    # g.figure.colorbar(plot, cax=g.cbar_ax, label='Number of MCSs')
                    sns.histplot(x=x, ax=ax_marg_x, color='black', stat='density', bins=20, kde=True)
                    sns.histplot(y=y, ax=ax_marg_y, color='black', stat='density', bins=20, kde=True)

                    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
                    ax_joint.axline((lims[0], lims[0]), (lims[1], lims[1]), linestyle='--', color='black')
                    ax_joint.set_xlabel(r'u600 - u850 [m s$^{-1}$]')
                    ax_joint.set_ylabel(r'T$_b$diff')                    

                    # g.ax_joint.set_xticks(np.arange(0, 6, 1))
                    # g.ax_joint.set_yticks(np.arange(0, 6, 1))
                    ax_joint.set_xlim(xedges.min(), xedges.max())
                    ax_joint.set_ylim(yedges.min(), yedges.max())

                    
        
                    
                    
                    ax_marg_x.set_title(f'{mname.upper()} ' + r'n$_{tracks}$ = ' + f'{n_tracks}', pad=20)

                plt.suptitle(f'SEASON: {season.upper()}; SURFACE: {surface.upper()}; DURATIONS: {duration.upper()}', y=1.1)
                
                plt.savefig(f'figs/jointdist_shear_tbdiff_MCScount_{season}_{surface}_duration-{duration}.png', bbox_inches = 'tight')
                plt.close()
                



def joint_dists_cr_pr(seasons, durations, surfaces, lifetime_mean=False):
    
    for season in seasons: 
        for duration in durations: 
            for surface in surfaces: 
                fig = plt.figure(figsize=(25, 10))
                subfigs = fig.subfigures(2, 2)
                for subfig, mname in zip(subfigs.flatten(), list(models_dict.keys())[2:]): 
                    output_addon = ''
                    ax_joint, ax_marg_x, ax_marg_y, cbar_ax = build_jointgrid_axes(subfig, cbar='True')

                    m_pid = models_dict[mname]['path_id']
                    model = m_pid.split('/')[1]

                    if season == 'all':
                        cmap = 'viridis'
                    elif season == 'djf': 
                        cmap = 'cividis'
                    elif season == 'jja': 
                        cmap = 'plasma'

                    shear_entr_ds = xr.open_dataset(f'/gws/ssde/j25b/mcs_prime/jtodd/entrainment/data/{m_pid}/mcs_entrainment_wam.nc')
                    pe_ds = xr.open_zarr(f'/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/{m_pid}/mcs_condensation_rate_wam_stats.zarr')

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



                    cr = pe_ds.cr_mean * 3600
                    x_axis  = pe_ds.pr_mean * 3600
                    y_axis  = pe_ds.pe_mean

                    

                    if lifetime_mean: 
                        output_addon = '_lifetime_mean'
                        cr = cr.mean(dim='times_3h', skipna=True)
                        x_axis = x_axis.mean(dim='times_3h', skipna=True)
                        y_axis = y_axis.mean(dim='times_3h', skipna=True)

                    cr = cr.values 
                    x_axis = x_axis.values 
                    y_axis = y_axis.values

                    valid = ~np.isnan(cr) & ~np.isnan(x_axis) & ~np.isnan(y_axis) 

                    cr = cr[valid]
                    x_axis  = x_axis[valid]
                    y_axis  = y_axis[valid]

                    xedges, yedges, counts = binned_stats(cr, x_axis, y_axis, nx=20, statistic='count', clip=True)
                
                    
                    # g = make_jointgrid(space=0.5, density_labels=True, cbar=True)
                    x, y = cr, x_axis

                    # sns.histplot(x=x, y=y, ax=g.ax_joint, bins=20, cbar=True,
                                # cbar_ax=g.cbar_ax, cbar_kws=dict(label='Count'))
                    
                    n_valid_MCS = counts.sum()
                    percentages = (counts.T / n_valid_MCS) * 100 

                    plot = ax_joint.pcolormesh(xedges, yedges, percentages, cmap=cmap, vmin=0, vmax=1.5)
                    fig.colorbar(plot, cax=cbar_ax, label=r'$\%$ MCSs')
                    # fig.colorbar(plot, cax=cbar_ax, label='Number of MCSs')
                    # g.figure.colorbar(plot, cax=g.cbar_ax, label='Number of MCSs')
                    sns.histplot(x=x, ax=ax_marg_x, color='black', stat='density', bins=20, kde=True)
                    sns.histplot(y=y, ax=ax_marg_y, color='black', stat='density', bins=20, kde=True)

                    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
                    ax_joint.axline((lims[0], lims[0]), (lims[1], lims[1]), linestyle='--', color='black')
                    ax_joint.set_xlabel(r'Condensation rate [mm hr$^{-1}$]')
                    ax_joint.set_ylabel(r'Surface precipitation [mm hr$^{-1}$]')                    

                    ax_joint.set_xlim(xedges.min(), xedges.max())
                    ax_joint.set_ylim(yedges.min(), yedges.max())

                    
        
                    
                    
                    ax_marg_x.set_title(f'{mname.upper()} ' + r'n$_{tracks}$ = ' + f'{n_tracks}', pad=20)

                plt.suptitle(f'SEASON: {season.upper()}; SURFACE: {surface.upper()}; DURATIONS: {duration.upper()}', y=1.1)
                plt.subplots_adjust(wspace=0.3)
                plt.savefig(f'figs/jointdist{output_addon}_cr_pr_MCScount_{season}_{surface}_duration-{duration}.png', bbox_inches = 'tight')
                plt.close()



def joint_dists_cr_pr_plus_contour(seasons, durations, surfaces):
    
    for season in seasons: 
        for duration in durations: 
            for surface in surfaces: 
                fig = plt.figure(figsize=(25, 10))
                subfigs = fig.subfigures(2, 2)
                for subfig, mname in zip(subfigs.flatten(), list(models_dict.keys())[2:]): 
                    output_addon = ''
                    ax_joint, ax_marg_x, ax_marg_y, cbar_ax = build_jointgrid_axes(subfig, cbar='True')

                    m_pid = models_dict[mname]['path_id']
                    model = m_pid.split('/')[1]

                    if season == 'all':
                        cmap = 'viridis'
                    elif season == 'djf': 
                        cmap = 'cividis'
                    elif season == 'jja': 
                        cmap = 'plasma'

                    shear_entr_ds = xr.open_dataset(f'/gws/ssde/j25b/mcs_prime/jtodd/entrainment/data/{m_pid}/mcs_entrainment_wam.nc')
                    pe_ds = xr.open_zarr(f'/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/{m_pid}/mcs_condensation_rate_wam_stats.zarr')

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
                    mask_tracks = dstracks_wam.tracks.values + 1  ## get rid of the previous offset

                    shear_entr_ds = shear_entr_ds.sel(tracks=mask_tracks)
                    pe_ds = pe_ds.sel(track=mask_tracks)

                    # non-lifetime-mean 
                    cr_raw = (pe_ds.cr_mean * 3600).values
                    pr_raw = (pe_ds.pr_mean * 3600).values
                    pe_raw = pe_ds.pe_mean.values

                    valid_raw = ~np.isnan(cr_raw) & ~np.isnan(pr_raw) & ~np.isnan(pe_raw)
                    cr_raw, pr_raw, pe_raw = cr_raw[valid_raw], pr_raw[valid_raw], pe_raw[valid_raw]

                    # lifetime-mean
                    cr_lt = (pe_ds.cr_mean * 3600).mean(dim='times_3h', skipna=True).values
                    pr_lt = (pe_ds.pr_mean * 3600).mean(dim='times_3h', skipna=True).values
                    pe_lt = pe_ds.pe_mean.mean(dim='times_3h', skipna=True).values

                    valid_lt = ~np.isnan(cr_lt) & ~np.isnan(pr_lt) & ~np.isnan(pe_lt)
                    cr_lt, pr_lt, pe_lt = cr_lt[valid_lt], pr_lt[valid_lt], pe_lt[valid_lt]

                    # define the share grid with the raw data + clip between 5th and 95th percentile
                    cr_edges, pr_edges, counts_raw = binned_stats(cr_raw, pr_raw, pe_raw, nx=20, statistic='count', clip=True)

                    # Bin the lifetime-mean data onto the shared grid
                    counts_lt, _, _, _ = binned_statistic_2d(cr_lt, pr_lt, pe_lt, statistic='count', bins=[cr_edges, pr_edges])

                    # histogram bins (colors)
                    n_valid_raw = counts_raw.sum()
                    percentages_raw = (counts_raw.T / n_valid_raw) * 100

                    plot = ax_joint.pcolormesh(cr_edges, pr_edges, percentages_raw, cmap=cmap, vmin=0, vmax=1.5)
                    fig.colorbar(plot, cax=cbar_ax, label=r'$\%$ MCSs')

                    # contours for the lifetime mean data
                    xcenters = (cr_edges[:-1] + cr_edges[1:]) / 2
                    ycenters = (pr_edges[:-1] + pr_edges[1:]) / 2
                    n_valid_lt = counts_lt.sum()
                    percentages_lt = (counts_lt.T / n_valid_lt) * 100

                    # Mask zero bins 
                    percentages_lt_masked = np.where(percentages_lt == 0, np.nan, percentages_lt)

                    # set the contour levels
                    levels = np.linspace(np.nanpercentile(percentages_lt_masked, 10), np.nanmax(percentages_lt_masked), 3)

                    # smooth contours
                    zoom_factor = 4
                    percentages_lt_upsampled = zoom(np.nan_to_num(percentages_lt_masked), zoom=zoom_factor, order=3)

                    xcenters_fine = np.linspace(xcenters.min(), xcenters.max(), percentages_lt_upsampled.shape[1])
                    ycenters_fine = np.linspace(ycenters.min(), ycenters.max(), percentages_lt_upsampled.shape[0])

                    contour_plot = ax_joint.contour(
                        xcenters_fine, ycenters_fine, percentages_lt_upsampled,
                        colors='red', levels=levels, linewidths=2.5
                    )
                    ax_joint.clabel(contour_plot, inline=True, fontsize=10)

                    # marginal plots
                    sns.histplot(x=cr_raw, ax=ax_marg_x, color='black', stat='density', bins=20, kde=True)
                    sns.histplot(y=pr_raw, ax=ax_marg_y, color='black', stat='density', bins=20, kde=True)
                    sns.kdeplot(x=cr_lt, ax=ax_marg_x, color='red', linewidth=2)
                    sns.kdeplot(y=pr_lt, ax=ax_marg_y, color='red', linewidth=2)  # fixed: was ax_marg_x

                    # 1:1 line
                    lims = [min(cr_raw.min(), pr_raw.min()), max(cr_raw.max(), pr_raw.max())]
                    ax_joint.axline((lims[0], lims[0]), (lims[1], lims[1]), linestyle='--', color='black')

                    ax_joint.set_xlabel(r'Condensation rate [mm hr$^{-1}$]')
                    ax_joint.set_ylabel(r'Surface precipitation [mm hr$^{-1}$]')

                    ax_joint.set_xlim(cr_edges.min(), cr_edges.max())
                    ax_joint.set_ylim(pr_edges.min(), pr_edges.max())

                    ax_marg_x.set_title(f'{mname.upper()} ' + r'n$_{tracks}$ = ' + f'{n_tracks}', pad=20)

                plt.suptitle(f'SEASON: {season.upper()}; SURFACE: {surface.upper()}; DURATIONS: {duration.upper()}', y=1.1)
                plt.subplots_adjust(wspace=0.3)
                plt.savefig(f'figs/joint_dists/jointdist_overlay_cr_pr_MCScount_{season}_{surface}_duration-{duration}.png', bbox_inches = 'tight')
                plt.close()


def joint_dists_vars(seasons, durations, surfaces, vars=['cr', 'pr'], lifetime_mean=False):
    for season in seasons: 
        for duration in durations: 
            for surface in surfaces: 
                fig = plt.figure(figsize=(25, 10))
                subfigs = fig.subfigures(2, 2)
                for subfig, mname in zip(subfigs.flatten(), list(models_dict.keys())[:4]): 
                    output_addon = ''
                    ax_joint, ax_marg_x, ax_marg_y, cbar_ax = build_jointgrid_axes(subfig, cbar='True')

                    m_pid = models_dict[mname]['path_id']
                    model = m_pid.split('/')[1]

                    if season == 'all':
                        cmap = 'Greens'
                    elif season == 'djf': 
                        cmap = 'Blues'
                    elif season == 'jja': 
                        cmap = 'YlOrRd'

                    

                    shear_entr_ds = xr.open_dataset(f'/gws/ssde/j25b/mcs_prime/jtodd/entrainment/data/{m_pid}/mcs_entrainment_wam.nc')
                    pe_ds = xr.open_zarr(f'/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/{m_pid}/mcs_condensation_rate_wam_stats.zarr')

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



                    cr = pe_ds.cr_mean * 3600
                    x_axis  = pe_ds.pr_mean * 3600
                    y_axis  = pe_ds.pe_mean

                    

                    if lifetime_mean: 
                        output_addon = '_lifetime_mean'
                        cr = cr.mean(dim='times_3h', skipna=True).values
                        x_axis = x_axis.mean(dim='times_3h', skipna=True).values
                        y_axis = y_axis.mean(dim='times_3h', skipna=True).values

                    valid = ~np.isnan(cr) & ~np.isnan(x_axis) & ~np.isnan(y_axis) 

                    cr = cr[valid]
                    x_axis  = x_axis[valid]
                    y_axis  = y_axis[valid]

                    xedges, yedges, counts = binned_stats(cr, x_axis, y_axis, nx=20, statistic='count', clip=True)
                
                    
                    # g = make_jointgrid(space=0.5, density_labels=True, cbar=True)
                    x, y = cr, x_axis

                
                    n_valid_MCS = counts.sum()
                    percentages = (counts.T / n_valid_MCS) * 100 

                    plot = ax_joint.pcolormesh(xedges, yedges, percentages, cmap=cmap, vmin=0, vmax=1.5)
                    fig.colorbar(plot, cax=cbar_ax, label=r'$\%$ MCSs')
                    # fig.colorbar(plot, cax=cbar_ax, label='Number of MCSs')
                    # g.figure.colorbar(plot, cax=g.cbar_ax, label='Number of MCSs')
                    sns.histplot(x=x, ax=ax_marg_x, color='black', stat='density', bins=20, kde=True)
                    sns.histplot(y=y, ax=ax_marg_y, color='black', stat='density', bins=20, kde=True)

                    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
                    ax_joint.axline((lims[0], lims[0]), (lims[1], lims[1]), linestyle='--', color='black')
                    ax_joint.set_xlabel(r'Condensation rate [mm hr$^{-1}$]')
                    ax_joint.set_ylabel(r'Surface precipitation [mm hr$^{-1}$]')                    

                    ax_joint.set_xlim(xedges.min(), xedges.max())
                    ax_joint.set_ylim(yedges.min(), yedges.max())

                    
        
                    
                    
                    ax_marg_x.set_title(f'{mname.upper()} ' + r'n$_{tracks}$ = ' + f'{n_tracks}', pad=20)

                plt.suptitle(f'SEASON: {season.upper()}; SURFACE: {surface.upper()}; DURATIONS: {duration.upper()}', y=1.1)
                plt.subplots_adjust(wspace=0.3)
                plt.savefig(f'figs/jointdist{output_addon}_cr_pr_MCScount_{season}_{surface}_duration-{duration}.png', bbox_inches = 'tight')
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
                fig, ax = plt.subplots()
                for mname, color in zip(list(models_dict.keys()), COLORS): 
                    
                    m_pid = models_dict[mname]['path_id']
                    model = m_pid.split('/')[1]

                    if season == 'all':
                        cmap = 'Greens'
                    elif season == 'djf': 
                        cmap = 'Blues'
                    elif season == 'jja': 
                        cmap = 'YlOrRd'

                    shear_entr_ds = xr.open_dataset(f'/gws/ssde/j25b/mcs_prime/jtodd/entrainment/data/{m_pid}/mcs_entrainment_wam.nc')
                    pe_ds = xr.open_zarr(f'/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/{m_pid}/mcs_condensation_rate_wam_stats.zarr')

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
                    
                plt.suptitle(f'SEASON: {season.upper()}; SURFACE: {surface.upper()}; DURATIONS: {duration.upper()}', y=1.2)
                
                plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncols=3)
                plt.savefig(f'figs/1Dhist_shear_tbdiff_{season}_{surface}_duration-{duration}.png', bbox_inches = 'tight')
                plt.close()



def plot_1h_norm_lifecycle(): 
    durations = ['all', 'short', 'long']
    for duration in durations: 
        fig, ax = plt.subplots()
        ax1 = ax.twinx()
        for mname, color in zip(MNAMES, COLORS): 
            m_id = models_dict[mname]['path_id']
            pe_ds = xr.open_dataset(f'/gws/ssde/j25b/mcs_prime/jtodd/precip_efficiency/data/{m_id}/lifecycle_PE_tbdiff_wam.nc')

            ax.plot(pe_ds[f'lifecycle_pctg_{duration}'], pe_ds[f'lifecycle_mean_PE_{duration}'], color=color, label=mname, linewidth=3)
            ax1.plot(pe_ds[f'lifecycle_pctg_{duration}'], pe_ds[f'lifecycle_mean_tbdiff_{duration}'], color=color, linewidth=2.5, linestyle='--')
            ax.set_xlabel(r'$\%$ of Lifecycle')
            ax.set_ylabel('Precip Efficiency')
            ax1.set_ylabel(r'T$_b$ diff [K]')
            ax.spines['right'].set_visible(True)
        ax.plot([], [], color='grey', alpha=0.4, linewidth=3, label='PE')    
        ax.plot([], [], color='grey', alpha=0.4, linewidth=2.5, linestyle='--', label=r'T$_b$ diff')  
        ax.legend(bbox_to_anchor = (1.15, 1.3), ncols=3)
        
        plt.savefig(f'figs/z9_models_1h_norm_PE+tbdiff_lifecycle_durations-{duration}.png', bbox_inches='tight')




# shear_tbdiff_PE_plot(seasons, durations, surfaces)
# shear_tbdiff_1Dhist()
# plot_1h_norm_lifecycle()

# joint_dists_tbdiff_shear(seasons, durations, surfaces) 

# joint_dists_cr_pr(seasons, durations, surfaces, lifetime_mean=True) 

joint_dists_cr_pr_plus_contour(seasons, durations, surfaces)