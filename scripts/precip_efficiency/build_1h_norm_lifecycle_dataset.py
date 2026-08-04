
import argparse
import numpy as np 
import pandas as pd
import src.hp_models as models 
import src.hp_utils as utils
from src.hp_utils import MAX_TIMES_3H
import xarray as xr


VAR = 'precip_efficiency'

"""
Pseudo-code: 

You have lots of MCSs, each one has a track with hourly data. Every three hours, you have the 3D data,
 and you want to make e.g. a lifecycle for all of these MCSs. For one MCS that starts at time 0, 
 you will only be able to produce the lifecycle at 3 hourly intervals (you can calc the values at the 'x's).
   One track is shown here, with time across the top:

0123456789
--x--x--x-

But you don't have just one MCS, you have lots. Each one starting at a random time (4 shown):

0123456789
--x--x--x-
 -x--x--x-
   --x--x-
     x--x-

For the first MCS, you have data at its 3rd timestep.
 For the second, at its 2nd. For the third, at its 3rd, and for the fourth, at its 1st.
   So averaged across all the MCSs, you can calculate a normalized lifecycle plot (of the variable defined at
     3 hourly intervals) at 1 hourly intervals. Even though for an individual MCS, this is not possible.


         
"""

def compute_donefile(model, region):
    return models.done_dir(model) / f'mcs_PE_tbdiff_lifecycle_{region}.done'

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def _1h_lifecycle(input_zarr, entr_ds, times_3h, dstracks_wam, model, region): 
    """
    Overall aim: 

    Have the 3hourly times that the dataset is available on 
    Have (for example) one track with an associated mask, that has 1hrly times 
    See when that track and its mask starts, in relation to any one of the 3hrly times 
    Set up some sort of bins (1h, 2h, 3h etc. ) then shifting to relation to next 3hrly timestep
    For the track, get the output data and put it in the correct bin (if it started 1h before the first available 3hrly 
    and lasts for 10h, then you technically have -1, 2, 5, 8)

    """
    done_file = models.done_dir(model) / f'mcs_lifecycle_PE_tbdiff_{region}.done'
    if done_file.exists():
        print(f"1h lifecycle computation complete for {model} and {region}")
        return

    print("computation started")

    # cr     = input_zarr.cr_mean.values    # (n_tracks, MAX_TIMES_3H)
    # pr     = input_zarr.pr_mean.values    # (n_tracks, MAX_TIMES_3H)
    pe     = input_zarr.pe_mean.values
    tbdiff = entr_ds.tb_diff_mean.values  # (n_tracks, MAX_TIMES_3H)
    w_eff = entr_ds.w_eff_mean.values
    start_time = pd.DatetimeIndex(dstracks_wam.start_basetime.values)
    jja_mask = (start_time.month >= 6) & (start_time.month <= 8)
    djf_mask = (start_time.month <= 2) | (start_time.month == 12)

    print("cr, pr, tbdiff loaded")

    season_filters = {
        'all': None, 
        'jja': jja_mask, 
        'djf': djf_mask
    }

    surface_filters = ['all', 'land', 'ocean']

    for season_label, season_mask in season_filters.items(): 

        print(f"processing season={season_label} tracks ...")

        if season_mask is not None:
            dstracks_season_filtered  = dstracks_wam.isel(tracks=season_mask)
        else:
            dstracks_season_filtered  = dstracks_wam

        for surface_filter in surface_filters:  

            
            
            surface_label = surface_filter
            print(f'processing surface={surface_label} tracks')
            dstracks_surface_filtered = utils.filter_surface(dstracks_season_filtered, surface_filter)


            dstracks_surface_filtered.track_duration.load()
            track_durations = dstracks_surface_filtered.track_duration.values

            duration_filters = {
                'all':   None,
                'short': track_durations < 10,
                'long':  track_durations >= 10,
            }

            n_bins_dict = {
                'all': 24, 
                'short': 8, 
                'long': 24


            }

            ds_lifecycle = xr.Dataset()

            n_steps = len(times_3h)

            for duration_label, mask in duration_filters.items():
                
                print(f"processing duration={duration_label} tracks ...")
                n_bins = n_bins_dict[duration_label]

                lifecycle_pctg = np.linspace(0, 100, n_bins)
                


                if mask is not None:
                    dstracks_duration_filtered  = dstracks_surface_filtered.isel(tracks=mask)
                    filtered_indices   = np.where(mask)[0]
                else:
                    dstracks_duration_filtered  = dstracks_surface_filtered
                    filtered_indices   = np.arange(len(dstracks_duration_filtered.tracks))

                start_times   = dstracks_duration_filtered.start_basetime.values
                first_3h_step = np.searchsorted(times_3h, start_times)
                durations     = dstracks_duration_filtered.track_duration.values
                n_tracks      = len(dstracks_duration_filtered.tracks)

                PE_bin_values     = [[] for _ in range(n_bins)]
                tbdiff_bin_values = [[] for _ in range(n_bins)]
                w_eff_bin_values = [[] for _ in range(n_bins)]


                for out_i in range(n_tracks):
                    if out_i % 100 == 0:
                        print(f'  out_i {out_i}/{n_tracks}', flush=True)

                    zarr_idx    = filtered_indices[out_i]
                    t_start_idx = first_3h_step[out_i]
                    start_time  = start_times[out_i]

                    closest_3h_idx  = np.searchsorted(times_3h, start_time)
                    closest_3h_time = times_3h[closest_3h_idx]
                    offset          = int((closest_3h_time - start_time) / np.timedelta64(1, 'h'))

                    # cr_track     = cr[zarr_idx]      # (MAX_TIMES_3H,)
                    # pr_track     = pr[zarr_idx]
                    pe_track       = pe[zarr_idx]
                    tbdiff_track = tbdiff[zarr_idx]
                    w_eff_track = w_eff[zarr_idx]


                    dur       = durations[out_i]
                    n_active  = int(np.ceil(dur / 3))
                    t_end_idx = min(t_start_idx + n_active + 2, n_steps)

                    for step in range(t_start_idx, t_end_idx):
                        li      = step - t_start_idx
                        bin_idx = int(((offset + li * 3) / dur) * n_bins)
                        if bin_idx < 0 or bin_idx >= n_bins:
                            continue

                        # valid_cr     = cr_track[li]
                        # valid_pr     = pr_track[li]
                        valid_pe       = pe_track[li]
                        valid_tbdiff = tbdiff_track[li]
                        valid_w_eff = w_eff_track[li]

                        if np.isnan(valid_pe):
                            continue

                        PE_t = valid_pe

                        PE_bin_values[bin_idx].append(float(PE_t))
                        if not np.isnan(valid_tbdiff):
                            tbdiff_bin_values[bin_idx].append(float(valid_tbdiff))
                        if not np.isnan(valid_w_eff): 
                            w_eff_bin_values[bin_idx].append(float(valid_w_eff))

                lifecycle_mean_PE     = np.array([np.nanmean(b) if len(b) > 0 else np.nan for b in PE_bin_values])
                lifecycle_mean_tbdiff = np.array([np.nanmean(b) if len(b) > 0 else np.nan for b in tbdiff_bin_values])
                lifecycle_mean_w_eff     = np.array([np.nanmean(b) if len(b) > 0 else np.nan for b in w_eff_bin_values])
                
                ds_lifecycle[f'lifecycle_pctg_{season_label}_{surface_label}_dur{duration_label}'] = xr.DataArray(
                    lifecycle_pctg, dims=[f'lifecycle_pctg_{season_label}_{surface_label}_dur{duration_label}'])
                
                ds_lifecycle[f'lifecycle_mean_PE_{season_label}_{surface_label}_dur{duration_label}'] = xr.DataArray(
                    lifecycle_mean_PE, dims=[f'lifecycle_pctg_{season_label}_{surface_label}_dur{duration_label}'],
                    attrs={'description': f'1h normalised lifecycle mean PE ({season_label} season, {surface_label} surface, {duration_label} duration tracks)'})
                
                ds_lifecycle[f'lifecycle_mean_tbdiff_{season_label}_{surface_label}_dur{duration_label}'] = xr.DataArray(
                    lifecycle_mean_tbdiff, dims=[f'lifecycle_pctg_{season_label}_{surface_label}_dur{duration_label}'],
                    attrs={'description': f'1h normalised lifecycle mean tb_diff ({season_label} season, {surface_label} surface, {duration_label} duration tracks)'})
                
                ds_lifecycle[f'lifecycle_mean_weff_{season_label}_{surface_label}_dur{duration_label}'] = xr.DataArray(
                    lifecycle_mean_w_eff, dims=[f'lifecycle_pctg_{season_label}_{surface_label}_dur{duration_label}'],
                    attrs={'description': f'1h normalised lifecycle mean w_eff ({season_label} season, {surface_label} surface, {duration_label} duration tracks)'})

    ds_lifecycle.to_netcdf(models.data_dir(model, VAR) / f'lifecycle_PE_tbdiff_{region}.nc')
    done_file.touch()
    print("Done")

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    models.add_model_arg(parser)
    models.add_region_arg(parser)
    # parser.add_argument('--n-timesteps', type=int, default=None, metavar='N',
    #                     help='Limit to first N timesteps (for testing)')
    
    args = parser.parse_args()

    region_cfg = models.REGIONS[args.region]

    # Patch module-level URL/path constants to match the chosen model/region.
    global MASK_URL, STATS_URL, ZOOM
    ZOOM             = models.MODELS[args.model]['zoom']
    MASK_URL         = models.mask_url(args.model)
    STATS_URL        = models.stats_url(args.model)
    
    dstracks = utils.load_track_stats(STATS_URL)
    mask_ds = xr.open_zarr(MASK_URL, chunks={})
    
    cr_ds     = xr.open_zarr(models.data_dir(args.model, VAR) / f'condensation_rate_{args.region}.zarr')
    precip_ds = utils.open_region_1h_dataset(args.model, region_cfg)

    
    dstracks_wam  = utils.filter_region_tracks(dstracks, region_cfg)
    
    _, _, times_3h = utils.align_times(cr_ds, mask_ds)
    precip_ds = precip_ds.sel(time=times_3h)

    print("opening input zarr")

    input_zarr = xr.open_zarr(models.data_dir(args.model, VAR) / f'mcs_condensation_rate_{args.region}_stats.zarr') #(track, times_3h, cell)
    entr_ds    = xr.open_dataset(models.data_dir(args.model) / 'mcs_entrainment_wam.nc')
    _1h_lifecycle(input_zarr, entr_ds, times_3h, dstracks_wam, args.model, args.region)
    

    
if __name__ == '__main__':
    main()



