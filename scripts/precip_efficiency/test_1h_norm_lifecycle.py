
import argparse
import numpy as np 
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
    return models.done_dir(model) / f'mcs_PE_lifecycle_{region}_test.done'

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def _1h_lifecycle(input_zarr, times_3h, dstracks_wam, model, region): 
    """
    Overall aim (I think): 

    Have the 3hourly times that the dataset is available on 
    Have (for example) one track with an associated mask, that has 1hrly times 
    See when that track and its mask starts, in relation to any one of the 3hrly times 
    Set up some sort of bins (1h, 2h, 3h etc. ) then shifting to relation to next 3hrly timestep
    For the track, get the output data and put it in the correct bin (if it started 1h before the first available 3hrly 
    and lasts for 10h, then you technically have -1, 2, 5, 8)

    Still trying to figure this out

    """
    done_file = compute_donefile(model, region)
    if done_file.exists(): 
        print(f"1h lifecycle computation complete for {model} and {region}")
        return
    ## input data has dimensions (track, times_3h)
    print("computation started")
    
    ## tracks has the same length as dstracks_wam 
    cr = input_zarr.cr_mean.values
    pr = input_zarr.pr_mean.values

    print("both cr and pr computed")

    # First 3-hourly step index for each WAM track (index into times_3h array)
    start_times = dstracks_wam.start_basetime.values   # (n_tracks,)
    first_3h_step = np.searchsorted(times_3h, start_times)  

    
    dstracks_wam.track_duration.load()
    durations = dstracks_wam.track_duration.values
    

    ### want to output, for track id x, at cells where track id is, for 3h indices where track is, the 
    ## condensation rates, precip rates
    n_steps = len(times_3h)
    n_tracks = len(dstracks_wam.tracks)
    
    n_bins = 24 
    
    bin_values = [[] for _ in range(n_bins)]

    print("starting loop")

    for out_i in range(n_tracks): 
        if out_i % 100 == 0:
            print(f'  out_i {out_i}/{n_tracks}', flush=True)
        t_start_idx = first_3h_step[out_i]  ## index, not actual time
        start_time = start_times[out_i]
        closest_3h_idx = np.searchsorted(times_3h, start_time)
        closest_3h_time = times_3h[closest_3h_idx]

        offset = int((closest_3h_time - start_time) / np.timedelta64(1, 'h'))

        cr_track = cr[out_i] # (times_3h)
        pr_track = pr[out_i]

        n_active    = int(np.ceil(durations[out_i] / 3))
        t_end_idx   = min(t_start_idx + n_active + 2, n_steps)

        for step in range(t_start_idx, t_end_idx): 
            li = step - t_start_idx
            dur = durations[out_i]
            bin_idx = int(((offset + li * 3) / dur) * n_bins)
            if bin_idx < 0 or bin_idx >= n_bins:
                continue
            
            
        
            valid_cr = cr_track[li]
            valid_pr = pr_track[li]
            
            if np.isnan(valid_cr) or valid_cr == 0 or np.isnan(valid_pr):
                continue
            

            PE_t = valid_pr / valid_cr

            bin_values[bin_idx].append(float(PE_t))

    lifecycle_mean_PE = np.array([np.nanmean(b) if len(b) > 0 else np.nan for b in bin_values])
    lifecycle_pctg = np.linspace(0, 100, n_bins)  

    ds_lifecycle = xr.Dataset({
    'lifecycle_mean_PE': xr.DataArray(lifecycle_mean_PE, dims=['lifecycle_pctg'],
                                      attrs={'description': '1h normalised lifecycle mean PE'}),
    },
    coords={'lifecycle_pctg': lifecycle_pctg})

    ds_lifecycle.to_netcdf(models.data_dir(model, VAR) / f'lifecycle_PE_{region}.nc')


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
    _1h_lifecycle(input_zarr, times_3h, dstracks_wam, args.model, args.region)
    

    
if __name__ == '__main__':
    main()



