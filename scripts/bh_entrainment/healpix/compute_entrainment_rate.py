""" Computes the per-track Becker and Hohenegger (2021) frozen MSE entrainment rate

Usage: python compute_entrainment_rate.py --init --model <model_id> --region <region> (wam default)
python compute_entrainment_rate.py --run --model <model_id> --region <region> (wam default)

Output: 
mcs_entr_rate_<region>.zarr 

dims: tracks: ; times_3h: ; pressure: 

currently no functionality to filter by surface, but should not be difficult to introduce

"""


import xarray as xr 
import numpy as np 
import src.hp_models as models
from src.hp_utils import open_region_dataset, load_track_stats,\
      filter_region_tracks, filter_surface, MAX_TIMES_3H
import argparse
from pathlib import Path
import dask.array as dsa
import pandas as pd

# CHUNK_SIZE = 10 


#-----------------------------------------------------------------------
# Zarr store initialization 
#-----------------------------------------------------------------------

def init_zarr(model, region, dstracks_wam, radius=50, surface='all'): 
    region_cfg = models.REGIONS[region]
    ds = open_region_dataset(model, region_cfg)


    n_times  = ds.sizes['time']
    n_pressures = ds.sizes['pressure']
    n_cells  = ds.sizes['cell']
    # n_chunks = (n_times + CHUNK_SIZE - 1) // CHUNK_SIZE

    n_tracks = dstracks_wam.sizes['tracks']

    template = xr.Dataset({
    'entrainment_rate': xr.DataArray(
        dsa.full((n_tracks, MAX_TIMES_3H, n_pressures), np.nan, dtype=np.float32,
                 chunks=(n_tracks, MAX_TIMES_3H, n_pressures)),
        dims=['tracks', 'times_3h', 'pressure'],
        attrs={'units': 'm-1'}),
    'base_time': xr.DataArray(
        dsa.full((n_tracks, MAX_TIMES_3H), np.datetime64('NaT', 'ns'), dtype='datetime64[ns]',
                 chunks=(n_tracks, MAX_TIMES_3H)),
        dims=['tracks', 'times_3h'])
    },
    coords={'tracks': dstracks_wam.tracks, 'pressure': ds.pressure.sortby('pressure', ascending=False)})

    if surface == 'all': 
        zarr_path = models.data_dir(model) / f'mcs_entr_rate_{region}_{radius}km.zarr'
    else: 
        zarr_path = models.data_dir(model) / f'mcs_entr_rate_{region}_{surface}_{radius}km.zarr'

    zarr_path.parent.mkdir(parents=True, exist_ok=True) 
    template.to_zarr(zarr_path, mode='w', zarr_format=2)

    done_dir = models.done_dir(model)
    done_dir.mkdir(parents=True, exist_ok=True)
    models.init_donefile(model, region, tag=f'mcs_entr_rate_{surface}_{radius}km').touch()
    print(f'Created {zarr_path}  shape=({n_tracks}, {MAX_TIMES_3H}, {n_pressures})')
    

#-----------------------------------------------------------------------
# Core computation
#-----------------------------------------------------------------------

def compute_entr_rate(ds):
    # mass flux weights
    mass_flux = ds.rho_updraft * ds.w_updraft  # (time, pressure, cell)

    if mass_flux.sum().values == 0: 
        return None

    # weighted mean fmse profile over all updraft cells and time
    fmse_u = (ds.fmse_updraft * mass_flux).sum('cell') / mass_flux.sum('cell')  
    fmse_e = ds.fmse_env.mean('cell')   
    z_mean = ds.z_updraft.mean('cell')  

    # convert to numpy for differentiation
    fmse_u_np  = fmse_u.values 
    z_np       = z_mean.values  

    dh_dz = np.gradient(fmse_u_np, z_np) 
    epsilon = -dh_dz / (fmse_u.values - fmse_e.values)  
    
    return epsilon.astype(np.float32)


def compute_track_entrainment(ds, dstracks_wam, model, region, radius=50, surface='all'): 
    if surface == 'all': 
        zarr_path     = models.data_dir(model) / f'mcs_entr_rate_{region}_{radius}km.zarr'
    else: 
        zarr_path = models.data_dir(model) / f'mcs_entr_rate_{region}_{surface}_{radius}km.zarr'

    n_tracks = dstracks_wam.sizes['tracks']
    n_pressures = ds.sizes['pressure']

    ## logic from calc_mcs_entrainment.py in proxy_entrainment/

    max_label  = int(dstracks_wam.tracks.values.max()) + 1  # mask value = track_idx + 1

    # Map mask track number → output row index
    track_nums = dstracks_wam.tracks.values.astype(int)   # original track indices from full dstracks (sparse after WAM filter)
    mask_num_to_out_idx = np.full(max_label + 1, -1, dtype=np.int32)
    for out_i, tn in enumerate(track_nums):
        # MM: Where does the +1 come from?
        mask_num_to_out_idx[tn + 1] = out_i    # mask value = track_idx + 1

    times_3h = ds.time.values
    start_times = dstracks_wam.start_basetime.values   # (n_tracks,)
    first_3h_step = np.searchsorted(times_3h, start_times)



    entr_rate_out = np.full((n_tracks, MAX_TIMES_3H, n_pressures), np.nan, dtype=np.float32)
    base_time_out = np.full((n_tracks, MAX_TIMES_3H),
                            np.datetime64('NaT', 'ns'), dtype='datetime64[ns]')
    n_steps = len(times_3h)
    for step, t in enumerate(times_3h):
        if step % 100 == 0:
            print(f'  {step}/{n_steps}  ({pd.Timestamp(t)})', flush=True)

        ds_t = ds.isel(time=step)

        track_ids     = ds_t.track_id.values
        active_tracks = np.unique(track_ids[~np.isnan(track_ids)]).astype(int)

        if len(active_tracks) == 0:
            continue 
        
        for track in active_tracks: 
            track_mask = (ds_t.track_id == track).values
            if not track_mask.any():
                continue

            ds_track = ds_t.isel(cell=track_mask)

            entr_rate = compute_entr_rate(ds_track)
            if entr_rate is None:
                continue

            mask_num = track  # track_id stores mask number directly
            out_i = mask_num_to_out_idx[mask_num] if mask_num <= len(mask_num_to_out_idx) - 1 else -1
            if out_i < 0:
                continue

            li = step - first_3h_step[out_i]
            if li < 0 or li >= MAX_TIMES_3H:
                continue

            entr_rate_out[out_i, li, :] = entr_rate
            base_time_out[out_i, li]    = t


    ds_out = xr.Dataset({
        'entrainment_rate': xr.DataArray(entr_rate_out, dims=['tracks', 'times_3h', 'pressure']),
        'base_time':        xr.DataArray(base_time_out, dims=['tracks', 'times_3h'])
    })

    ds_out.to_zarr(zarr_path, region={'tracks': slice(0, n_tracks), 
                                       'times_3h': slice(0, MAX_TIMES_3H)})

    print('Done')




def main():

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--init', action='store_true',
                       help='Create empty zarr store (run once)')
    group.add_argument('--run', action='store_true',
                       help='Compute the entrainment rates')
    parser.add_argument('--n-timesteps', type=int, default=None, metavar='N',
                        help='Limit timesteps processed (for testing)')
    parser.add_argument('--surface', choices=['all', 'land', 'ocean'], default='all',
                        help='Filter MCS by mean land fraction: land (>0.8), ocean (<0.2), all (default)')
    parser.add_argument('--radius', type=int, default=50, metavar='N', 
                        help='radius in km to select from')
    models.add_model_arg(parser)
    models.add_region_arg(parser)
    args = parser.parse_args()

    global STATS_URL 
    STATS_URL = models.stats_url(args.model)

    dstracks     = load_track_stats(STATS_URL)
    region_cfg   = models.REGIONS[args.region]
    dstracks_wam = filter_region_tracks(dstracks, region_cfg)
    dstracks_wam = filter_surface(dstracks_wam, args.surface)

    if args.init:
        init_zarr(args.model, args.region, dstracks_wam, args.radius, args.surface)


    if args.run: 
        print("Opening and computing zarr ....")
        ds = xr.open_zarr(models.data_dir(args.model) /\
                 f'mcs_env_updraft_fmse_{args.radius}km.zarr').compute()
        print("Computing track entrainment")
        compute_track_entrainment(ds, dstracks_wam, args.model, 
                                  args.region, args.radius, args.surface)


if __name__ == '__main__':
    main()
