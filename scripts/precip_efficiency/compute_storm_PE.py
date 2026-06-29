import xarray as xr 
import numpy as np 
import src.hp_models as models
from src.hp_utils import load_track_stats,\
      filter_region_tracks, filter_surface, MAX_TIMES_3H
import argparse
from pathlib import Path
import dask.array as dsa
import pandas as pd

VAR = 'precip_efficiency'


def init_zarr(model, region, all_track_ids): 
    done_file = models.init_donefile(model, region, tag='mcs_precip_efficiency')
    if done_file.exists():
        print('INIT Already done, skipping ...')
        return 
    
    n_tracks = len(all_track_ids)

    template = xr.Dataset({
        'precip_eff': xr.DataArray(
            dsa.full((n_tracks, MAX_TIMES_3H), np.nan, dtype=np.float32,
                     chunks=(n_tracks, MAX_TIMES_3H)),
            dims=['tracks', 'times_3h'],
            attrs={'units': 'dimensionless'}),
        'base_time': xr.DataArray(
            dsa.full((n_tracks, MAX_TIMES_3H), np.datetime64('NaT', 'ns'), dtype='datetime64[ns]',
                     chunks=(n_tracks, MAX_TIMES_3H)),
            dims=['tracks', 'times_3h'])
        },
        coords={'tracks': all_track_ids})

    
    zarr_path = models.data_dir(model, VAR) / f'mcs_precip_efficiency_{region}.zarr'
    
    

    zarr_path.parent.mkdir(parents=True, exist_ok=True) 
    template.to_zarr(zarr_path, mode='w', zarr_format=2)

    done_dir = models.done_dir(model)
    done_dir.mkdir(parents=True, exist_ok=True)
    done_file.touch()
    print(f'Created {zarr_path}  shape=({n_tracks}, {MAX_TIMES_3H})')
    

def compute_donefile(model, region):
    return models.done_dir(model) / f'mcs_precip_efficiency_computation_{region}.done'

def compute_track_PE(ds, model, region): 
    done_file = compute_donefile(model, region)
    if done_file.exists(): 
        print('COMPUTATION Already done, skipping ... ')
        return 
    
    zarr_path = models.data_dir(model, VAR) / f'mcs_precip_efficiency_{region}.zarr'
    

    times_3h = ds.time.values

    # build all_track_ids and first_3h_step lazily
    all_track_ids_set  = set()
    first_3h_step_dict = {}

    for t in range(len(times_3h)):
        track_ids_t = ds.track_id.isel(time=t).values
        where_not_nan = track_ids_t[~np.isnan(track_ids_t)]
        active = np.unique(where_not_nan).astype(int)
        for tid in active:
            all_track_ids_set.add(tid)
            if tid not in first_3h_step_dict:
                first_3h_step_dict[tid] = t

    all_track_ids = np.array(sorted(all_track_ids_set))
    n_tracks      = len(all_track_ids)
    max_label     = int(all_track_ids.max()) + 1
    mask_num_to_out_idx = np.full(max_label + 1, -1, dtype=np.int32)
    for out_i, tid in enumerate(all_track_ids):
        mask_num_to_out_idx[tid] = out_i

    first_3h_step = np.array([first_3h_step_dict[tid] for tid in all_track_ids])

    tid = all_track_ids[0]
    out_i = mask_num_to_out_idx[tid]
    print(f"first_3h_step: {first_3h_step[out_i]}")
    print(f"corresponding time: {times_3h[first_3h_step[out_i]]}")

    # check when it actually first appears
    for t in range(len(times_3h)):
        track_ids_t = ds.track_id.isel(time=t).values
        if tid in track_ids_t:
            print(f"first actual appearance: step {t}, time {times_3h[t]}")
            break

    PE_out        = np.full((n_tracks, MAX_TIMES_3H), np.nan, dtype=np.float32)
    base_time_out = np.full((n_tracks, MAX_TIMES_3H),
                            np.datetime64('NaT', 'ns'), dtype='datetime64[ns]')

    n_steps = len(times_3h)
    for step, t in enumerate(times_3h):
        if step % 100 == 0:
            print(f'  {step}/{n_steps}  ({pd.Timestamp(t)})', flush=True)

        ds_t          = ds.isel(time=step)
        track_ids     = ds_t.track_id.values
        active_tracks = np.unique(track_ids[~np.isnan(track_ids)]).astype(int)

        if len(active_tracks) == 0:
            continue 
        
        for track in active_tracks: 
            track_mask = (ds_t.track_id == track).values
            if not track_mask.any():
                continue

            ds_track = ds_t.isel(cell=track_mask)

            cr_track = ds_track['condensation_rate'].compute().mean(dim='cell').values
            pr_track = ds_track['precip_flux'].compute().mean(dim='cell').values

            if cr_track == 0:
                continue

            PE = pr_track / cr_track

            mask_num = track
            out_i    = mask_num_to_out_idx[mask_num] if mask_num <= len(mask_num_to_out_idx) - 1 else -1
            if out_i < 0:
                continue

            if track == tid:
                li = step - first_3h_step[out_i]
                print(f"step={step}, li={li}, PE={PE}")

            li = step - first_3h_step[out_i]
            if li < 0 or li >= MAX_TIMES_3H:
                continue

            PE_out[out_i, li]     = PE
            base_time_out[out_i, li] = t

    ds_out = xr.Dataset({
        'precip_eff': xr.DataArray(PE_out,        dims=['tracks', 'times_3h']),
        'base_time':  xr.DataArray(base_time_out, dims=['tracks', 'times_3h'])
    })

    ds_out.to_zarr(zarr_path, region={'tracks': slice(0, n_tracks), 
                                      'times_3h': slice(0, MAX_TIMES_3H)})

    done_file.touch()
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
    models.add_model_arg(parser)
    models.add_region_arg(parser)
    args = parser.parse_args()

    global STATS_URL 
    STATS_URL = models.stats_url(args.model)

    # dstracks     = load_track_stats(STATS_URL)
    # region_cfg   = models.REGIONS[args.region]
    # dstracks_wam = filter_region_tracks(dstracks, region_cfg)
    
    zarr_path = models.data_dir(args.model, VAR) / f'mcs_condensation_rate_{args.region}.zarr'

    if args.init:
        init_done = models.init_donefile(args.model, args.region, )
        ds = xr.open_zarr(zarr_path)
        track_id_array = ds.track_id.values  # (time, cell) - load once
        all_track_ids  = np.unique(track_id_array[~np.isnan(track_id_array)]).astype(int)
        init_zarr(args.model, args.region, all_track_ids)
        

    if args.run:
        
        ds = xr.open_zarr(zarr_path).compute()
        print("Computing track PE")
        compute_track_PE(ds, args.model, 
                                args.region)
        

if __name__ == '__main__':
    main()
