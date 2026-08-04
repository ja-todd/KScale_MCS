import argparse
import numpy as np 
import xarray as xr 
import dask.array as dsa 
import warnings
import src.hp_models as models 
import pandas as pd
import time
import os
from pathlib import Path

from src.hp_utils import open_region_dataset, compute_wam_positions,\
            align_times, open_region_1h_dataset, load_track_stats, filter_region_tracks,\
                MAX_TIMES_3H
import sys


### filter annoying warnings 
warnings.filterwarnings('ignore', message='.*The return type of `Dataset.dims`.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Relative humidity >120%.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*divide by zero encountered in log.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*invalid value encountered in divide.*', category=RuntimeWarning)

CHUNK_SIZE = 10 

ZOOM             = None
MASK_URL         = None

VAR = 'precip_efficiency'

# ---------------------------------------------------------------------------
# Initialize zarr store 
# ---------------------------------------------------------------------------
def init_zarr(dstracks_wam, model, region):
    done_file = models.init_donefile(model, region, tag='mcs_condensation_rate_stats')
    if done_file.exists():
        print(f'INIT for {model} Already done, skipping ...')
        return

    n_tracks   = dstracks_wam.sizes['tracks']
    track_nums = dstracks_wam.tracks.values.astype(int)


    template = xr.Dataset({
        'cr_mean': xr.DataArray(
            dsa.full((n_tracks, MAX_TIMES_3H), np.nan, dtype=np.float32,
                     chunks=(100, MAX_TIMES_3H)),
            dims=['track', 'times_3h'],
            attrs={'units': 'kg m-2 s-1'}),
        'pr_mean': xr.DataArray(
            dsa.full((n_tracks, MAX_TIMES_3H), np.nan, dtype=np.float32,
                     chunks=(100, MAX_TIMES_3H)),
            dims=['track', 'times_3h'],
            attrs={'units': 'kg m-2 s-1'}),
        'pe_mean': xr.DataArray(
            dsa.full((n_tracks, MAX_TIMES_3H), np.nan, dtype=np.float32,
                     chunks=(100, MAX_TIMES_3H)),
            dims=['track', 'times_3h'],
            attrs={'units': 'dimensionless'}),
        'cr_sum': xr.DataArray(
            dsa.full((n_tracks, MAX_TIMES_3H), np.nan, dtype=np.float32,
                      chunks=(100, MAX_TIMES_3H)),
            dims=['track', 'times_3h'],
            attrs={'units': 'kg m-2 s-1'}),
        'pr_sum': xr.DataArray(
            dsa.zeros((n_tracks, MAX_TIMES_3H), np.nan, dtype=np.float32,
                      chunks=(100, MAX_TIMES_3H)),
            dims=['track', 'times_3h'],
            attrs={'units': 'kg m-2 s-1'}),
        'base_time': xr.DataArray(
            dsa.full((n_tracks, MAX_TIMES_3H), np.datetime64('NaT', 'ns'),
                     dtype='datetime64[ns]', chunks=(100, MAX_TIMES_3H)),
            dims=['track', 'times_3h']),
    },
    coords={
        'track':        track_nums
    })

    zarr_path = models.data_dir(model, VAR) / f'mcs_condensation_rate_{region}_stats.zarr'
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_zarr(zarr_path, mode='w', zarr_format=2, write_empty_chunks=False)

    done_dir = models.done_dir(model)
    done_dir.mkdir(parents=True, exist_ok=True)
    models.init_donefile(model, region, tag='mcs_condensation_rate_stats').touch()
    print(f'Created {zarr_path}  shape=({n_tracks}, {MAX_TIMES_3H})')
## don't use chunk approach (fine as input data is 2D anyway )
## need to modify zarr 
## need to modify main and submit.py

def compute_donefile(model, region):
    return models.done_dir(model) / f'mcs_condensation_rates_{region}_stats.done'

def compute_track_condensation_rates(cr_ds, precip_ds, mask_ds, dstracks_wam, wam_positions,
                                      cr_idxs, mask_idxs, times_3h, model, region, n_timesteps=None):
    done_file = compute_donefile(model, region)
    if done_file.exists():
        print(f'COMPUTATION already done for {model}, skipping ...')
        return

    zarr_path = models.data_dir(model, VAR) / f'mcs_condensation_rate_{region}_stats.zarr'

    track_nums = dstracks_wam.tracks.values.astype(int)
    max_label  = int(track_nums.max()) + 1
    mask_num_to_out_idx = np.full(max_label + 1, -1, dtype=np.int32)
    for out_i, tn in enumerate(track_nums):
        mask_num_to_out_idx[tn + 1] = out_i

    start_times   = dstracks_wam.start_basetime.values
    first_3h_step = np.searchsorted(times_3h, start_times)

    n_tracks = len(track_nums)

    n_steps = len(times_3h)


    cr_means       = np.full((n_tracks, MAX_TIMES_3H), np.nan, dtype=np.float32)
    pr_means       = np.full((n_tracks, MAX_TIMES_3H), np.nan, dtype=np.float32)
    pe_means       = np.full((n_tracks, MAX_TIMES_3H), np.nan, dtype=np.float32)
    cr_sums        = np.full((n_tracks, MAX_TIMES_3H), np.nan, dtype=np.float32)
    pr_sums        = np.full((n_tracks, MAX_TIMES_3H), np.nan, dtype=np.float32)
    base_time_out = np.full((n_tracks, MAX_TIMES_3H),
                             np.datetime64('NaT', 'ns'), dtype='datetime64[ns]')

    for step, (ci, mi, t) in enumerate(zip(cr_idxs, mask_idxs, times_3h)):
        if step % 100 == 0:
            print(f'  step {step}/{n_steps}', flush=True)

        mask_global = mask_ds.mcs_mask.isel(time=mi).compute().values
        mask_global = np.nan_to_num(mask_global, nan=0.0)
        mask_wam    = mask_global[wam_positions].astype(np.int32)

        cr_t = cr_ds['condensation_rate'].isel(time=ci).compute().values
        pr_t = precip_ds['pr'].isel(time=step).compute().values

        active_tracks = np.unique(mask_wam[mask_wam > 0])

        for mask_tr in active_tracks:
            out_i = mask_num_to_out_idx[mask_tr] if mask_tr <= len(mask_num_to_out_idx) - 1 else -1
            if out_i < 0:
                continue
            li = step - first_3h_step[out_i]
            if li < 0 or li >= MAX_TIMES_3H:
                continue

            mcs_bool = mask_wam == mask_tr
            if not mcs_bool.any():
                continue

            cr_mcs = cr_t[mcs_bool]
            pr_mcs = pr_t[mcs_bool]

            cr_summed = cr_mcs.sum() ## eliminates issue of having lots of zeros (no condensation)
            pr_summed = pr_mcs.sum()

            cr_mean = np.nanmean(cr_mcs)
            pr_mean = np.nanmean(pr_mcs)

            cr_means[out_i, li]      = cr_mean
            pr_means[out_i, li]      = pr_mean
            pe_means[out_i, li]      = pr_summed / cr_summed if cr_summed > 0 else np.nan
            cr_sums[out_i, li]       = cr_summed
            pr_sums[out_i, li]       = pr_summed
            base_time_out[out_i, li] = t

    ds_out = xr.Dataset({
        'cr_mean':   xr.DataArray(cr_means,      dims=['track', 'times_3h'], attrs={'units': 'kg m-2 s-1'}),
        'pr_mean':   xr.DataArray(pr_means,      dims=['track', 'times_3h'], attrs={'units': 'kg m-2 s-1'}),
        'pe_mean':   xr.DataArray(pe_means,      dims=['track', 'times_3h']),
        'cr_sum':   xr.DataArray(cr_sums,      dims=['track', 'times_3h'], attrs={'units': 'kg m-2 s-1'}),
        'pr_sum':   xr.DataArray(pr_sums,      dims=['track', 'times_3h'], attrs={'units': 'kg m-2 s-1'}),
        'base_time': xr.DataArray(base_time_out, dims=['track', 'times_3h']),
    },
    coords={
        'track':        track_nums
    })

    ds_out.to_zarr(zarr_path, mode = 'w')

    done_file.touch()
    print("Computation done")


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
    
    dstracks = load_track_stats(STATS_URL)
    mask_ds = xr.open_zarr(MASK_URL, chunks={})
    
    cr_ds     = xr.open_zarr(models.data_dir(args.model, VAR) / f'condensation_rate_{args.region}.zarr')
    precip_ds = open_region_1h_dataset(args.model, region_cfg)

    wam_positions = compute_wam_positions(cr_ds, mask_ds)
    dstracks_wam  = filter_region_tracks(dstracks, region_cfg)
    
    cr_idxs, mask_idxs, times_3h = align_times(cr_ds, mask_ds)
    precip_ds = precip_ds.sel(time=times_3h)

    print(f'Creating zarr with {len(cr_idxs)} timesteps for '
          f'{dstracks_wam.sizes["tracks"]} WAM tracks...')
    
    init_zarr(dstracks_wam, args.model, args.region)

    print(f'Processing {len(cr_idxs)} timesteps for '
          f'{dstracks_wam.sizes["tracks"]} WAM tracks...')
    

    compute_track_condensation_rates(cr_ds, precip_ds, mask_ds, dstracks_wam, 
                    wam_positions, cr_idxs, mask_idxs, times_3h, args.model, args.region)

    
if __name__ == '__main__':
    main()
