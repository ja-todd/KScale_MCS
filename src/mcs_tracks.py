""" 
Utils for MCS track output from PyFLEXTRKR

""" 

import argparse
import json
import numpy as np 
import xarray as xr 
import dask.array as dsa 
from multiprocessing import Pool 
import warnings
import sys
import intake 
from pathlib import Path 
import easygems.healpix as egh 
import src.models as models 
import microphysics as micro
from metpy.calc import dewpoint_from_relative_humidity, virtual_temperature_from_dewpoint
from metpy.units import units 
import pandas as pd


### filter annoying warnings 
warnings.filterwarnings('ignore', message='.*The return type of `Dataset.dims`.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Relative humidity >120%.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*divide by zero encountered in log.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*invalid value encountered in divide.*', category=RuntimeWarning)


MAX_TIMES_3H = 217

ZOOM             = None
MASK_URL         = None
STATS_URL        = None
VAR_ZARR         = None


#----------------------------------------------------------------------
# Data loading 
#----------------------------------------------------------------------

def load_track_stats():
    """Fetch PyFLEXTRKR track statistics from on disc."""
    print('Loading MCS track statistics...')

    ## commented out is the process if using S3 URL
    # response = requests.get(STATS_URL, stream=True)
    # dstracks = xr.open_dataset(BytesIO(response.content), mask_and_scale=True)  
    dstracks = xr.open_dataset(STATS_URL, mask_and_scale=True)  ## used if the data on disk

    # Round times to nearest second (small offset in source data)
    def _round(t):
        return (np.round(t.astype(int) / 1e9) * 1e9).astype('datetime64[ns]')

    for field in ['base_time', 'start_basetime', 'end_basetime']:
        dstracks[field].load()
        tmask = ~np.isnan(dstracks[field].values)
        dstracks[field].values[tmask] = _round(dstracks[field].values[tmask])

    return dstracks


def open_var():
    """Open the local variable zarr store."""
    print('Opening variable zarr...')
    return xr.open_zarr(VAR_ZARR)


def open_mcs_mask():
    """Open the MCS pixel mask zarr from storage on disc."""
    print('Opening MCS mask zarr...')
    return xr.open_zarr(MASK_URL, chunks={})



# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def compute_wam_positions(var_ds, mask_ds):
    """
    Compute positional indices of WAM cells within the global mask cell array.
    Returns an int array of shape (n_wam_cells,).
    """
    wam_cells    = var_ds.cell.values             # HEALPix cell numbers, WAM subset
    global_cells = mask_ds.healpix_index.values    # HEALPix cell numbers, global (0…N-1)

    # global_cells is 0,1,2,...,N-1 so positions == wam_cells, but use searchsorted
    # for correctness in case of non-contiguous ranges.
    # MM: explain searchsorted.
    positions = np.searchsorted(global_cells, wam_cells)
    assert np.all(global_cells[positions] == wam_cells), \
        'WAM cell indices not found in global mask — zoom mismatch?'
    return positions


def filter_region_tracks(dstracks, region_cfg):
    """
    Keep only tracks whose centroid enters the analysis region (with buffer)
    at any point during their lifetime.
    """
    display = region_cfg['display']
    print(f'Filtering tracks to {display} region...')
    dstracks.meanlat.load()
    dstracks.meanlon.load()

    lat   = dstracks.meanlat.values        # (tracks, times)
    lon   = dstracks.meanlon.values        # (tracks, times), in [0, 360]
    lon180 = (lon + 180) % 360 - 180      # convert to [-180, 180]

    in_lat = (lat   > region_cfg['buf_lat_min']) & (lat   < region_cfg['buf_lat_max'])
    in_lon = (lon180 > region_cfg['buf_lon_min']) & (lon180 < region_cfg['buf_lon_max'])
    in_region = (in_lat & in_lon).any(axis=1)

    filtered = dstracks.isel(tracks=in_region)
    print(f'  {int(in_region.sum())} / {dstracks.sizes["tracks"]} tracks pass {display} filter')
    return filtered

LAND_FRAC_THRESHOLD  = 0.8   # mean pf_landfrac above this → land MCS
OCEAN_FRAC_THRESHOLD = 0.2   # mean pf_landfrac below this → ocean MCS


def filter_surface(dstracks, surface):
    """
    Filter tracks by mean land fraction (pf_landfrac, expressed as 0–1).
      'land'  : mean pf_landfrac > 0.8
      'ocean' : mean pf_landfrac < 0.2
      'all'   : no filter (default)
    """
    if surface == 'all':
        return dstracks

    print(f'Filtering tracks by surface type: {surface}...')
    dstracks.pf_landfrac.load()
    mean_lf = np.nanmean(dstracks.pf_landfrac.values, axis=1)  # (tracks,)

    if surface == 'land':
        mask = mean_lf > LAND_FRAC_THRESHOLD
    else:  # ocean
        mask = mean_lf < OCEAN_FRAC_THRESHOLD

    filtered = dstracks.isel(tracks=mask)
    print(f'  {int(mask.sum())} / {dstracks.sizes["tracks"]} tracks pass {surface} filter')
    return filtered



def align_times(entr_ds, mask_ds):
    """
    Find 3-hourly entrainment times that exist in the hourly mask dataset.
    Returns:
        times_3h   : 1-D array of datetime64 values (3-hourly, in overlap)
        mask_indices: corresponding positional indices in mask_ds.time
    """
    var_times = entr_ds.time.values
    mask_times = mask_ds.time.values

    mask_time_to_idx = {pd.Timestamp(t): i for i, t in enumerate(mask_times)}

    overlap_entr_idx = []
    overlap_mask_idx = []
    for i, t in enumerate(var_times):
        ts = pd.Timestamp(t)
        if ts in mask_time_to_idx:
            overlap_entr_idx.append(i)
            overlap_mask_idx.append(mask_time_to_idx[ts])

    print(f'Overlap: {len(overlap_entr_idx)} 3-hourly timesteps '
          f'({pd.Timestamp(var_times[overlap_entr_idx[0]])} – '
          f'{pd.Timestamp(var_times[overlap_entr_idx[-1]])})')

    return (np.array(overlap_entr_idx),
            np.array(overlap_mask_idx),
            var_times[overlap_entr_idx])




def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    models.add_model_arg(parser)
    models.add_region_arg(parser)
    parser.add_argument('--output', default=None,
                        help='Output NetCDF path (default: data/<model>/mcs_entrainment_<region>[_<surface>].nc)')
    parser.add_argument('--n-timesteps', type=int, default=None, metavar='N',
                        help='Limit to first N timesteps (for testing)')
    parser.add_argument('--surface', choices=['all', 'land', 'ocean'], default='all',
                        help='Filter MCS by mean land fraction: land (>0.8), ocean (<0.2), all (default)')
    args = parser.parse_args()

    region_cfg = models.REGIONS[args.region]

    if args.output is None:
        suffix = f'_{args.surface}' if args.surface != 'all' else ''
        args.output = str(
            models.data_dir(args.model) / f'mcs_{var}_{args.region}{suffix}.nc'
        )

    # Patch module-level URL/path constants to match the chosen model/region.
    global VAR_ZARR, MASK_URL, STATS_URL, ZOOM
    ZOOM             = models.MODELS[args.model]['zoom']
    MASK_URL         = models.mask_url(args.model)
    STATS_URL        = models.stats_url(args.model)
    VAR_ZARR = models.data_dir(args.model) / f'{var}_{args.region}.zarr'

    dstracks         = load_track_stats()
    var_ds           = open_var()
    mask_ds          = open_mcs_mask()

    wam_positions    = compute_wam_positions(var_ds, mask_ds)
    dstracks_wam     = filter_region_tracks(dstracks, region_cfg)
    dstracks_wam     = filter_surface(dstracks_wam, args.surface)

    var_idxs, mask_idxs, times_3h = align_times(var_ds, mask_ds)

    if args.n_timesteps is not None:
        var_idxs = var_idxs[:args.n_timesteps]
        mask_idxs = mask_idxs[:args.n_timesteps]
        times_3h  = times_3h[:args.n_timesteps]

    print(f'Processing {len(var_idxs)} timesteps for '
          f'{dstracks_wam.sizes["tracks"]} WAM tracks...')



    arrays, base_time_out = compute_track_entrainment(
        var_ds, mask_ds, dstracks_wam,
        wam_positions, var_idxs, mask_idxs, times_3h
    )

    ds_out = build_output_dataset(arrays, base_time_out, dstracks_wam, times_3h)
    save_output(ds_out, args.output)


if __name__ == '__main__':
    main()
