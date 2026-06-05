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


### filter annoying warnings 
warnings.filterwarnings('ignore', message='.*The return type of `Dataset.dims`.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Relative humidity >120%.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*divide by zero encountered in log.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*invalid value encountered in divide.*', category=RuntimeWarning)


MAX_TIMES_3H = 217

VARS = 'fmse'

ZOOM             = None
MASK_URL         = None
STATS_URL        = None
ENTRAINMENT_ZARR = None


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


def open_entrainment():
    """Open the local entrainment zarr store."""
    print('Opening entrainment zarr...')
    return xr.open_zarr(ENTRAINMENT_ZARR)


def open_mcs_mask():
    """Open the MCS pixel mask zarr from storage on disc."""
    print('Opening MCS mask zarr...')
    return xr.open_zarr(MASK_URL, chunks={})



# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def compute_wam_positions(entr_ds, mask_ds):
    """
    Compute positional indices of WAM cells within the global mask cell array.
    Returns an int array of shape (n_wam_cells,).
    """
    wam_cells    = entr_ds.cell.values             # HEALPix cell numbers, WAM subset
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