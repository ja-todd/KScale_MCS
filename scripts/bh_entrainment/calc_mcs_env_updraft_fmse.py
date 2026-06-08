""" 
Compute per-track frozen MSE statistics for region-filtered MCS tracks

Links fmse_<region>.zarr (3-hourly) with the PyFLEXTRKR MCS pixel mask (hourly) to produce
a NetCDF with dims (tracks, times_3h) following PyFLEXTRKR output conventions.


Code logic: 

Create a frozen MSE dataset filtered on the MCS locations, with track_id in the output zarr

Usage: 

    python submit.py --model <model_id> --script calc_mcs_fmse 

""" 
""" 
Utils for MCS track output from PyFLEXTRKR

""" 

import argparse
import json
import numpy as np 
import xarray as xr 
import dask.array as dsa 
import warnings
import sys
import intake 
from pathlib import Path 
import easygems.healpix as egh 
import src.models as models 
from src.utils import haversine, open_region_dataset
import pandas as pd


### filter annoying warnings 
warnings.filterwarnings('ignore', message='.*The return type of `Dataset.dims`.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Relative humidity >120%.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*divide by zero encountered in log.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*invalid value encountered in divide.*', category=RuntimeWarning)

CHUNK_SIZE = 10 
RADIUS = None
MAX_TIMES_3H = 217

ZOOM             = None
MASK_URL         = None
STATS_URL        = None
FMSE_ZARR         = None


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


def open_fmse():
    """Open the local variable zarr store."""
    print('Opening variable zarr...')
    return xr.open_zarr(FMSE_ZARR)


def open_mcs_mask():
    """Open the MCS pixel mask zarr from storage on disc."""
    print('Opening MCS mask zarr...')
    return xr.open_zarr(MASK_URL, chunks={})

def hp_mods(ds):
    return ds.rename({'healpix_index': 'cell'}).pipe(egh.attach_coords)


def _region_mask(ds, region_cfg):
    """Boolean cell mask for the analysis region (handles lon wrap-around)."""
    lon_min, lon_max = region_cfg['lon_min'], region_cfg['lon_max']
    lat_min, lat_max = region_cfg['lat_min'], region_cfg['lat_max']
    if lon_min > lon_max:   # region wraps across 0°/360°
        lon_mask = (ds.lon > lon_min) | (ds.lon < lon_max)
    else:
        lon_mask = (ds.lon > lon_min) & (ds.lon < lon_max)
    return lon_mask & (ds.lat > lat_min) & (ds.lat < lat_max)


def open_region_dataset(model, region_cfg):
    zoom = models.MODELS[model]['zoom']
    cat  = intake.open_catalog(models.CATALOG_URL)['UK']
    ds3h = cat[model](zoom=zoom, time='PT3H').to_dask().pipe(hp_mods)
    return ds3h.isel(cell=_region_mask(ds3h, region_cfg))


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



def align_times(fmse_ds, mask_ds):
    """
    Find 3-hourly entrainment times that exist in the hourly mask dataset.
    Returns:
        times_3h   : 1-D array of datetime64 values (3-hourly, in overlap)
        mask_indices: corresponding positional indices in mask_ds.time
    """
    var_times = fmse_ds.time.values
    mask_times = mask_ds.time.values

    mask_time_to_idx = {pd.Timestamp(t): i for i, t in enumerate(mask_times)}

    overlap_fmse = []
    overlap_mask_idx = []
    for i, t in enumerate(var_times):
        ts = pd.Timestamp(t)
        if ts in mask_time_to_idx:
            overlap_fmse.append(i)
            overlap_mask_idx.append(mask_time_to_idx[ts])

    print(f'Overlap: {len(overlap_fmse)} 3-hourly timesteps '
          f'({pd.Timestamp(var_times[overlap_fmse[0]])} – '
          f'{pd.Timestamp(var_times[overlap_fmse[-1]])})')

    return (np.array(overlap_fmse),
            np.array(overlap_mask_idx),
            var_times[overlap_fmse])


def init_zarr(model, region): 
    region_cfg = models.REGIONS[region]
    ds = open_region_dataset(model, region_cfg)

    n_times  = ds.sizes['time']
    n_pressures = ds.sizes['pressure']
    n_cells  = ds.sizes['cell']
    n_chunks = (n_times + CHUNK_SIZE - 1) // CHUNK_SIZE

    template_data    = dsa.full((n_times, n_pressures, n_cells), np.nan, dtype=np.float32,
                        chunks=(CHUNK_SIZE, n_pressures, n_cells))

    template = xr.Dataset(
        { 
            'fmse_env': xr.DataArray(template_data, dims=['time', 'pressure', 'cell'], attrs = {'units': 'J kg-1'}),
            'fmse_updraft': xr.DataArray(template_data, dims=['time', 'pressure', 'cell'], attrs = {'units': 'J kg-1'}),  
            'track_id': xr.DataArray(dsa.full((n_times, n_cells), np.nan, dtype=np.float32, 
                 chunks=(CHUNK_SIZE, n_cells)),
        dims=['time', 'cell'],
        attrs={'description': 'MCS track number at each cell'})


        }, 
        coords={'time': ds.time, 'pressure': ds.pressure.sortby('pressure', ascending=False), 'cell': ds.cell, 'lat': ds.lat, 'lon': ds.lon},
        
    )

    zarr_path = models.data_dir(model) / f'mcs_env_updraft_fmse_{region}.zarr'
    zarr_path.parent.mkdir(parents=True, exist_ok=True) 
    template.to_zarr(zarr_path, mode='w', zarr_format=2)

    done_dir = models.done_dir(model)
    done_dir.mkdir(parents=True, exist_ok=True)
    models.init_donefile(model, region, tag='mcs_env_updraft_fmse').touch()
    print(f'Created {zarr_path}  shape=({n_times}, {n_pressures}, {n_cells})')
    print(f'Submit array 0-{n_chunks - 1}  ({n_chunks} jobs)  via: python submit.py --model {model}')

### effect of changing the radius - 10 km might 


# def get_env_fmse(fmse_t, updraft_lats, updraft_lons, all_lats, all_lons, updraft_bool, radius=10):
#     # distances from ALL updraft cells to ALL WAM cells - shape (n_updrafts, n_cells)
#     env_mask = dsa.zeros(len(all_lats), dtype=bool)
#     for lat, lon in zip(updraft_lats, updraft_lons):
#         dist = haversine(lat, lon, all_lats, all_lons)
#         env_mask |= (dist < radius) & (~updraft_bool)
    
#     return fmse_t[:, env_mask.compute()]

def get_env_fmse(fmse_t, updraft_lats, updraft_lons, all_lats, all_lons, 
                 updraft_bool, radius=10):
    """
    For each updraft cell, compute mean environment FMSE within radius.
    Returns fmse_env_per_updraft of shape (pressure, n_updrafts)
    """
    n_updrafts = len(updraft_lats)
    n_pressure = fmse_t.shape[0]
    fmse_env_per_updraft = np.full((n_pressure, n_updrafts), np.nan, dtype=np.float32)
    
    for idx, (lat, lon) in enumerate(zip(updraft_lats, updraft_lons)):
        dist = haversine(lat, lon, all_lats, all_lons)
        local_env_mask = (dist < radius) & (~updraft_bool)
        if local_env_mask.any():
            fmse_env_per_updraft[:, idx] = fmse_t[:, local_env_mask].mean(axis=1)
    
    return fmse_env_per_updraft  # (pressure, n_updrafts)


def compute_chunk(full_ds, fmse_ds, mask_ds, chunk_idx, model, region, n_timesteps=None): 
    done_file = models.chunk_donefile(model, chunk_idx, tag='mcs_env_updraft_fmse')
    if done_file.exists():
        print(f'Chunk {chunk_idx} already done, skipping.')
        return
    zarr_path     = models.data_dir(model) / f'mcs_env_updraft_fmse_{region}.zarr' 
    
    wam_positions = compute_wam_positions(fmse_ds, mask_ds)
    fmse_idxs, mask_idxs, _ = align_times(fmse_ds, mask_ds)

    t_start  = chunk_idx * CHUNK_SIZE
    t_end    = min(t_start + CHUNK_SIZE, fmse_ds.sizes['time'])
    if n_timesteps is not None:
        t_end = min(t_start + n_timesteps, t_end)
    n_chunk  = t_end - t_start

    print(f'Chunk {chunk_idx}: time[{t_start}:{t_end}] ({n_chunk} timesteps)')
    fmse_chunk  = fmse_ds.isel(time=slice(t_start, t_end))['fmse'].compute().values
    n_cells = fmse_chunk.shape[2]


    fmse_updraft_out = np.full_like(fmse_chunk, np.nan, dtype=np.float32)
    fmse_env_out = np.full_like(fmse_chunk, np.nan, dtype=np.float32)
    track_id_out = np.full((n_chunk, n_cells), np.nan, dtype=np.float32)

    in_chunk        = (fmse_idxs >= t_start) & (fmse_idxs < t_end)
    fmse_idxs_chunk = fmse_idxs[in_chunk]
    mask_idxs_chunk = mask_idxs[in_chunk]

    w_chunk = full_ds.wa.isel(time=slice(t_start, t_end))
    ql_chunk = full_ds.clw.isel(time=slice(t_start, t_end))
    qi_chunk = full_ds.cli.isel(time=slice(t_start, t_end))
    qc_chunk = ql_chunk + qi_chunk 

    for fi, mi in zip(fmse_idxs_chunk, mask_idxs_chunk):
        i = fi - t_start

        mask_global = mask_ds.mcs_mask.isel(time=mi).compute().values
        mask_global = np.nan_to_num(mask_global, nan=0.0)
        mask_wam    = mask_global[wam_positions].astype(np.int32)

        mcs_bool = mask_wam > 0
        if not mcs_bool.any():
            continue  # no MCS at this timestep, leave as NaN
        
        w_t = w_chunk.isel(time=i).compute().values
        qc_t = qc_chunk.isel(time=i).compute().values

        w_mcs = w_t[:, mcs_bool]
        qc_mcs = qc_t[:, mcs_bool]

        w_mask = w_mcs > 1
        qc_mask = qc_mcs > 1e-5
        updraft_mask = w_mask & qc_mask


        cell_updraft = updraft_mask.any('pressure')
        cell_indices = np.where(cell_updraft.values)[0]

        original_cell_indices = np.where(mcs_bool)[0]
        updraft_original_indices = original_cell_indices[cell_indices]

        fmse_t = fmse_ds.fmse.isel(time=fi).compute().values  # (pressure, cell)
        fmse_updrafts = fmse_t[:, updraft_original_indices]    # (pressure, n_updrafts)
        updraft_lats = fmse_ds.lat.isel(cell=updraft_original_indices).values
        updraft_lons = fmse_ds.lon.isel(cell=updraft_original_indices).values


        updraft_bool = np.zeros(n_cells, dtype=bool)
        updraft_bool[updraft_original_indices] = True

        fmse_env_per_updraft = get_env_fmse(fmse_t, updraft_lats, updraft_lons, 
                                fmse_ds.lat.values, fmse_ds.lon.values, updraft_bool)


        ### write to output properly - unsure how 
        fmse_updraft_out[i, :, updraft_original_indices]  = fmse_t[:, updraft_original_indices]
        fmse_env_out[i, :, updraft_original_indices]      = fmse_env_per_updraft
        track_id_out[i, updraft_original_indices]         = mask_wam[updraft_original_indices]

    ds_out = xr.Dataset({
        'fmse_env': xr.DataArray(fmse_env_out,          dims=['time', 'pressure', 'cell']),
        'fmse_updraft': xr.DataArray(fmse_updraft_out,          dims=['time', 'pressure', 'cell']),  
        'track_id': xr.DataArray(track_id_out, dims=['time', 'cell'])
    })

    ds_out.to_zarr(zarr_path, region={'time': slice(t_start, t_end)})

    done_file.touch()
    print(f'Chunk {chunk_idx} written')


def main():
    # Two modes:
    #   --init --model <model> [--region <region>]   — initialise zarr store
    #   <json_file> <task_index>                     — process one SLURM task

    if len(sys.argv) >= 3 and not sys.argv[1].startswith('-'):
        # SLURM task mode: positional json_file task_index
        json_file  = Path(sys.argv[1])
        task_index = int(sys.argv[2])

        if not json_file.exists():
            sys.exit(f'Error: task file {json_file} not found')
    
        task_cfg   = json.loads(json_file.read_text())
        
        if task_index >= len(task_cfg['tasks']):
            sys.exit(f'Error: task index {task_index} out of range')

        model      = task_cfg['model']
        region     = task_cfg['region']
        chunk      = task_cfg['tasks'][task_index]['chunk']
        region_cfg = models.REGIONS[region]

        global MASK_URL, ZOOM

        ZOOM = models.MODELS[model]['zoom']
        MASK_URL = models.mask_url(model)
        mask_ds = xr.open_zarr(MASK_URL, chunks={}, mask_and_scale=False)

        fmse_ds = xr.open_zarr(models.data_dir(model) / f'fmse_{region}.zarr')
        full_ds = open_region_dataset(model, region_cfg).\
            sortby('pressure', ascending=False)
        

        compute_chunk(full_ds, fmse_ds, mask_ds, chunk, 
                      model, region)  
        return

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--init', action='store_true',
                       help='Create empty zarr store (run once, or let submit.py handle this)')
    parser.add_argument('--n-timesteps', type=int, default=None, metavar='N',
                        help='Limit timesteps processed (for testing)')
    models.add_model_arg(parser)
    models.add_region_arg(parser)
    args = parser.parse_args()

    if args.init:
        init_zarr(args.model, args.region)



if __name__ == '__main__':
    main()

