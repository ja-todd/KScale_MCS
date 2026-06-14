""" 
Compute per-track frozen MSE statistics for region-filtered MCS tracks

Links fmse_<region>.zarr (3-hourly) with the PyFLEXTRKR MCS pixel mask (hourly) to produce
a NetCDF with dims (tracks, times_3h) following PyFLEXTRKR output conventions.


Code logic: 

Output updraft frozen moist static energy and per-updraft mean frozen moist static energy for the environment

Usage: 

    python submit.py --model <model_id> --script calc_mcs_env_updraft_fmse 

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
import src.hp_models as models 
from src.hp_utils import haversine, open_region_dataset, align_times,\
                     compute_wam_positions
                    
import pandas as pd


### filter annoying warnings 
warnings.filterwarnings('ignore', message='.*The return type of `Dataset.dims`.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Relative humidity >120%.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*divide by zero encountered in log.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*invalid value encountered in divide.*', category=RuntimeWarning)

CHUNK_SIZE = 10 

ZOOM             = None
MASK_URL         = None

# ---------------------------------------------------------------------------
# Build initial zarr store 
# ---------------------------------------------------------------------------


def init_zarr(model, region, radius): 
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
            'z_updraft':   xr.DataArray(template_data, dims=['time', 'pressure', 'cell'], attrs={'units': 'm'}), 
            'rho_updraft' : xr.DataArray(template_data, dims=['time', 'pressure', 'cell'], attrs = {'units': 'kg m-3'}),
            'w_updraft' : xr.DataArray(template_data, dims=['time', 'pressure', 'cell'], attrs = {'units': 'm s-1'}),
            'track_id': xr.DataArray(dsa.full((n_times, n_cells), np.nan, dtype=np.float32, 
                 chunks=(CHUNK_SIZE, n_cells)),
        dims=['time', 'cell'],
        attrs={'description': 'MCS track number at each cell'})


        }, 
        coords={'time': ds.time, 'pressure': ds.pressure.sortby('pressure', ascending=False), 'cell': ds.cell, 'lat': ds.lat, 'lon': ds.lon},
        
    )

    zarr_path = models.data_dir(model) / f'mcs_env_updraft_fmse_{radius}km.zarr'
    zarr_path.parent.mkdir(parents=True, exist_ok=True) 
    template.to_zarr(zarr_path, mode='w', zarr_format=2)

    done_dir = models.done_dir(model)
    done_dir.mkdir(parents=True, exist_ok=True)
    models.init_donefile(model, region, tag=f'mcs_env_updraft_fmse_{radius}km').touch()
    print(f'Created {zarr_path}  shape=({n_times}, {n_pressures}, {n_cells})')
    print(f'Submit array 0-{n_chunks - 1}  ({n_chunks} jobs)  via: python submit.py --model {model}')

def get_env_fmse(fmse_t, updraft_lats, updraft_lons, all_lats, all_lons, 
                 updraft_bool, radius):
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


def compute_chunk(full_ds, fmse_ds, mask_ds, chunk_idx, model, region, radius, n_timesteps=None): 
    done_file = models.chunk_donefile(model, chunk_idx, tag=f'mcs_env_updraft_fmse_{radius}km')
    if done_file.exists():
        print(f'Chunk {chunk_idx} already done, skipping.')
        return
    zarr_path     = models.data_dir(model) / f'mcs_env_updraft_fmse_{radius}km.zarr' 
    
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
    z_updraft_out   = np.full_like(fmse_chunk, np.nan, dtype=np.float32)
    rho_updraft_out = np.full_like(fmse_chunk, np.nan, dtype=np.float32)
    w_updraft_out = np.full_like(fmse_chunk, np.nan, dtype=np.float32)
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
            continue  # leave as NaN because no MCS here
        
        w_t = w_chunk.isel(time=i).compute().values
        qc_t = qc_chunk.isel(time=i).compute().values

        w_mcs = w_t[:, mcs_bool]
        qc_mcs = qc_t[:, mcs_bool]

        w_mask = w_mcs > 1
        qc_mask = qc_mcs > 1e-5
        updraft_mask = w_mask & qc_mask


        cell_updraft = updraft_mask.any(axis=0)
        cell_indices = np.where(cell_updraft)[0]

        original_cell_indices = np.where(mcs_bool)[0]
        updraft_original_indices = original_cell_indices[cell_indices]

        fmse_t = fmse_ds.fmse.isel(time=fi).compute().values 
        z_t = fmse_ds.z.isel(time=fi).compute().values
        rho_t = fmse_ds.rho.isel(time=fi).compute().values


        fmse_updrafts = fmse_t[:, updraft_original_indices]   
        z_updrafts = z_t[:, updraft_original_indices] 
        rho_updrafts = rho_t[:, updraft_original_indices]
        w_updrafts = w_t[:, updraft_original_indices]
        updraft_lats = fmse_ds.lat.isel(cell=updraft_original_indices).values
        updraft_lons = fmse_ds.lon.isel(cell=updraft_original_indices).values


        updraft_bool = np.zeros(n_cells, dtype=bool)
        updraft_bool[updraft_original_indices] = True

        fmse_env_per_updraft = get_env_fmse(fmse_t, updraft_lats, updraft_lons, 
                                fmse_ds.lat.values, fmse_ds.lon.values, updraft_bool, radius)


        
        fmse_updraft_out[i, :, updraft_original_indices]  = fmse_updrafts.T
        fmse_env_out[i, :, updraft_original_indices]      = fmse_env_per_updraft.T
        rho_updraft_out[i, :, updraft_original_indices]   = rho_updrafts.T
        z_updraft_out[i, :, updraft_original_indices]     = z_updrafts.T
        w_updraft_out[i, :, updraft_original_indices]     = w_updrafts.T
        track_id_out[i, updraft_original_indices]         = mask_wam[updraft_original_indices]

    ds_out = xr.Dataset({
        'fmse_env': xr.DataArray(fmse_env_out,          dims=['time', 'pressure', 'cell']),
        'fmse_updraft': xr.DataArray(fmse_updraft_out,          dims=['time', 'pressure', 'cell']),  
        'z_updraft': xr.DataArray(z_updraft_out,          dims=['time', 'pressure', 'cell']),
        'rho_updraft': xr.DataArray(rho_updraft_out,          dims=['time', 'pressure', 'cell']),
        'w_updraft': xr.DataArray(w_updraft_out,          dims=['time', 'pressure', 'cell']),
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
        radius     = task_cfg['radius']
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
                      model, region, radius)  
        return

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--init', action='store_true',
                       help='Create empty zarr store (run once, or let submit.py handle this)')
    parser.add_argument('--n-timesteps', type=int, default=None, metavar='N',
                        help='Limit timesteps processed (for testing)')
    parser.add_argument('--radius', type=int, default=None,
                    help='Environment radius in km (only for calc_env_updraft_fmse)')
    models.add_model_arg(parser)
    models.add_region_arg(parser)
    args = parser.parse_args()

    if args.init:
        init_zarr(args.model, args.region, args.radius)



if __name__ == '__main__':
    main()

