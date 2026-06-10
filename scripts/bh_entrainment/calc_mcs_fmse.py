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
from pathlib import Path 
import easygems.healpix as egh 
import src.models as models 
from src.utils import open_region_dataset, compute_wam_positions, align_times


### filter annoying warnings 
warnings.filterwarnings('ignore', message='.*The return type of `Dataset.dims`.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Relative humidity >120%.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*divide by zero encountered in log.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*invalid value encountered in divide.*', category=RuntimeWarning)

CHUNK_SIZE = 10 

ZOOM             = None
MASK_URL         = None


# ---------------------------------------------------------------------------
# Initialize zarr store 
# ---------------------------------------------------------------------------

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
            'fmse': xr.DataArray(template_data, dims=['time', 'pressure', 'cell'], attrs = {'units': 'J kg-1'}), 
            'track_id': xr.DataArray(dsa.full((n_times, n_cells), np.nan, dtype=np.float32, 
                 chunks=(CHUNK_SIZE, n_cells)),
        dims=['time', 'cell'],
        attrs={'description': 'MCS track number at each cell'})


        }, 
        coords={'time': ds.time, 'pressure': ds.pressure.sortby('pressure', ascending=False), 'cell': ds.cell, 'lat': ds.lat, 'lon': ds.lon},
        
    )

    zarr_path = models.data_dir(model) / f'mcs_fmse_{region}.zarr'
    zarr_path.parent.mkdir(parents=True, exist_ok=True) 
    template.to_zarr(zarr_path, mode='w', zarr_format=2)

    done_dir = models.done_dir(model)
    done_dir.mkdir(parents=True, exist_ok=True)
    models.init_donefile(model, region, tag='mcs_fmse').touch()
    print(f'Created {zarr_path}  shape=({n_times}, {n_pressures}, {n_cells})')
    print(f'Submit array 0-{n_chunks - 1}  ({n_chunks} jobs)  via: python submit.py --model {model}')


def compute_chunk(fmse_ds, mask_ds, chunk_idx, model, region, n_timesteps=None): 
    done_file = models.chunk_donefile(model, chunk_idx, tag='mcs_fmse')
    if done_file.exists():
        print(f'Chunk {chunk_idx} already done, skipping.')
        return
    zarr_path     = models.data_dir(model) / f'mcs_fmse_{region}.zarr' 
    
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


    mcs_fmse_out = np.full_like(fmse_chunk, np.nan, dtype=np.float32)
    track_id_out = np.full((n_chunk, n_cells), np.nan, dtype=np.float32)

    in_chunk        = (fmse_idxs >= t_start) & (fmse_idxs < t_end)
    fmse_idxs_chunk = fmse_idxs[in_chunk]
    mask_idxs_chunk = mask_idxs[in_chunk]

    for fi, mi in zip(fmse_idxs_chunk, mask_idxs_chunk):
        i = fi - t_start

        mask_global = mask_ds.mcs_mask.isel(time=mi).compute().values
        mask_global = np.nan_to_num(mask_global, nan=0.0)
        mask_wam    = mask_global[wam_positions].astype(np.int32)

        mcs_bool = mask_wam > 0
        if not mcs_bool.any():
            continue  # no MCS at this timestep, leave as NaN

        mcs_fmse_out[i, :, mcs_bool] = fmse_chunk[i, :, mcs_bool]
        track_id_out[i, mcs_bool] = mask_wam[mcs_bool]

    ds_out = xr.Dataset({
        'fmse': xr.DataArray(mcs_fmse_out,          dims=['time', 'pressure', 'cell']), 
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
        
        global MASK_URL, ZOOM
        ZOOM = models.MODELS[model]['zoom']
        MASK_URL = models.mask_url(model)
        mask_ds = xr.open_zarr(MASK_URL, chunks={})
        fmse_ds = xr.open_zarr(models.data_dir(model) / f'fmse_{region}.zarr')

        compute_chunk(fmse_ds, mask_ds, chunk, model, region)
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
