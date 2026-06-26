import argparse
import dask.array as dsa
import json
import numpy as np 
from pathlib import Path
import src.hp_models as models 
import src.hp_utils as utils
import src.microphysics as micro
import sys

import warnings 
import xarray as xr

CHUNK_SIZE = 10 

### filter annoying warnings 
warnings.filterwarnings('ignore', message='.*The return type of `Dataset.dims`.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Relative humidity >120%.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*divide by zero encountered in log.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*invalid value encountered in divide.*', category=RuntimeWarning)



#-----------------------------------------------------------------------
# Zarr store initialization 
#-----------------------------------------------------------------------

def init_zarr(model, region): 
    region_cfg = models.REGIONS[region]
    ds = utils.open_region_dataset(model, region_cfg)

    n_times  = ds.sizes['time']
    n_cells  = ds.sizes['cell']
    n_chunks = (n_times + CHUNK_SIZE - 1) // CHUNK_SIZE

    template_data    = dsa.full((n_times, n_cells), np.nan, dtype=np.float32,
                        chunks=(CHUNK_SIZE, n_cells))

    template = xr.Dataset(
        { 
            'condensation_rate': xr.DataArray(template_data, dims=['time', 'cell'], attrs = {'units': 'kg m-2 s-1'}), 
        }, 
        coords={'time': ds.time, 'cell': ds.cell, 'lat': ds.lat, 'lon': ds.lon},
        
    )
    var = 'precip_efficiency'
    zarr_path = models.data_dir(model, var) / f'condensation_rate_{region}.zarr'
    zarr_path.parent.mkdir(parents=True, exist_ok=True) 
    template.to_zarr(zarr_path, mode='w', zarr_format=2)

    done_dir = models.done_dir(model)
    done_dir.mkdir(parents=True, exist_ok=True)
    models.init_donefile(model, region, tag='condensation_rate').touch()
    print(f'Created {zarr_path}  shape=({n_times}, {n_cells})')
    print(f'Submit array 0-{n_chunks - 1}  ({n_chunks} jobs)  via: python submit.py --model {model}')


def compute_chunk(chunk_idx, model, region, n_timesteps=None): 
    done_file = models.chunk_donefile(model, chunk_idx, tag='condensation_rate')
    if done_file.exists():
        print(f'Chunk {chunk_idx} already done, skipping.')
        return

    region_cfg = models.REGIONS[region]
    ds         = utils.open_region_dataset(model, region_cfg)
    var = 'precip_efficiency'
    zarr_path  = models.data_dir(model, var) / f'condensation_rate_{region}.zarr' 

    t_start  = chunk_idx * CHUNK_SIZE
    t_end    = min(t_start + CHUNK_SIZE, ds.sizes['time'])
    if n_timesteps is not None:
        t_end = min(t_start + n_timesteps, t_end)
    n_chunk  = t_end - t_start

    print(f'Chunk {chunk_idx}: time[{t_start}:{t_end}] ({n_chunk} timesteps)')
    ds_chunk  = ds.isel(time=slice(t_start, t_end))
    ds_desc   = ds_chunk.sortby('pressure', ascending=False) 
    
    vertical_velocity = ds_desc.wa
    temp = ds_desc.ta
    pressure = ds_desc.pressure * 100   ## convert to Pa
    qcloud = ds_desc.cli + ds_desc.clw

    condensation_rate_t = micro.get_condensation_rate(vertical_velocity, temp, pressure)
    condensation_cloud = condensation_rate_t.where(qcloud > 0, 0)
    condensation_masked = condensation_cloud.where(condensation_cloud > 0, 0)
    condensation_rate = micro.pressure_integration(condensation_masked, -pressure, axis=1)
    
    condensation_rate_out = condensation_rate.astype(np.float32)

    ds_out = xr.Dataset({
        'condensation_rate': xr.DataArray(condensation_rate_out,          dims=['time', 'cell']), 
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
        compute_chunk(chunk, model, region)
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
