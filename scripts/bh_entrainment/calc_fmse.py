
"""

MSE computation (h = c_p*T + gz + L_v*q_v + L_i*q_i) (Stirling and Stratton, 2012; Becker and Hohenegger, 2021)
Produces 4D field of frozen MSE over <region> cells. 

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
from src.utils import open_region_dataset
import microphysics as micro
from metpy.calc import dewpoint_from_relative_humidity, virtual_temperature_from_dewpoint
from metpy.units import units 


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
            'z': xr.DataArray(template_data, dims=['time', 'pressure', 'cell'], attrs = {'units': 'm'}),
            'rho': xr.DataArray(template_data, dims=['time', 'pressure', 'cell'], attrs = {'units': 'kg m-3'})

        }, 
        coords={'time': ds.time, 'pressure': ds.pressure.sortby('pressure', ascending=False), 'cell': ds.cell, 'lat': ds.lat, 'lon': ds.lon},
        
    )

    zarr_path = models.data_dir(model) / f'fmse_{region}.zarr'
    zarr_path.parent.mkdir(parents=True, exist_ok=True) 
    template.to_zarr(zarr_path, mode='w', zarr_format=2)

    done_dir = models.done_dir(model)
    done_dir.mkdir(parents=True, exist_ok=True)
    models.init_donefile(model, region, tag='fmse').touch()
    print(f'Created {zarr_path}  shape=({n_times}, {n_pressures}, {n_cells})')
    print(f'Submit array 0-{n_chunks - 1}  ({n_chunks} jobs)  via: python submit.py --model {model}')



#-----------------------------------------------------------------------
# Chunk processing (MSE computation)
#-----------------------------------------------------------------------

def compute_chunk(chunk_idx, model, region, n_timesteps=None): 
    done_file = models.chunk_donefile(model, chunk_idx, tag='fmse')
    if done_file.exists():
        print(f'Chunk {chunk_idx} already done, skipping.')
        return

    region_cfg = models.REGIONS[region]
    ds         = open_region_dataset(model, region_cfg)
    zarr_path  = models.data_dir(model) / f'fmse_{region}.zarr' 

    t_start  = chunk_idx * CHUNK_SIZE
    t_end    = min(t_start + CHUNK_SIZE, ds.sizes['time'])
    if n_timesteps is not None:
        t_end = min(t_start + n_timesteps, t_end)
    n_chunk  = t_end - t_start

    print(f'Chunk {chunk_idx}: time[{t_start}:{t_end}] ({n_chunk} timesteps)')
    ds_chunk  = ds.isel(time=slice(t_start, t_end))
    ds_desc   = ds_chunk.sortby('pressure', ascending=False) 
    
    p_diffs = ds_desc.pressure.diff('pressure').values[np.newaxis, :, np.newaxis] * 100

    desc_t = ds_desc.ta.compute().values * units.K
    desc_rh = (ds_desc.hur.compute().values / 100) * units.dimensionless
    desc_rh = np.clip(desc_rh.magnitude, 1e-6, 1.0) * units.dimensionless
    desc_p = ds_desc.pressure.compute().values[np.newaxis, :, np.newaxis] * units.hPa

    desc_q = ds_desc.hus.compute().values
    desc_q = np.nan_to_num(ds_desc.hus.compute().values, nan=0.0)

    desc_qi = ds_desc.cli.compute().values
    desc_qi = np.nan_to_num(ds_desc.cli.compute().values, nan=0.0)

    desc_td = dewpoint_from_relative_humidity(desc_t, desc_rh)
    desc_tv = virtual_temperature_from_dewpoint(desc_p, desc_t, desc_td)
    
    rho = (desc_p * 100) / (micro.R * desc_tv)
    rho_out = rho.magnitude.astype(np.float32)

    dz = (-p_diffs / (rho[:, :-1, :] * micro.g)).magnitude  #inverse hydrostatic 
    z = np.concatenate([np.zeros((dz.shape[0], 1, dz.shape[2])), np.nancumsum(dz, axis=1)], axis=1)
    z_out = z.astype(np.float32)


    fmse_out = (micro.cp * desc_t.magnitude) + (micro.g * z) + (micro.Lv * desc_q) + (micro.Lf * desc_qi)
    fmse_out = fmse_out.astype(np.float32)

    print('fmse sample:', fmse_out[0, -1, 1000])

    ds_out = xr.Dataset({
        'fmse': xr.DataArray(fmse_out,          dims=['time', 'pressure', 'cell']), 
        'z':xr.DataArray(z_out,                 dims=['time', 'pressure', 'cell']), 
        'rho':xr.DataArray(rho_out,                 dims=['time', 'pressure', 'cell'])
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






