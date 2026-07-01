import argparse
import json
import numpy as np 
import xarray as xr 
import dask.array as dsa 
import warnings
import sys
from pathlib import Path 
import easygems.healpix as egh 
import src.hp_models as models 
from src.hp_utils import open_region_dataset, compute_wam_positions, align_times, open_region_1h_dataset

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

def init_zarr(model, region): 
    region_cfg = models.REGIONS[region]
    ds = open_region_dataset(model, region_cfg)

    n_times  = ds.sizes['time']
    n_cells  = ds.sizes['cell']
    n_chunks = (n_times + CHUNK_SIZE - 1) // CHUNK_SIZE

    template_data    = dsa.full((n_times, n_cells), np.nan, dtype=np.float32,
                        chunks=(CHUNK_SIZE, n_cells))

    template = xr.Dataset(
        { 
            'condensation_rate': xr.DataArray(template_data, dims=['time', 'cell'], attrs = {'units': 'kg m2 s-1'}),
            'precip_flux': xr.DataArray(template_data, dims=['time', 'cell'], attrs = {'units': 'kg m2 s-1'}),
            'track_id': xr.DataArray(template_data, dims=['time', 'cell'], attrs={'description': 'MCS track number at each cell'})
        }, 
        coords={'time': ds.time, 'cell': ds.cell, 'lat': ds.lat, 'lon': ds.lon},
        
    )
    zarr_path = models.data_dir(model, VAR) / f'mcs_condensation_rate_{region}.zarr'
    zarr_path.parent.mkdir(parents=True, exist_ok=True) 
    template.to_zarr(zarr_path, mode='w', zarr_format=2)

    done_dir = models.done_dir(model)
    done_dir.mkdir(parents=True, exist_ok=True)
    models.init_donefile(model, region, tag='mcs_condensation_rate').touch()
    print(f'Created {zarr_path}  shape=({n_times}, {n_cells})')
    print(f'Submit array 0-{n_chunks - 1}  ({n_chunks} jobs)  via: python submit.py --model {model}')


def compute_chunk(cr_ds, precip_ds, mask_ds, chunk_idx, model, region, n_timesteps=None): 
    done_file = models.chunk_donefile(model, chunk_idx, tag='mcs_condensation_rate')
    if done_file.exists():
        print(f'Chunk {chunk_idx} already done, skipping.')
        return
    zarr_path              = models.data_dir(model, VAR) / f'mcs_condensation_rate_{region}.zarr' 
    
    wam_positions          = compute_wam_positions(cr_ds, mask_ds)
    cr_idxs, mask_idxs, _  = align_times(cr_ds, mask_ds)
    cr_pr_idxs, pr_idxs, _ = align_times(cr_ds, precip_ds)

    t_start   = chunk_idx * CHUNK_SIZE
    t_end     = min(t_start + CHUNK_SIZE, cr_ds.sizes['time'])
    if n_timesteps is not None:
        t_end = min(t_start + n_timesteps, t_end)
    n_chunk   = t_end - t_start

    in_chunk        = (cr_idxs >= t_start) & (cr_idxs < t_end)
    cr_idxs_chunk   = cr_idxs[in_chunk]
    mask_idxs_chunk = mask_idxs[in_chunk]

    cr_to_pr = dict(zip(cr_pr_idxs, pr_idxs))
    pr_idxs_chunk = np.array([cr_to_pr[ci] for ci in cr_idxs_chunk])

    print(f'Chunk {chunk_idx}: time[{t_start}:{t_end}] ({n_chunk} timesteps)')
    cr_chunk        = cr_ds.isel(time=slice(t_start, t_end))['condensation_rate'].compute().values
    pr_chunk        = precip_ds.isel(time=pr_idxs_chunk)['pr'].compute().values
    n_cells         = cr_ds.sizes['cell']

    mcs_cr_out      = np.full_like(cr_chunk, np.nan, dtype=np.float32)
    mcs_pr_out      = np.full_like(pr_chunk, np.nan, dtype=np.float32)
    track_id_out    = np.full((n_chunk, n_cells), np.nan, dtype=np.float32)

    for idx, (ci, mi) in enumerate(zip(cr_idxs_chunk, mask_idxs_chunk)):
        i    = ci - t_start
        i_pr = idx

        mask_global = mask_ds.mcs_mask.isel(time=mi).compute().values
        mask_global = np.nan_to_num(mask_global, nan=0.0)
        mask_wam    = mask_global[wam_positions].astype(np.int32)

        mcs_bool    = mask_wam > 0
        if not mcs_bool.any():
            continue  # no MCS at this timestep, leave as NaN

        mcs_cr_out[i, mcs_bool]    = cr_chunk[i, mcs_bool]
        mcs_pr_out[i, mcs_bool]    = pr_chunk[i_pr, mcs_bool]
        track_id_out[i, mcs_bool]  = mask_wam[mcs_bool]

    ds_out = xr.Dataset({
        'condensation_rate': xr.DataArray(mcs_cr_out,          dims=['time', 'cell']), 
        'precip_flux'      : xr.DataArray(mcs_pr_out,          dims=['time', 'cell']),
        'track_id':          xr.DataArray(track_id_out,        dims=['time', 'cell'])
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
        mask_ds = xr.open_zarr(MASK_URL, chunks={})


        cr_ds     = xr.open_zarr(models.data_dir(model, VAR) / f'condensation_rate_{region}.zarr')
        precip_ds = open_region_1h_dataset(model, region_cfg)

        compute_chunk(cr_ds, precip_ds, mask_ds, chunk, model, region)
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