""" 
Compute per-track frozen MSE statistics for region-filtered MCS tracks

Links fmse_<region>.zarr (3-hourly) with the PyFLEXTRKR MCS pixel mask (hourly) to produce
a NetCDF with dims (tracks, times_3h) following PyFLEXTRKR output conventions.

OR (if --no-mcs): 
Computes frozen MSE statistics for region-filtered MCS updrafts & associated environments 



Code logic: 

Output updraft frozen moist static energy and per-updraft (MCS or not) mean frozen moist static energy for the environment

Usage: 

    python submit.py --model <model_id> --script calc_mcs_env_updraft_fmse 
    python submit.py --model <model_id> --script calc_mcs_env_updraft_fmse --radius <radius> --region <region> --no-mcs
    python submit.py --model <model_id> --script calc_mcs_env_updraft_fmse --radius <radius> --region <region>

"""

import argparse
import json
import numpy as np 
import xarray as xr 
import dask.array as dsa 
import warnings
import sys
from pathlib import Path 
import src.microphysics as micro
import src.hp_models as models 
from src.hp_utils import haversine, open_region_dataset, align_times,\
                     compute_wam_positions
                    
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


def init_zarr(model, region, radius, mcs=True): 
    region_cfg = models.REGIONS[region]
    ds = open_region_dataset(model, region_cfg)

    n_times     = ds.sizes['time']
    n_pressures = ds.sizes['pressure']
    n_cells     = ds.sizes['cell']
    n_chunks    = (n_times + CHUNK_SIZE - 1) // CHUNK_SIZE

    template_data = dsa.full((n_times, n_pressures, n_cells), np.nan, dtype=np.float32,
                        chunks=(CHUNK_SIZE, n_pressures, n_cells))

    common_vars = {
        'fmse_env':             xr.DataArray(template_data, dims=['time', 'pressure', 'cell'], attrs={'units': 'J kg-1'}),
        'fmse_updraft':         xr.DataArray(template_data, dims=['time', 'pressure', 'cell'], attrs={'units': 'J kg-1'}),
        'z_updraft':            xr.DataArray(template_data, dims=['time', 'pressure', 'cell'], attrs={'units': 'm'}),
        'rho_updraft':          xr.DataArray(template_data, dims=['time', 'pressure', 'cell'], attrs={'units': 'kg m-3'}),
        'w_updraft':            xr.DataArray(template_data, dims=['time', 'pressure', 'cell'], attrs={'units': 'm s-1'}),
        'updraft_mass_flux':    xr.DataArray(template_data, dims=['time', 'pressure', 'cell'], attrs={'units': 'kg m-2 s-1'}),
        'updraft_buoyancy':    xr.DataArray(template_data, dims=['time', 'pressure', 'cell'], attrs={'units': 'm s-2'})
    }

    common_coords = {'time': ds.time, 'pressure': ds.pressure.sortby('pressure', ascending=False), 
                     'cell': ds.cell, 'lat': ds.lat, 'lon': ds.lon}

    if mcs:
        common_vars['track_id'] = xr.DataArray(
            dsa.full((n_times, n_cells), np.nan, dtype=np.float32, chunks=(CHUNK_SIZE, n_cells)),
            dims=['time', 'cell'],
            attrs={'description': 'MCS track number at each cell'}
        )
        zarr_path = models.data_dir(model) / f'mcs_env_updraft_fmse_{region}_{radius}km.zarr'
        done_tag  = f'mcs_env_updraft_fmse_{radius}km'
    else:
        zarr_path = models.data_dir(model) / f'env_updraft_fmse_{region}_{radius}km.zarr'
        done_tag  = f'env_updraft_fmse_{radius}km'

    template = xr.Dataset(common_vars, coords=common_coords)

    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_zarr(zarr_path, mode='w', zarr_format=2)

    done_dir = models.done_dir(model)
    done_dir.mkdir(parents=True, exist_ok=True)
    models.init_donefile(model, region, tag=done_tag).touch()

    print(f'Created {zarr_path}  shape=({n_times}, {n_pressures}, {n_cells})')
    print(f'Submit array 0-{n_chunks - 1}  ({n_chunks} jobs)  via: python submit.py --model {model}')


def get_env_field(field_t, updraft_lats, updraft_lons, all_lats, all_lons, 
                 updraft_bool, radius, batch_size=100):
    """
    For each updraft cell, compute mean environment FMSE within radius.
    Returns fmse_env_per_updraft of shape (pressure, n_updrafts)
    """
    n_updrafts = len(updraft_lats)
    n_pressure = field_t.shape[0]
    field_env_per_updraft = np.full((n_pressure, n_updrafts), np.nan, dtype=np.float32)
    
    for start in range(0, n_updrafts, batch_size): 
        end=min(start + batch_size, n_updrafts)

        batch_lats = updraft_lats[start:end]
        batch_lons = updraft_lons[start:end]

        dist_matrix = haversine(batch_lats[:, None], batch_lons[:, None], 
                                all_lats[None, :], all_lons[None, :])
        
        dist_within_radius = dist_matrix < radius 
        is_env_cell = ~updraft_bool     # (N, )

        env_matrix = dist_within_radius & is_env_cell[None, :]
        no_env_matrix = ~env_matrix.any(axis=1)

        env_mask = np.where(env_matrix[None, :, :], 
                            field_t[:, None, :], np.nan)

        batch_env_mean = np.nanmean(env_mask, axis=2)

        batch_env_mean[:, no_env_matrix] = np.nan

        field_env_per_updraft[:, start:end] = batch_env_mean

    return field_env_per_updraft # (pressure, n_updrafts)



def compute_chunk_no_mcs(full_ds, fmse_ds, chunk_idx,
                          model, region, radius, n_timesteps=None, w_updraft_threshold=1, qc_updraft_threshold=1e-5):
    done_file = models.chunk_donefile(model, chunk_idx, tag=f'env_updraft_fmse_{region}_{radius}km')
    if done_file.exists():
        print(f'Chunk {chunk_idx} already done, skipping.')
        return
    zarr_path     = models.data_dir(model) / f'env_updraft_fmse_{region}_{radius}km.zarr'
    
    fmse_idxs = np.arange(t_start, t_end)

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
    updraft_mass_flux_out = np.full_like(fmse_chunk, np.nan, dtype=np.float32)
    updraft_buoyancy_out = np.full_like(fmse_chunk, np.nan, dtype=np.float32)

    in_chunk        = (fmse_idxs >= t_start) & (fmse_idxs < t_end)
    fmse_idxs_chunk = fmse_idxs[in_chunk]

    w_chunk = full_ds.wa.isel(time=slice(t_start, t_end))
    ql_chunk = full_ds.clw.isel(time=slice(t_start, t_end))
    qi_chunk = full_ds.cli.isel(time=slice(t_start, t_end))
    qc_chunk = ql_chunk + qi_chunk 

    p_levs = fmse_ds.pressure.values
    p500_idx = np.where(p_levs == 500)[0].item(0)
    all_lats = fmse_ds.lat.values 
    all_lons = fmse_ds.lon.values

    for fi in fmse_idxs_chunk: 
        w_t = w_chunk.isel(time=fi).compute().values
        qc_t = qc_chunk.isel(time=fi).compute().values

        w_mask       = w_t > w_updraft_threshold
        qc_mask      = qc_t > qc_updraft_threshold
        updraft_mask = w_mask & qc_mask

        cell_updraft = updraft_mask.any(axis=0)
        where_updraft = np.where(cell_updraft)[0]

        
        w_500 = w_t[p500_idx, :]
        qc_500 = qc_t[p500_idx, :]
        updraft_500_mask = (w_500 > w_updraft_threshold) & (qc_500 > qc_updraft_threshold)

        deep_updraft_mask = np.zeros(len(where_updraft), dtype=bool)

        updraft_lats = all_lats[where_updraft]
        updraft_lons = all_lons[where_updraft]
        


        dist_matrix = haversine(
            updraft_lats[:, None], updraft_lons[:, None], 
            all_lats[None, :], all_lons[None, :]
        )

        nearby_matrix = dist_matrix <= 10 
        deep_updraft_mask = (nearby_matrix & updraft_500_mask[None, :]).any(axis=1)

        where_updraft = where_updraft[deep_updraft_mask]

        if len(where_updraft) == 0:
            continue
        
        updraft_lats = all_lats[where_updraft]
        updraft_lons = all_lons[where_updraft]

        fmse_t = fmse_ds.fmse.isel(time=fi).compute().values 
        z_t    = fmse_ds.z.isel(time=fi).compute().values
        rho_t  = fmse_ds.rho.isel(time=fi).compute().values

        fmse_updrafts = fmse_t[:, where_updraft]   
        z_updrafts    = z_t[:, where_updraft] 
        rho_updrafts  = rho_t[:, where_updraft]
        w_updrafts    = w_t[:, where_updraft]
        updraft_mass_fluxes = rho_updrafts * w_updrafts

        updraft_bool = np.zeros(n_cells, dtype=bool)
        updraft_bool[where_updraft] = True

        fmse_env_per_updraft = get_env_field(fmse_t, updraft_lats, updraft_lons, 
                                all_lats, all_lons, updraft_bool, radius)
        
        rho_env_per_updraft = get_env_field(rho_t, updraft_lats, updraft_lons, 
                                all_lats, all_lons, updraft_bool, radius)
        
        updraft_buoyancy = micro.g * ((rho_env_per_updraft - rho_updrafts) / rho_updrafts)
        
        fmse_updraft_out[fi, :, where_updraft]  = fmse_updrafts.T
        fmse_env_out[fi, :, where_updraft]      = fmse_env_per_updraft.T
        rho_updraft_out[fi, :, where_updraft]   = rho_updrafts.T
        z_updraft_out[fi, :, where_updraft]     = z_updrafts.T
        w_updraft_out[fi, :, where_updraft]     = w_updrafts.T
        updraft_mass_flux_out[fi, :, where_updraft] = updraft_mass_fluxes.T
        updraft_buoyancy_out[fi, :, where_updraft] = updraft_buoyancy.T

    ds_out = xr.Dataset({
        'fmse_env': xr.DataArray(fmse_env_out,          dims=['time', 'pressure', 'cell']),
        'fmse_updraft': xr.DataArray(fmse_updraft_out,          dims=['time', 'pressure', 'cell']),  
        'z_updraft': xr.DataArray(z_updraft_out,          dims=['time', 'pressure', 'cell']),
        'rho_updraft': xr.DataArray(rho_updraft_out,          dims=['time', 'pressure', 'cell']),
        'w_updraft': xr.DataArray(w_updraft_out,          dims=['time', 'pressure', 'cell']),
        'updraft_mass_flux': xr.DataArray(updraft_mass_flux_out,          dims=['time', 'pressure', 'cell']),
        'updraft_buoyancy': xr.DataArray(updraft_buoyancy_out,          dims=['time', 'pressure', 'cell']),
    })
    
    print(f"Chunk {chunk_idx}: before to_zarr")

    ds_out.to_zarr(zarr_path, region={'time': slice(t_start, t_end)})

    print(f"Chunk {chunk_idx}: after to_zarr")

    done_file.touch()
    print(f'Chunk {chunk_idx} written')


def compute_chunk(full_ds, fmse_ds, mask_ds, chunk_idx,
                   model, region, radius, n_timesteps=None, w_updraft_threshold=1, qc_updraft_threshold=1e-5): 
    done_file = models.chunk_donefile(model, chunk_idx, tag=f'mcs_env_updraft_fmse_{region}_{radius}km')
    if done_file.exists():
        print(f'Chunk {chunk_idx} already done, skipping.')
        return
    zarr_path     = models.data_dir(model) / f'mcs_env_updraft_fmse_{region}_{radius}km.zarr' 
    
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
    updraft_mass_flux_out = np.full_like(fmse_chunk, np.nan, dtype=np.float32)
    updraft_buoyancy_out = np.full_like(fmse_chunk, np.nan, dtype=np.float32)
    track_id_out = np.full((n_chunk, n_cells), np.nan, dtype=np.float32)

    in_chunk        = (fmse_idxs >= t_start) & (fmse_idxs < t_end)
    fmse_idxs_chunk = fmse_idxs[in_chunk]
    mask_idxs_chunk = mask_idxs[in_chunk]

    w_chunk = full_ds.wa.isel(time=slice(t_start, t_end))
    ql_chunk = full_ds.clw.isel(time=slice(t_start, t_end))
    qi_chunk = full_ds.cli.isel(time=slice(t_start, t_end))
    qc_chunk = ql_chunk + qi_chunk 

    p_levs = fmse_ds.pressure.values
    p500_idx = np.where(p_levs == 500)[0].item(0)
    all_lats = fmse_ds.lat.values 
    all_lons = fmse_ds.lon.values

    for fi, mi in zip(fmse_idxs_chunk, mask_idxs_chunk):
        
        i = fi - t_start
        print(f"Chunk {chunk_idx}: starting timestep {i} fi={fi}")

        print("getting global mask")
        mask_global = mask_ds.mcs_mask.isel(time=mi).compute().values
        mask_global = np.nan_to_num(mask_global, nan=0.0)
        mask_wam    = mask_global[wam_positions].astype(np.int32)

        print("got global mask")

        mcs_bool = mask_wam > 0 # (N, )
        if not mcs_bool.any():
            continue  # leave as NaN because no MCS here
        
        w_t = w_chunk.isel(time=i).compute().values
        qc_t = qc_chunk.isel(time=i).compute().values

        w_mcs = w_t[:, mcs_bool]
        qc_mcs = qc_t[:, mcs_bool]

        w_mask       = w_mcs > w_updraft_threshold
        qc_mask      = qc_mcs > qc_updraft_threshold
        updraft_mask = w_mask & qc_mask


        cell_updraft = updraft_mask.any(axis=0)  ### boolean array (N, ) for every grid cell
        where_updraft = np.where(cell_updraft)[0] ### integer index array, where cell_updraft is True

        where_mcs    = np.where(mcs_bool)[0]  ## mcs mask on original grid 
        where_mcs_updraft = where_mcs[where_updraft]  ## cell positions of the updrafts on original region grid

        
        w_500 = w_t[p500_idx, :]
        qc_500 = qc_t[p500_idx, :]
        updraft_500_mask = (w_500 > w_updraft_threshold) & (qc_500 > qc_updraft_threshold)

        deep_updraft_mask = np.zeros(len(where_mcs_updraft), dtype=bool)

        updraft_lats = all_lats[where_mcs_updraft]
        updraft_lons = all_lons[where_mcs_updraft]
        


        dist_matrix = haversine(
            updraft_lats[:, None], updraft_lons[:, None], 
            all_lats[None, :], all_lons[None, :]
        )

        nearby_matrix = dist_matrix <= 10 
        deep_updraft_mask = (nearby_matrix & updraft_500_mask[None, :]).any(axis=1)

        where_mcs_updraft = where_mcs_updraft[deep_updraft_mask]

        if len(where_mcs_updraft) == 0:
            continue
        
        updraft_lats = all_lats[where_mcs_updraft]
        updraft_lons = all_lons[where_mcs_updraft]

        

        fmse_t = fmse_ds.fmse.isel(time=fi).compute().values 
        z_t    = fmse_ds.z.isel(time=fi).compute().values
        rho_t  = fmse_ds.rho.isel(time=fi).compute().values

        fmse_updrafts = fmse_t[:, where_mcs_updraft]   
        z_updrafts    = z_t[:, where_mcs_updraft] 
        rho_updrafts  = rho_t[:, where_mcs_updraft]
        w_updrafts    = w_t[:, where_mcs_updraft]
        updraft_mass_fluxes = rho_updrafts * w_updrafts
        
        

        updraft_bool = np.zeros(n_cells, dtype=bool)
        updraft_bool[where_mcs_updraft] = True

        fmse_env_per_updraft = get_env_field(fmse_t, updraft_lats, updraft_lons, 
                                all_lats, all_lons, updraft_bool, radius)

        rho_env_per_updraft = get_env_field(rho_t, updraft_lats, updraft_lons, 
                                all_lats, all_lons, updraft_bool, radius)
        
        updraft_buoyancy = micro.g * ((rho_env_per_updraft - rho_updrafts) / rho_updrafts)


        
        fmse_updraft_out[i, :, where_mcs_updraft]       = fmse_updrafts.T
        fmse_env_out[i, :, where_mcs_updraft]           = fmse_env_per_updraft.T
        rho_updraft_out[i, :, where_mcs_updraft]        = rho_updrafts.T
        z_updraft_out[i, :, where_mcs_updraft]          = z_updrafts.T
        w_updraft_out[i, :, where_mcs_updraft]          = w_updrafts.T
        updraft_mass_flux_out[fi, :, where_mcs_updraft] = updraft_mass_fluxes.T
        updraft_buoyancy_out[fi, :, where_mcs_updraft]  = updraft_buoyancy.T
        track_id_out[i, where_mcs_updraft]              = mask_wam[where_mcs_updraft]

    ds_out = xr.Dataset({
        'fmse_env': xr.DataArray(fmse_env_out,          dims=['time', 'pressure', 'cell']),
        'fmse_updraft': xr.DataArray(fmse_updraft_out,          dims=['time', 'pressure', 'cell']),  
        'z_updraft': xr.DataArray(z_updraft_out,          dims=['time', 'pressure', 'cell']),
        'rho_updraft': xr.DataArray(rho_updraft_out,          dims=['time', 'pressure', 'cell']),
        'w_updraft': xr.DataArray(w_updraft_out,          dims=['time', 'pressure', 'cell']),
        'updraft_mass_flux': xr.DataArray(updraft_mass_flux_out,          dims=['time', 'pressure', 'cell']),
        'updraft_buoyancy': xr.DataArray(updraft_buoyancy_out,          dims=['time', 'pressure', 'cell']),
        'track_id': xr.DataArray(track_id_out, dims=['time', 'cell'])
    })
    
    print(f"Chunk {chunk_idx}: before to_zarr")

    ds_out.to_zarr(zarr_path, region={'time': slice(t_start, t_end)})

    print(f"Chunk {chunk_idx}: after to_zarr")

    done_file.touch()
    print(f'Chunk {chunk_idx} written')

def main():
    if len(sys.argv) >= 3 and not sys.argv[1].startswith('-'):
        json_file  = Path(sys.argv[1])
        task_index = int(sys.argv[2])

        if not json_file.exists():
            sys.exit(f'Error: task file {json_file} not found')
    
        task_cfg = json.loads(json_file.read_text())
        
        if task_index >= len(task_cfg['tasks']):
            sys.exit(f'Error: task index {task_index} out of range')

        model  = task_cfg['model']
        region = task_cfg['region']
        radius = task_cfg['radius']
        mcs    = task_cfg['mcs']          # <-- from json
        chunk  = task_cfg['tasks'][task_index]['chunk']
        region_cfg = models.REGIONS[region]

        global MASK_URL, ZOOM
        ZOOM     = models.MODELS[model]['zoom']
        MASK_URL = models.mask_url(model)
        mask_ds  = xr.open_zarr(MASK_URL, chunks={}, mask_and_scale=False)

        fmse_ds = xr.open_zarr(models.data_dir(model) / f'fmse_{region}.zarr')
        full_ds = open_region_dataset(model, region_cfg).sortby('pressure', ascending=False)

        if mcs:
            compute_chunk(full_ds, fmse_ds, mask_ds, chunk, model, region, radius)
        else:
            compute_chunk_no_mcs(full_ds, fmse_ds, chunk, model, region, radius)     # <-- passed through
        return

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--init', action='store_true')
    parser.add_argument('--n-timesteps', type=int, default=None, metavar='N')
    parser.add_argument('--radius', type=int, default=None)
    parser.add_argument('--no-mcs', action='store_false', dest='mcs',
                    help='Process all updrafts in domain rather than MCS only')
    models.add_model_arg(parser)
    models.add_region_arg(parser)
    args = parser.parse_args()

    if args.init:
        init_zarr(args.model, args.region, args.radius, mcs=args.mcs)



if __name__ == '__main__':
    main()

