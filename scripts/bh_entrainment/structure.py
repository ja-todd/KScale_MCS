import numpy as np 
import xarray as xr 
import dask.array as dsa 
from multiprocessing import Pool 
from pathlib import Path 
import easygems.healpix as egh 
import src.models as models 
import microphysics as mp



CHUNK_SIZE = 10 
N_WORKERS = 10 



### filter annoying warnings 
warnings.filterwarnings('ignore', message='.*The return type of `Dataset.dims`.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Relative humidity >120%.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*divide by zero encountered in log.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*invalid value encountered in divide.*', category=RuntimeWarning)



#----------------------------------------------------------------------
# Data loading 
#----------------------------------------------------------------------

"""These functions are basically the same as MMs code from the 2026 Kscale Hackathon"""

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


def open_region_1h_dataset(model, region_cfg):
    zoom = models.MODELS[model]['zoom']
    cat  = intake.open_catalog(models.CATALOG_URL)['UK']
    ds1h = cat[model](zoom=zoom, time='PT1H').to_dask().pipe(hp_mods)
    return ds1h.isel(cell=_region_mask(ds1h, region_cfg))



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
            'fmse': xr.DataArray(template_data, dims=['time', 'pressure', 'cell'], attrs = {'units': 'J kg-1'})


        }, 
        coords={'time': ds.time, 'pressure': ds.pressure, 'cell': ds.cell, 'lat': ds.lat, 'lon': ds.lon},
        
        )

    zarr_path = models.data_dir(model) / f'fmse_{region}.zarr'    ### check this 
    zarr_path.parent.mkdir(parents=True, exist_ok=True)   # #check this 
    template.to_zarr(zarr_path, mode='w', zarr_format=2)

    done_dir = models.done_dir(model)
    done_dir.mkdir(parents=True, exist_ok=True)
    models.init_donefile(model, region).touch()
    print(f'Created {zarr_path}  shape=({n_times}, {n_cells})')
    print(f'Submit array 0-{n_chunks - 1}  ({n_chunks} jobs)  via: python submit.py --model {model}')

#-----------------------------------------------------------------------
# MSE computation (h = c_p*T + gz + L_v*q_v + L_i*q_i) (Becker et al., 2018; Becker and Hohenegger, 2021)
#-----------------------------------------------------------------------

def calc_fmse(): 
    fmse = mp.cp*T + mp.g*z + mp.Lv*q + mp.Lf*q_i

    return fmse 


def _worker(args): 
    """ Pool worker: Frozen MSE for all cells at a single timestep """










#-----------------------------------------------------------------------
# Chunk processing
#-----------------------------------------------------------------------

def compute_chunk(chunk_idx, model, region, n_timesteps=None): 
    done_file  = models.chunk_donefile(model, chunk_idx)   ### need to check whether this is fine 
    if done_file.exists():
        print(f'Chunk {chunk_idx} already done, skipping.')
        return

    region_cfg = models.REGIONS[region]
    ds         = open_region_dataset(model, region_cfg)
    ds1h       = open_region_1h_dataset(model, region_cfg)
    zarr_path  = models.data_dir(model) / f'entrainment_{region}.zarr'  ## check this 

    t_start  = chunk_idx * CHUNK_SIZE
    t_end    = min(t_start + CHUNK_SIZE, ds.sizes['time'])
    if n_timesteps is not None:
        t_end = min(t_start + n_timesteps, t_end)
    n_chunk  = t_end - t_start

    print(f'Chunk {chunk_idx}: time[{t_start}:{t_end}] ({n_chunk} timesteps)')
    ds_chunk  = ds.isel(time=slice(t_start, t_end))
    ds_desc   = ds_chunk.sortby('pressure', ascending=False)  ## probably don't need to do this 
    
    ta_np = ds.desc.ta.compute().values 
    
    



#-----------------------------------------------------------------------
# 
#-----------------------------------------------------------------------





