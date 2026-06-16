import math as maths
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.hp_models as models
import intake
import easygems.healpix as egh
import warnings

warnings.filterwarnings('ignore', message='.*The return type of `Dataset.dims`.*', category=FutureWarning)


def hp_mods(ds):
    """Convert from CF-compliant to be compatible with egh, and attach lat/lon coords"""
    if 'healpix_index' in list(ds.dims): 
        return ds.rename({'healpix_index': 'cell'}).pipe(egh.attach_coords)
    else: 
        return ds.pipe(egh.attach_coords)


def plot_all_fields(ds_plot):
    """Plot all fields for a given dataset. Assumes that each field is 2D - i.e. sel(time=..., [pressure=...]) has been applied"""
    zoom = ds_plot.crs.attrs['refinement_level']
    projection = ccrs.Robinson(central_longitude=0)
    # Do not plot orog, land surf.
    plot_vars = [(name, da) for name, da in ds_plot.data_vars.items() if name not in {'orog', 'sftlf', 'weights'}]
    rows = maths.ceil(len(plot_vars) / 5)
    fig, axes = plt.subplots(rows, 5, figsize=(30, rows * 20 / 6), subplot_kw={'projection': projection}, layout='constrained')
    if 'pressure' in ds_plot.coords:
        plt.suptitle(f'{ds_plot.simulation} z{zoom} @{float(ds_plot.pressure)}hPa')
    else:
        plt.suptitle(f'{ds_plot.simulation} z{zoom}')

    for ax, (name, da) in zip(axes.flatten(), plot_vars):
        if name == 'mrsol':
            da = da.isel(depth=0)
            name = 'mrsol@depth=0'
        time = pd.Timestamp(ds_plot.time.values.item())

        if abs(da.max() + da.min()) / (da.max() - da.min()) < 0.5:
            # data looks like it needs a diverging cmap.
            # figure out some nice bounds.
            pl, pu = np.percentile(da.values[~np.isnan(da.values)], [2, 98])
            vmax = np.abs([pl, pu]).max()
            kwargs = dict(
                cmap='bwr',
                vmin=-vmax,
                vmax=vmax,
            )
        else:
            kwargs = {}
        ax.set_title(f'time: {time} - {name}')
        ax.set_global()
        im = egh.healpix_show(da, ax=ax, **kwargs);
        long_name = da.long_name

        plt.colorbar(im, label=f'{long_name} ({da.attrs.get("units", "-")})')
        ax.coastlines()


def haversine(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

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
    catalog_model_key = models.MODELS[model].get('catalog_key', model)
    cat  = intake.open_catalog(models.CATALOG_URL)['UK']
    ds3h = cat[catalog_model_key](zoom=zoom, time='PT3H').to_dask().pipe(hp_mods)
    return ds3h.isel(cell=_region_mask(ds3h, region_cfg))


def open_region_1h_dataset(model, region_cfg):
    zoom = models.MODELS[model]['zoom']
    cat  = intake.open_catalog(models.CATALOG_URL)['UK']
    ds1h = cat[model](zoom=zoom, time='PT1H').to_dask().pipe(hp_mods)
    return ds1h.isel(cell=_region_mask(ds1h, region_cfg))


#----------------------------------------------------------------------
# MCS-specific functions 
#----------------------------------------------------------------------

# ceil(650 h / 3 h): max track duration in 3-hourly steps
MAX_TIMES_3H = 217

def load_track_stats(STATS_URL):
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

def open_mcs_mask(MASK_URL):
    """Open the MCS pixel mask zarr from storage on disc."""
    print('Opening MCS mask zarr...')
    return xr.open_zarr(MASK_URL, chunks={})



def align_times(ds, mask_ds):
    """
    Find 3-hourly dataset times that exist in the hourly mask dataset.
    Returns:
        times_3h   : 1-D array of datetime64 values (3-hourly, in overlap)
        mask_indices: corresponding positional indices in mask_ds.time
    """
    var_times = ds.time.values
    mask_times = mask_ds.time.values

    mask_time_to_idx = {pd.Timestamp(t): i for i, t in enumerate(mask_times)}

    overlap_var = []
    overlap_mask_idx = []
    for i, t in enumerate(var_times):
        ts = pd.Timestamp(t)
        if ts in mask_time_to_idx:
            overlap_var.append(i)
            overlap_mask_idx.append(mask_time_to_idx[ts])

    print(f'Overlap: {len(overlap_var)} 3-hourly timesteps '
          f'({pd.Timestamp(var_times[overlap_var[0]])} – '
          f'{pd.Timestamp(var_times[overlap_var[-1]])})')

    return (np.array(overlap_var),
            np.array(overlap_mask_idx),
            var_times[overlap_var])




def compute_wam_positions(var_ds, mask_ds):
    """
    Compute positional indices of WAM cells within the global mask cell array.
    Returns an int array of shape (n_wam_cells,).
    """
    wam_cells    = var_ds.cell.values             # HEALPix cell numbers, WAM subset
    if 'healpix_index' in list(mask_ds.dims): 
        global_cells = mask_ds.healpix_index.values    # HEALPix cell numbers, global (0…N-1)
    else: 
        global_cells = mask_ds.cell.values

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
