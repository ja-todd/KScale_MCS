import math as maths
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.models as models
import intake
import easygems.healpix as egh
import warnings

warnings.filterwarnings('ignore', message='.*The return type of `Dataset.dims`.*', category=FutureWarning)


def hp_mods(ds):
    """Convert from CF-compliant to be compatible with egh, and attach lat/lon coords"""
    return ds.rename({'healpix_index': 'cell'}).pipe(egh.attach_coords)


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
    cat  = intake.open_catalog(models.CATALOG_URL)['UK']
    ds3h = cat[model](zoom=zoom, time='PT3H').to_dask().pipe(hp_mods)
    return ds3h.isel(cell=_region_mask(ds3h, region_cfg))


def open_region_1h_dataset(model, region_cfg):
    zoom = models.MODELS[model]['zoom']
    cat  = intake.open_catalog(models.CATALOG_URL)['UK']
    ds1h = cat[model](zoom=zoom, time='PT1H').to_dask().pipe(hp_mods)
    return ds1h.isel(cell=_region_mask(ds1h, region_cfg))
