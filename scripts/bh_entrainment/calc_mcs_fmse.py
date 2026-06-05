""" 
Compute per-track frozen MSE statistics for region-filtered MCS tracks

Links fmse_<region>.zarr (3-hourly with the PyFLEXTRKR MCS pixel mask (hourly) to produce
a NetCDF with dims (tracks, times_3h) following PyFLEXTRKR output conventions.

Usage: 

""" 

import argparse
import json
import numpy as np 
import xarray as xr 
import dask.array as dsa 
from multiprocessing import Pool 
import warnings
import sys
import intake 
from pathlib import Path 
import easygems.healpix as egh 
import src.models as models 
import src.mcs_tracks
import microphysics as micro
from metpy.calc import dewpoint_from_relative_humidity, virtual_temperature_from_dewpoint
from metpy.units import units 


### filter annoying warnings 
warnings.filterwarnings('ignore', message='.*The return type of `Dataset.dims`.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Relative humidity >120%.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*divide by zero encountered in log.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*invalid value encountered in divide.*', category=RuntimeWarning)



