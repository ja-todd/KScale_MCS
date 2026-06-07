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
import src.utils as utils 
import pandas as pd

### filter annoying warnings 
warnings.filterwarnings('ignore', message='.*The return type of `Dataset.dims`.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Relative humidity >120%.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*divide by zero encountered in log.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*invalid value encountered in divide.*', category=RuntimeWarning)


