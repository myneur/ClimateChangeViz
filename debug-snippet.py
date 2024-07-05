# with open('ClimateProjections.py', 'r') as f: exec(f.read())
# with open('debug-snippet.py', 'r') as f: exec(f.read())
# debug()

import re
import os

import fnmatch
import glob

import xarray as xr


BLUE = "\033[34m"
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
BLUEBG = "\033[44m"
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
GREY = "\033[47;30m"
HEADER = '\033[95m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
RESET = "\033[0m"

ds = xr.open_dataset(['tas_Amon_UKESM1-0-LL_ssp245_r13i1p1f2_gn_205001-210012_v20190507.nc', 'tas_Amon_UKESM1-0-LL_ssp245_r1i1p1f2_gn_205001-210012_v20190507.nc'])
ds = ds['tas']
import visualisations

chart = visualisations.Charter()
chart.plot([ds], series='model')