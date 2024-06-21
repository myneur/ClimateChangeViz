# with open('ClimateProjections.py', 'r') as f: exec(f.read())
# with open('debug-snippet.py', 'r') as f: exec(f.read())
# debug()

import os
import xarray as xr
import pandas as pd
import numpy as np
import cftime

DATADIR = '/Users/myneur/Downloads/ClimateData/debug/'

def geog_agg(fn):
    print(fn)
    try:
        ds = xr.open_dataset(f'{DATADIR}{fn}')
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return 
    exp = ds.attrs['experiment_id']
    mod = ds.attrs['source_id']
    da = ds['tas']
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"
    da_weighted = da.weighted(weights)
    da_agg = da_weighted.mean(['lat', 'lon'])
    da_yr = da_agg.groupby('time.year').mean()
    da_yr = da_yr - 273.15
    da_yr = da_yr.assign_coords(model=mod)
    da_yr = da_yr.expand_dims('model')
    da_yr = da_yr.assign_coords(experiment=exp)
    da_yr = da_yr.expand_dims('experiment')


    units = da_agg['time'].attrs['units']
    #cftime.num2date(time_data, units=time_units, calendar='360_day')
    print(fn)
    print(f'units: {units} ')
    return da_yr
    da_yr.to_netcdf(path=f'{DATADIR}cmip6_agg_{exp}_{mod}_{str(da_yr.year[0].values)}.nc')

files = [f for f in os.listdir(DATADIR) if f.endswith('.nc') and f.startswith(f'tas_')] 
ds
for file in files[0:1]: 
    ds = geog_agg(file)


