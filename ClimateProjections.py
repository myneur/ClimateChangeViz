# API keys from cds.climate.copernicus.eu must be in ~/.cdsapirc

# About visualized models: https://confluence.ecmwf.int/display/CKB/CMIP6%3A+Global+climate+projections#CMIP6:Globalclimateprojections-Models

# Calculations based on colab notebook: ecmwf-projects.github.io/copernicus-training-c3s/projections-cmip6.html

# Data-sets: cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6 | aims2.llnl.gov

# TODO
# unziping to different folders not to interfere
# 0. aggregated models by state as series to show how big is the coverage of the models
# 1. historical data (measurements from the constant set of stations normalized to the lastest, most complete set, to be independent on addtions)
# 2. plot range of forecast to the right edge of the chart

# WHAT to plot
variable = 'temperature'
#variable = 'max_temperature'  
variable = 'discovery'
stacked = False # aggregate into buckets
reaggregate = False # compute aggregations regardles if they already exist

#variable = 'history' # not working yet: https://web.archive.org/web/20240516185454/https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=form

scenarios = {
  'to-visualize': {
    'historical': "hindcast", 
    'ssp119': "1.5°: carbon neutral in 2050", 
    'ssp126': "2°: carbon neutral in 2075",
    'ssp245': "3° no decline till ½ millenia",},
  'out-of-focus': {
    'ssp534os': "peak at 2040, then steeper decline",
    'ssp570': "4° 2× emissions in 2100",
    'ssp585': "5° 3× emissions in 2075"}
  }

mark_failing_scenarios = True # Save unavailable experiments not to retry downloading again and again. Clean it in 'metadata/status.json'. 

# WHERE to download data (most models have hundreds MB globally)
# a subfolder according to variable is expected
DATADIR = f'/Users/myneur/Downloads/ClimateData/{variable}/'

# LET'S RUN

from glob import glob
from pathlib import Path
import os
from os.path import basename

# Data & Date
import numpy as np
import xarray as xr
import pandas as pd
import cftime
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

# Utils
import downloader
import visualizations

import util
import traceback

forecast_from = 2015 # hidcast data not available beyond 2014 anyway for most models

experiments = scenarios['to-visualize'].keys()
variables = {
  'temperature': {'request': 'near_surface_air_temperature', 'dataset': 'tas'},
  'discovery': {'request': 'near_surface_air_temperature', 'dataset': 'tas'},
  'max_temperature': {'request': 'daily_maximum_near_surface_air_temperature', 'dataset': 'tasmax', 
  'historical': {'request': '2m_temperature', 'dataset': 'tas'}}
}

md = util.loadMD('model_md')

models = md[variable] # models to be downloaded – when empty, only already downloaded files will be visualized

# DOWNLOADING 

if variable in ('temperature', 'discovery'):
  unavailable_experiments = downloader.download(models, experiments, DATADIR, mark_failing_scenarios=mark_failing_scenarios, forecast_from=forecast_from)
elif variable == 'max_temperature':
  experiments = ['historical', 'ssp245']
  frequency='daily'
  downloader.download(models, experiments, DATADIR, variable=variables[variable]['request'], area=md['area']['cz'], frequency=frequency, mark_failing_scenarios=mark_failing_scenarios, forecast_from=forecast_from)
elif variable == 'history':
  downloader.reanalysis()


cmip6_nc = list()
cmip6_nc_rel = glob(f'{DATADIR}tas*.nc')
for i in cmip6_nc_rel:
    cmip6_nc.append(os.path.basename(i))

K = 273.15 # Kelvins
def geog_agg(fn, buckets=None, area=None):
  try:
    ds = xr.open_dataset(f'{DATADIR}{fn}')

    var = 'tasmax' if 'tasmax' in ds else 'tas'
    exp = ds.attrs['experiment_id']
    mod = ds.attrs['source_id']

    # Fixing inconsistent naming
    if 'lat' in ds.coords: lat, lon = 'lat', 'lon' 
    else: lat, lon = 'latitude', 'longitude'
    
    # Narrow to selected variable
    
    da = ds[var] 
    if 'height' in da.coords:
      da = da.drop_vars('height')

    # filter within area
    if area: 
      if len(area)>3:
        da.sel({lat: slice(area[0], area[2]), lon: slice(area[1], area[3])})# N-S # W-E
      else:
        da.sel({lat: lat_value, lon: lon_value}, method='nearest')
    
    # Aggregate spatially 
    
    # Maximums
    if var == 'tasmax':
      da_agg = da.max([lat, lon])
    
    # Averages
    else:
      # Weight as longitude gird shrinks with latitude
      weights = np.cos(np.deg2rad(da[lat]))
      weights.name = "weights"
      da_weighted = da.weighted(weights)
      da_agg = da_weighted.mean([lat, lon])

    # Aggregate time
    if var == 'tasmax':
      if buckets:
        t30 = ((da_agg >= (30+K)) & (da_agg < (35+K))).resample(time='YE').sum(dim='time')
        t35 = (da_agg >= (35+K)).resample(time='YE').sum(dim='time')
        da_yr = xr.Dataset(
          {'bucket': (('bins', 'time'), [t30, t35])},
          coords={'bins': ['30-35', '35+'],'time': t30.time})

        da_yr = da_yr.assign_coords(year=da_yr['time'].dt.year)
        da_yr = da_yr.drop_vars('time')
        da_yr = da_yr.rename({'time': 'year'})
        
      else:
        da_yr = da_agg.groupby('time.year').max()
    else: 
      da_yr = da_agg.groupby('time.year').mean()

    if not buckets:
      da_yr = da_yr - K # °C
    
    # Dimensions
    da_yr = da_yr.assign_coords(model=mod)
    da_yr = da_yr.expand_dims('model')
    da_yr = da_yr.assign_coords(experiment=exp)
    da_yr = da_yr.expand_dims('experiment')

    # Attributes
    #ds.attrs['height'] =
    da_yr.to_netcdf(path=f'{DATADIR}cmip6_agg_{exp}_{mod}_{str(da_yr.year[0].values)}.nc')

  except Exception as e: print(f"\nError aggregating {fn}: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)

print('Opening aggregations')

for filename in cmip6_nc:
  model, experiment = filename.split('_')[2:4]
  
  try:
    candidate_files = [f for f in os.listdir(DATADIR) if f.endswith('.nc') and f.startswith(f'cmip6_agg_{experiment}_{model}')] # TODO: multiple files for multiple years can exist
    if not len(candidate_files) or reaggregate:
      print(f'aggregating {model} {experiment}')
      geog_agg(filename, buckets=stacked)

  except Exception as e: print(f"Error in {filename}: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)

try:
  data_ds = xr.open_mfdataset(f'{DATADIR}cmip6_agg_*.nc', combine='nested', concat_dim='model')
  data_ds.load()

  not_read = set(models)-set(data_ds.model.values.flat)
  if not_read: print("\nNOT read: '" + ' '.join(map(str, not_read)) +"'")
  
  # removing historical data before 2014, because some models can include them despite request
  def filter_years(ds):
    if 'historical' in ds['experiment']:
      ds = ds.sel(year=ds['year'] < forecast_from)
    return ds
  data_ds_filtered = data_ds.groupby('experiment').map(filter_years) #, squeeze=True

  # merging data series of different periods for the same model
  data_ds_filtered = data_ds.groupby('model').mean('model')
  print (data_ds_filtered)

# VISUALIZE

  if stacked:

    # COUNTS IN BUCKETS
    chart = visualizations.Charter(data_ds_filtered, variable=variable)

    if variable == 'max_temperature':    
      chart.stack(title=f'Tropic days (in Czechia) projection ({len(models)} CMIP6 models)', ylabel='Tropic days annualy', marker=forecast_from)
  
  else: 
    if variable == 'discovery':
      chart = visualizations.Charter(data_ds, variable=variable)
      chart.plot(what={'experiment': "ssp126"})
    else:

      # TEMPERATURES
      data = data_ds_filtered[variables[variable]['dataset']]

      # drop models with some unavailable experiments
      #unavailable_models = [list(unavailable_experiments.values())]
      #unavailable_models = [val for models in unavailable_experiments.values() for val in models]
      #data = data.sel(model=~data.model.isin(unavailable_models))
      # OR
      # keep the most of the data 
      #data = data.sel(experiment=~data.experiment.isin(['ssp119']))
      #data = data.sel(model=~data.model.isin(unavailable_experiments['ssp245']))
      # TODO must be done from the dataset, not to risk data inconsistencies

      data_90 = data.quantile(0.9, dim='model')
      data_10 = data.quantile(0.1, dim='model')
      data_50 = data.quantile(0.5, dim='model')

      chart = visualizations.Charter(data, range={'top': data_90, 'mean': data_50, 'bottom': data_10}, variable=variable)

      preindustrial_temp = data_50.sel(year=slice(1850, 1900)).mean(dim='year').mean(dim='experiment').item()

      if variable == 'temperature':
        model_count = set(data.model.values.flat)
        chart.plot(title=f'Global temperature projections ({len(model_count)} CMIP6 models)', zero=preindustrial_temp, reference_lines=[0, 2], labels=scenarios['to-visualize']) #ylabel='Temperature difference from 1850-1900'
        #chart.plot(what={'experiment': "ssp126"})
      else:
        maxes = {'Madrid': 35}
        model_count = set(data_ds_filtered.model.values.flat)
        chart.plot(title=f'Maximal temperature (in Czechia) projections ({len(model_count)} CMIP6 models)', ylabel='Max Temperature (°C)', reference_lines=[preindustrial_temp], labels=scenarios['to-visualize'])

except Exception as e: print(f"\nError: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)