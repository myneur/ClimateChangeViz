# API keys from cds.climate.copernicus.eu must be in ~/.cdsapirc

# About visualized models: https://confluence.ecmwf.int/display/CKB/CMIP6%3A+Global+climate+projections#CMIP6:Globalclimateprojections-Models

# Calculations based on colab notebook: ecmwf-projects.github.io/copernicus-training-c3s/projections-cmip6.html

# Data-sets: cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6 | Models list/search on: aims2.llnl.gov

# TODO
# 2. Include all models with either by best estimate of AR6 or TCR Screen (likely) 1.4-2.2º as guided on https://www.nature.com/articles/d41586-022-01192-2.epdf
# 3. plot labels explaining the model slection or selections means(s) at 2100 to the right edge of the chart
# 5. historical data (measurements from the constant set of stations normalized to the lastest, most complete set, to be independent on addtions)


# WHAT
reaggregate = False # compute aggregations regardles if they already exist


# LEGEND
scenarios = { # CO2 emissions scenarios charted on https://www.carbonbrief.org/cmip6-the-next-generation-of-climate-models-explained/
  'to-visualize': {
    'historical': "hindcast",
    'ssp119': "1.5° = carbon neutral in 2050", 
    'ssp126': "2° = carbon neutral in 2075",
    'ssp245': "3° = no decline till ½ millenia"},
  'out-of-focus': {
    'ssp534os': "peak at 2040, then steeper decline",
    'ssp570': "4° = 2× emissions in 2100",
    'ssp585': "5° = 3× emissions in 2075"}
  }
forecast_from = 2015 # Forecasts from 2014 or 2015? Hindcast untill 2018 or 2019?

# MODELS to be downloaded – when empty, only already downloaded files will be visualized
mark_failing_scenarios = True # Save unavailable experiments not to retry downloading again and again. Clean it in 'metadata/status.json'. 

# DOWNLOAD LOCATION (most models have hundreds MB globally)
# a subfolder according to variable is expected
DATADIR = f'/Users/myneur/Downloads/ClimateData/'

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
from util import debug
import traceback



# UNCOMENT WHAT TO DOWNLOAD, COMPUTE AND VISUALIZE:

def main():
  #return GlobalTemperature()
  return maxTemperature()
  #return tropicDaysBuckets()
  
  #return discovery() # with open('ClimateProjections.py', 'r') as f: exec(f.read())


# VISUALIZATIONS

experiments = list(scenarios['to-visualize'].keys())

def GlobalTemperature():
  variable = 'temperature'; global DATADIR; DATADIR = DATADIR + variable + '/'
  
  #models = md[variable]
  models = md["all_models"].keys()

  #models = set(); for s in md["by_states"].items(): models |= set(s[1])

  unavailable_experiments = downloader.download(models, experiments, DATADIR, mark_failing_scenarios=mark_failing_scenarios, skip_failing_scenarios=mark_failing_scenarios, forecast_from=forecast_from)
  aggregate(var='tas')
  data = loadAggregated()
  data = data['tas']
  data = cleanUpData(data)
  
  #data = models_with_all_experiments(data, drop_experiments=['ssp119'])
  data = models_with_all_experiments(data)
  
  quantile_ranges = quantiles(data, (.1, .5, .9))

  chart = visualizations.Charter(variable=variable, 
    title=f'Global temperature projections ({len(set(data.model.values.flat))} CMIP6 models)', )
  chart.plot(
    data, ranges=quantile_ranges, 
    zero=preindustrial_temp(quantile_ranges[1]), 
    reference_lines=[0, 2], labels=scenarios['to-visualize']) #ylabel='Temperature difference from 1850-1900'
  chart.show()
  chart.save()
  return data

# monthly: 'monthly_maximum_near_surface_air_temperature', 'tasmax', 'frequency': 'monthly'
def maxTemperature():
  variable = 'max_temperature'; global DATADIR; DATADIR = DATADIR + variable + '/'
  
  #models = md['daily_models']
  models = list(md["all_models"].keys())

  unavailable_experiments = downloader.download(
    models[0:0], 
    ['historical', 'ssp245'], DATADIR, 
    variable='daily_maximum_near_surface_air_temperature', 
    frequency='daily', 
    area=md['area']['cz'], mark_failing_scenarios=mark_failing_scenarios, skip_failing_scenarios=mark_failing_scenarios, forecast_from=forecast_from)
  
  aggregate(var='tasmax')
  
  data = cleanUpData(loadAggregated())
  data = models_with_all_experiments(data)

  data = data['tasmax']
  quantile_ranges = quantiles(data, (.1, .5, .9))
  maxes = {'Madrid': 35}

  chart = visualizations.Charter(variable=variable,
    title=f'Maximal temperature (in Czechia) projections ({len(set(data.model.values.flat))} CMIP6 models)', 
    ylabel='Max Temperature (°C)')
  
  chart.plot(
    data, ranges=quantile_ranges, 
    reference_lines=[preindustrial_temp(quantile_ranges[1]),40], labels=scenarios['to-visualize'])
  chart.show()
  chart.save()

def tropicDaysBuckets():
  variable = 'max_temperature'; global DATADIR; DATADIR = DATADIR + variable + '/'
  
  models = list(md["all_models"].keys())

  unavailable_experiments = downloader.download(
    models[0:0], 
    ['historical', 'ssp245'], DATADIR, 
    variable='daily_maximum_near_surface_air_temperature', 
    frequency='daily', 
    area=md['area']['cz'], mark_failing_scenarios=mark_failing_scenarios, skip_failing_scenarios=mark_failing_scenarios, forecast_from=forecast_from)
  
  aggregate(var='tasmax', stacked=True)
  
  data = cleanUpData(loadAggregated())

  data = models_with_all_experiments(data)

  model_count = len(set(data.model.values.flat))
  models = data.model.values.flat
  data = data.median(dim='model').max(dim='experiment')      

  chart = visualizations.Charter(variable=variable, models=models,
    title=f'Tropic days (in Czechia) projection ({model_count} CMIP6 models)', 
    subtitle="When no decline of emissions till 2050 (ssp245 scenario)", 
    ylabel='Tropic days annualy')
  chart.stack(
    data, 
    marker=forecast_from)
  chart.show()
  chart.save()

def discovery():
  variable = 'discovery'; global DATADIR; DATADIR = DATADIR + variable + '/'
  #models = ["CIESM", "CMCC-CM2-SR5", "FGOALS-g3", "NorESM2-MM", "MPI-ESM-1-2-HAM", "INM-CM4-8", "TaiESM1", "HadGEM3-GC31-MM", "NorESM2-LM", "MIROC-ES2L", "CAS-ESM2-0", "BCC-CSM2-MR", "ACCESS-CM2", "NESM3", "E3SM-1-0", "INM-CM5-0", "KACE-1-0-G", "FGOALS-f3-L", "CNRM-ESM2-1", "MRI-ESM2-0", "CESM2-WACCM-FV2", "CanESM5", "MPI-ESM1-2-HR", "CNRM-CM6-1", "EC-Earth3", "IITM-ESM", "MIROC6", "GISS-E2-2-G", "EC-Earth3-Veg", "CESM2", "CNRM-CM6-1-HR", "SAM0-UNICON", "GISS-E2-1-H", "MPI-ESM1-2-L", "AWI-CM-1-1-MR", "MCM-UA-1-0", "GFDL-CM4", "ACCESS-ESM1-5", "HadGEM3-GC31-LL", "CAMS-CSM1-0", "UKESM1-0-LL", "MPI-ESM1-2-LR", "FIO-ESM-2-0", "NorCPM1", "CanESM5-CanOE", "GFDL-ESM4", "IPSL-CM6A-LR", "BCC-ESM1", "CESM2-WACCM"]
  #models = md['smaller_models4testing']
  models = list(md["all_models"].keys())
  downloader.download(
    ['ACCESS-CM2','CESM2', 'CIESM', 'CMCC-CM2'], 
    [ 'ssp245'], 
    DATADIR, 
    variable='near_surface_air_temperature', frequency='monthly',
    #variable='daily_maximum_near_surface_air_temperature', frequency='daily',
    area=md['area']['cz'],
    start=2000,
    end=2030,
    forecast_from=2010,
    skip_failing_scenarios=True, mark_failing_scenarios=True, 
    )
  aggregate(var='tas')
  data = loadAggregated()
  
  chart = visualizations.Charter(variable=variable)
  #chart.plot(data, what={'experiment': None})
  
  #data = data['tas']
  #quantile_ranges = quantiles(data, (.1, .5, .9))
  #chart.plot(data, ranges=quantile_ranges, what='mean')
  
  chart.plot(data, what={'experiment':'ssp245'})
  #chart.plot(data, what={'experiment':'historical'})
  chart.show()
  return data


def history():
  # not working yet: https://web.archive.org/web/20240516185454/https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=form
  unavailable_experiments = downloader.reanalysis()



# COMPUTATION

def quantiles(data, quantiles):
  quantilized = []
  for q in quantiles:
    quantilized.append(data.quantile(q, dim='model'))
  
  return quantilized

def create_buckets(da_agg):
  t30 = ((da_agg >= (30+K)) & (da_agg < (35+K))).resample(time='YE').sum(dim='time')
  t35 = (da_agg >= (35+K)).resample(time='YE').sum(dim='time') # t35 = ((da_agg >= (35+K)) & (da_agg < np.inf)).resample(time='YE').sum(dim='time')
  da_yr = xr.Dataset(
    {'bucket': (('bins', 'time'), [t30, t35])},
    coords={'bins': ['30-35', '35+'],'time': t30.time})

  da_yr = da_yr.assign_coords(year=da_yr['time'].dt.year)
  da_yr = da_yr.drop_vars('time')
  da_yr = da_yr.rename({'time': 'year'})

  return da_yr


K = 273.15 # Kelvins
def geog_agg(filename, var='tas', buckets=None, area=None):
  try:
    ds = xr.open_dataset(f'{DATADIR}{filename}')

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
    
    # AGGREGATE SPATIALLY
    
    # MAX
    if var == 'tasmax':
      da_agg = da.max([lat, lon])
    
    # AVG
    else:
      # Weight as longitude gird shrinks with latitude
      weights = np.cos(np.deg2rad(da[lat]))
      weights.name = "weights"
      da_weighted = da.weighted(weights)
      da_agg = da_weighted.mean([lat, lon])

    # AGGREGATE TIME

    if buckets:
      da_yr = create_buckets(da_agg)
    else:
      # MAX
      if var == 'tasmax':
        da_yr = da_agg.groupby('time.year').max()

      # AVG
      else: 
        da_yr = da_agg.groupby('time.year').mean()

      da_yr = da_yr - K # °C
    
    # CONTEXT
    da_yr = da_yr.assign_coords(model=mod)
    da_yr = da_yr.expand_dims('model')
    da_yr = da_yr.assign_coords(experiment=exp)
    da_yr = da_yr.expand_dims('experiment')
    #ds.attrs['height'] = ...

    # SAVE
    model, experiment, variant, grid, time = filename.split('_')[2:7]
    da_yr.to_netcdf(path=f'{DATADIR}cmip6_agg_{exp}_{mod}_{variant}_{grid}_{time}.nc')
    #da_yr.to_netcdf(path=f'{DATADIR}cmip6_agg_{exp}_{mod}_{str(da_yr.year[0].values)}.nc')

  except Exception as e: print(f"\nError aggregating {filename}: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)

def aggregate(stacked=None, var='tas'):
  dataFiles = list()
  for i in glob(f'{DATADIR}tas*.nc'):
      dataFiles.append(os.path.basename(i))
  for filename in dataFiles:
    model, experiment, variant, grid, time = filename.split('_')[2:7]
    try:
      candidate_files = [f for f in os.listdir(DATADIR) if f.endswith('.nc') and f.startswith(f'cmip6_agg_{experiment}_{model}_{variant}_{grid}_{time}')] 
      # NOTE it expects the same filename strucutre, which seems to be followed, but might be worth checking for final run (or regenerating all)
      if reaggregate or not len(candidate_files):
        print('.', end='')
        geog_agg(filename, var=var, buckets=stacked)
      print()

    except Exception as e: print(f"Error in {filename}: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)

def loadAggregated(models=None, experiments=None, unavailable_experiments=None, wildcard=''):
  print('Opening aggregations')
  try:
    data_ds = None
    data_ds = xr.open_mfdataset(f'{DATADIR}cmip6_agg_*{wildcard}*.nc', combine='nested', concat_dim='model') # when problems with loading # data_ds = xr.open_mfdataset(f'{DATADIR}cmip6_agg_*.nc')
    data_ds.load()

    return data_ds

    files_to_load = f'{DATADIR}cmip6_agg_*{wildcard}*.nc'

    
  
    for i in glob(files_to_load):
        filename = os.path.abspath(i) # os.path.basename(i)
        print(filename)
        new_ds = xr.open_dataset(filename)
        if data_ds is None:
            data_ds = new_ds
        else:
            #data_ds = xr.combine_by_coords([data_ds, new_ds])
            #data_ds = xr.combine_nested([data_ds, new_ds], concat_dim=['experiment', 'model', 'bins', 'year'], combine_attrs='override')
            data_ds = xr.combine_nested([data_ds, new_ds], concat_dim=['model'])

  except Exception as e: 
    print(f"Error: {type(e).__name__}: {e}"); 
    traceback.print_exc(limit=1)

  not_read = set(models)-set(data_ds.model.values.flat) if data_ds else print("Nothing read at all")
  if not_read: print("\nNOT read: '" + ' '.join(map(str, not_read)) +"'")

  #print(len(set(data_ds.sel(experiment='ssp126').model.values.flat)))
  #print(sorted(set(data_ds.sel(experiment='ssp126').model.values.flat)))

  return data_ds

def cleanUpData(data):
  # removing historical data before 2014, because some models can include them despite request
  try:
    def filter_years(ds):
      if 'historical' in ds['experiment']:
        ds = ds.sel(year=ds['year'] < forecast_from)
      return ds
    data = data.groupby('experiment').map(filter_years) #, squeeze=True
  
    # merging data series of different periods for the same model
    #models = data.model.values
    #if len(models) > len(set(models)):
    data = data.groupby('model').mean('model')
    return data

  except Exception as e: print(f"Error: {type(e).__name__}: {e}"); traceback.print_exc()  

def models_with_all_experiments(data, drop_experiments=None):

  if drop_experiments:
    data = data.sel(experiment =~ data.experiment.isin(drop_experiments))

  experiments = set(data.experiment.values.flat)
  experiments = experiments - {'historical'} 
  availability = []
  for experiment in experiments:

    available = set(data.sel(experiment=experiment).dropna(dim='model', how='all').model.values)
    availability.append(available)
    #print(experiment)
    #print(len(available), ' '.join(available))
    print(f'\n{experiment}: {len(available)} models')
    print(available)

  keep = availability[0]
  for a in availability[1:]:
    keep &= a

  keep = list(keep)
  data = data.sel(model = data.model.isin(keep))
  data = data.dropna(dim='model', how='all')
  
  remained = set(data.model.values.flat)
  print(f"remained: {len(remained)} models")
  print(remained)

  return data


def preindustrial_temp(data):
  return data.sel(year=slice(1850, 1900)).mean(dim='year').mean(dim='experiment').item()

md = util.loadMD('model_md')

# RUN the function defined in the 'run' at the top
try:
  result = main()
  #result = globals()[run]()
except Exception as e: print(f"\nError: {type(e).__name__}: {e}"); traceback.print_exc()