# API keys from cds.climate.copernicus.eu must be in ~/.cdsapirc

# About visualized models: https://confluence.ecmwf.int/display/CKB/CMIP6%3A+Global+climate+projections#CMIP6:Globalclimateprojections-Models

# Calculations based on colab notebook: ecmwf-projects.github.io/copernicus-training-c3s/projections-cmip6.html

# Data-sets: cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6 | Models list/search on: aims2.llnl.gov

# TODO
# unziping to different folders not to interfere
# 1. Buckets median
# 2. Include all from aims2.llnl.gov or just some variants from the family (e. g. High resolution and not Low): https://www.nature.com/articles/d41586-022-01192-2.epdf
# 3. Dealing with hot models ECS>4.5: shade range in 2 transparencies? AFAIK there is no consensus on their probability: https://www.carbonbrief.org/cmip6-the-next-generation-of-climate-models-explained/
# 4. aggregated models by state as series to show how big is the coverage of the models
# 5. historical data (measurements from the constant set of stations normalized to the lastest, most complete set, to be independent on addtions)
# 6. plot range of forecast to the right edge of the chart

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
  #maxTemperature()
  tropicDaysBuckets()
  #GlobalTemperature()
  #discovery()


# VISUALIZATIONS

experiments = scenarios['to-visualize'].keys()

def GlobalTemperature():
  variable = 'temperature'; global DATADIR; DATADIR = DATADIR + variable + '/'
  
  models = md[variable]
  models = md["model_resolutions"].keys()
  unavailable_experiments = downloader.download(models, experiments, DATADIR, mark_failing_scenarios=mark_failing_scenarios, forecast_from=forecast_from)
  
  aggregate()
  
  data_ds_filtered = cleanUpData(loadAggregated(models))
  
  data_ds_filtered = filter_to_models_with_most_experiments(data_ds_filtered, unavailable_experiments)
  #data_ds_filtered = filter_to_models_with_all_experiments(data_ds_filtered, unavailable_experiments)
  
  data_50, data_90, data_10 = averageModels('tas', data_ds_filtered, unavailable_experiments)

  chart = visualizations.Charter(variable=variable, 
    title=f'Global temperature projections ({len(set(data_ds_filtered.model.values.flat))} CMIP6 models)', )
  chart.plot(
    data_ds_filtered, limits={'top': data_90, 'mean': data_50, 'bottom': data_10}, 
    zero=preindustrial_temp(data_50), 
    reference_lines=[0, 2], labels=scenarios['to-visualize']) #ylabel='Temperature difference from 1850-1900'
  #chart.plot(what={'experiment': "ssp126"})
  return data_ds_filtered

# monthly: 'monthly_maximum_near_surface_air_temperature', 'tasmax', 'frequency': 'monthly'
def maxTemperature():
  variable = 'max_temperature'; global DATADIR; DATADIR = DATADIR + variable + '/'
  
  models = md[variable]
  models = md["model_resolutions"].keys()

  unavailable_experiments = downloader.download(models, 
    ['historical', 'ssp245'], DATADIR, 
    variable='daily_maximum_near_surface_air_temperature', 
    frequency='daily', 
    area=md['area']['cz'], mark_failing_scenarios=mark_failing_scenarios, forecast_from=forecast_from)
  
  aggregate()
  
  data_ds_filtered = cleanUpData(loadAggregated(models))
  data_50, data_90, data_10 = averageModels('tasmax', data_ds_filtered, unavailable_experiments)
  maxes = {'Madrid': 35}

  chart = visualizations.Charter(variable=variable,
    title=f'Maximal temperature (in Czechia) projections ({len(set(data_ds_filtered.model.values.flat))} CMIP6 models)', 
    ylabel='Max Temperature (°C)')
  
  chart.plot(
    data_ds_filtered, limits={'top': data_90, 'mean': data_50, 'bottom': data_10}, 
    reference_lines=[preindustrial_temp(data_50)], labels=scenarios['to-visualize'])

def tropicDaysBuckets():
  variable = 'max_temperature'; global DATADIR; DATADIR = DATADIR + variable + '/'
  
  models = md[variable]
  models = md["model_resolutions"].keys()

  unavailable_experiments = downloader.download(models, 
    ['historical', 'ssp245'], DATADIR, 
    variable='daily_maximum_near_surface_air_temperature', 
    frequency='daily', 
    area=md['area']['cz'], mark_failing_scenarios=mark_failing_scenarios, forecast_from=forecast_from)
  
  aggregate(stacked=True)
  
  data = cleanUpData(loadAggregated(models))
  model_count = len(set(data.model.values.flat))
  #data = data.median(dim='model')

  chart = visualizations.Charter(variable=variable, 
    title=f'Tropic days (in Czechia) projection ({model_count} CMIP6 models)', 
    subtitle="When no decline of emissions till 2050 (ssp245 scenario)", 
    ylabel='Tropic days annualy')
  chart.stack(
    data, 
    marker=forecast_from)

def discovery():
  variable = 'discovery'; global DATADIR; DATADIR = DATADIR + variable + '/'
  
  models = md["model_resolutions"].keys(); #["GISS-E2-1-G", "CESM2-FV2", "E3SM-1-1", "E3SM-1-1-ECA", "AWI-ESM-1-1-LR", "MIROC-ES2H", "CMCC-ESM2", "KIOST-ESM", "CMCC-CM2-HR4", "IPSL-CM5A2-INCA", "EC-Earth3-CC", "EC-Earth3-Veg-LR", "EC-Earth3-AerChem"]
  scenarios = ['ssp126']
  downloader.download(models, scenarios, DATADIR, mark_failing_scenarios=mark_failing_scenarios, forecast_from=forecast_from)
  aggregate()
  data = loadAggregated(models)
  chart = visualizations.Charter(variable=variable)
  chart.plot(data, what={'experiment': scenarios})


def history():
  # not working yet: https://web.archive.org/web/20240516185454/https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=form
  unavailable_experiments = downloader.reanalysis()



# COMPUTATION

def averageModels(var, data_ds_filtered, unavailable_experiments):
  # TEMPERATURES
  data = data_ds_filtered[var]

  data_90 = data.quantile(0.9, dim='model')
  data_10 = data.quantile(0.1, dim='model')
  data_50 = data.quantile(0.5, dim='model')
  return data_50, data_90, data_10

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
    da_yr.to_netcdf(path=f'{DATADIR}cmip6_agg_{exp}_{mod}_{str(da_yr.year[0].values)}.nc')

  except Exception as e: print(f"\nError aggregating {fn}: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)

def aggregate(stacked=None):
  dataFiles = list()
  for i in glob(f'{DATADIR}tas*.nc'):
      dataFiles.append(os.path.basename(i))

  for filename in dataFiles:
    model, experiment, variant, grid, time = filename.split('_')[2:7]
    try:
      candidate_files = [f for f in os.listdir(DATADIR) if f.endswith('.nc') and f.startswith(f'cmip6_agg_{experiment}_{model}_{grid}_{time}')] 
      # NOTE it expects the same filename strucutre, which seems to be followed, but might be worth checking for final run (or regenerating all)
      if reaggregate or not len(candidate_files):
        geog_agg(filename, buckets=stacked)

    except Exception as e: print(f"Error in {filename}: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)

def loadAggregated(models):
  print('Opening aggregations')
  data_ds = xr.open_mfdataset(f'{DATADIR}cmip6_agg_*.nc', combine='nested', concat_dim='model')
  #data_ds = xr.open_mfdataset(f'{DATADIR}cmip6_agg_*.nc')
  data_ds.load()
  # combined_ds = xr.combine_by_coords([data_ds1, data_ds2])
  #combined_ds.load()

  not_read = set(models)-set(data_ds.model.values.flat)
  if not_read: print("\nNOT read: '" + ' '.join(map(str, not_read)) +"'")

  return data_ds

def cleanUpData(data_ds):
  # removing historical data before 2014, because some models can include them despite request
  try:
    def filter_years(ds):
      if 'historical' in ds['experiment']:
        ds = ds.sel(year=ds['year'] < forecast_from)
      return ds
    data_ds_filtered = data_ds.groupby('experiment').map(filter_years) #, squeeze=True
  
    # merging data series of different periods for the same model
    # TODO if len(models) > len(unique_models)
    data_ds_filtered = data_ds_filtered.groupby('model').mean('model')
    return data_ds_filtered

  except Exception as e: print(f"Error in {filename}: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)

def filter_to_models_with_all_experiments(data, missing_models):
  # TODO must be done from the dataset, not to risk data inconsistencies potentially introduced by unavailable_experiments

  miss = set(missing_models['ssp119']) | se(missing_models['ssp126']) | set(missing_models['ssp245'])
  
  unavailable_models = [val for models in unavailable_experiments.values() for val in models]
  return data.sel(model =~ data.model.isin(unavailable_models))
  

def filter_to_models_with_most_experiments(data, missing_models):
  # TODO must be done from the dataset, not to risk data inconsistencies potentially introduced by unavailable_experiments
  
  # drop experiment that we know have the least of models
  data_ds_filtered = data.sel(experiment =~ data.experiment.isin(['ssp119']))


  avail1 = set(data_ds_filtered.sel(experiment='ssp126').dropna(dim='model', how='all').model.values)
  print("126", len(avail1), avail1)
  avail2 = set(data_ds_filtered.sel(experiment='ssp245').dropna(dim='model', how='all').model.values)
  print("245", len(avail2), avail2)
  keep = list(avail1 & avail2)
  print("245", len(keep), keep)

  intersection = data_ds_filtered.sel(model = data_ds_filtered.model.isin(keep))
  intersection = intersection.dropna(dim='model', how='all')
  remained = intersection.model.values
  print("remained", len(remained), remained)
  return intersection


def WIP_filter_to_models_with_most_experiments(data):
  # TODO must be done from the dataset, not to risk data inconsistencies potentially introduced by unavailable_experiments
  
  # drop experiment that we know have the least of models
  data_ds_filtered = data.sel(experiment =~ data.experiment.isin(['ssp119']))
  ssp126_models = data.sel(experiment =~ data.experiment.isin(['ssp126'])).model.values.flat
  
  # drop missing experiments in the other one
  return data_ds_filtered.sel(model =~ data_ds_filtered.model.isin(ssp126_models))



def preindustrial_temp(data):
  return data.sel(year=slice(1850, 1900)).mean(dim='year').mean(dim='experiment').item()

md = util.loadMD('model_md')

# RUN the function defined in the 'run' at the top
try:
  main()
  #result = globals()[run]()
except Exception as e: print(f"\nError: {type(e).__name__}: {e}"); traceback.print_exc()#limit=1)