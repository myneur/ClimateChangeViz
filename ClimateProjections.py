# About visualized models: https://confluence.ecmwf.int/display/CKB/CMIP6%3A+Global+climate+projections#CMIP6:Globalclimateprojections-Models

# Aggregations based on colab notebook: ecmwf-projects.github.io/copernicus-training-c3s/projections-cmip6.html

# Data-sets: cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6 | Model availability: cds.climate.copernicus.eu | aims2.llnl.gov

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
    'ssp370': "4° = 2× emissions in 2100",
    'ssp534os': "peak at 2040, then steeper decline",
    'ssp585': "5° = 3× emissions in 2075"}
  }
forecast_from = 2015 # Forecasts are actually from 2014. Hindcast untill 2018 or 2019?

# MODELS to be downloaded – when empty, only already downloaded files will be visualized
mark_failing_scenarios = True # Save unavailable experiments not to retry downloading again and again. Clean it in 'metadata/status.json'. 

# DOWNLOAD LOCATION (most models have hundreds MB globally)
# a subfolder according to variable is expected

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


RED = '\033[91m'
RESET = "\033[0m"
YELLOW = '\033[33m'


# UNCOMENT WHAT TO DOWNLOAD, COMPUTE AND VISUALIZE:

DATADIR = os.path.expanduser(f'~/Downloads/ClimateData/')

def main():
  return GlobalTemperature()
  #return maxTemperature()
  #return tropicDaysBuckets()
  #return discovery() # with open('ClimateProjections.py', 'r') as f: exec(f.read())

# VISUALIZATIONS

experiments = list(scenarios['to-visualize'].keys())

def GlobalTemperature():
  variable = 'temperature'; global DATADIR; DATADIR = DATADIR + variable + '/'
  
  models = pd.read_csv('metadata/models.csv')

  not_hot_models = models[models['tcr'] <= 2.2]
  likely_models = not_hot_models[(not_hot_models['tcr'] >= 1.4)]
  #likely_models = models[(models['tcr'] >= 1.4) & (models['tcr'] <= 2.2)]
  not_hot_ecs_models = models[(models['ecs'] <= 4.5)]
  datastore = downloader.DownloaderCopernicus(DATADIR)
  unavailable_experiments = datastore.download(models['model'].values, experiments, 
    skip_failing_scenarios=mark_failing_scenarios, mark_failing_scenarios=mark_failing_scenarios, forecast_from=forecast_from)
  aggregate(var='tas')
  data = loadAggregated()
  data = data['tas']
  data = cleanUpData(data)

  data = normalize(data)
  
  data = models_with_all_experiments(data, drop_experiments=['ssp119'], dont_count_historical=True)
  #data = models_with_all_experiments(data, dont_count_historical=True)
  
  quantile_ranges = quantiles(data, (.1, .5, .9))
  model_set = set(data.model.values.flat)

  observed_t = load_observed_temperature()

  preindustrial_t = preindustrial_temp(quantile_ranges[1])

  chart = visualizations.Charter(variable=variable, 
    title=f'Global temperature projections ({len(model_set)} CMIP6 models)', 
    zero=preindustrial_t, ylimit=[-1,4], reference_lines=[0, 2]
    )

  chart.scatter(observed_t + preindustrial_t, label='measurements') # the observations are already relative to 1850-1900 preindustrial average


  chart.plot([quantile_ranges[0], quantile_ranges[-1]], ranges=True, labels=scenarios['to-visualize'], models=model_set)
  chart.plot(quantile_ranges[1:2], labels=scenarios['to-visualize'], models=model_set)

  not_hot_data = data.sel(model = data.model.isin(not_hot_models['model'].values))
  #chart.plot(quantiles(not_hot_data, [.5]), alpha=.6)

  likely_data = data.sel(model = data.model.isin(likely_models['model'].values))
  #chart.plot(quantiles(likely_data, [.5]), alpha=.3)

  not_hot_ecs_data = data.sel(model = data.model.isin(not_hot_ecs_models['model'].values))
  #chart.plot(quantiles(not_hot_ecs_data, [.5]), alpha=.3)

  m1 = models[models['model'].isin(model_set)]
  print(f"+{m1['tcr'].mean():.2f}° ⌀2100: ALL {len(m1)}× ")
  m2 = models[models['model'].isin(not_hot_data.model.values.flat)]
  print(f"+{m2['tcr'].mean():.2f}° ⌀2100: NOT HOT TCR {len(m2)}× ")
  m3 = models[models['model'].isin(likely_data.model.values.flat)]
  print(f"+{m3['tcr'].mean():.2f}° ⌀2100: LIKELY {len(m3)}× ")
  m4 = models[models['model'].isin(not_hot_ecs_data.model.values.flat)]
  print(f"+{m4['tcr'].mean():.2f}° ⌀2100: NOT HOT ECS {len(m4)}× ")

  #final_t_all = quantile_ranges[1].sel(experiment='ssp245').where(data['year'] > 2090, drop=True).mean().item() - preindustrial_t
  #final_t_likely = likely_t[0].sel(experiment='ssp245').where(data['year'] > 2090, drop=True).mean().item() - preindustrial_t
  #print(f"\nALL models {final_t_likely:.2f} > LIKELY models {final_t_all:.2f} ssp245\n")
  #chart.rightContext([final_t_likely, final_t_all])

  # 3. plot labels explaining the model slection or selections means(s) at 2100 to the right edge of the chart


  chart.show()
  chart.save()
  
  classify_models(data, models, likely_data, likely_models, preindustrial_t)
  
  return data


def classify_models(data, models, likely_data, likely_models, preindustrial_t):
  hot_models = models[(models['tcr'] > 2.2)]['model'].values

  chart = visualizations.Charter(title=f'Global temperature projections ({len(set(data.model.values.flat))} CMIP6 models)', zero=preindustrial_t, reference_lines=[0, 2])

  for model in data.model.values.flat:
    if model in hot_models:
      color = 'red' 
    elif model in likely_models['model'].values: 
      color = 'green'
    else:
      color = 'blue'

    first_decade_t = data.sel(model=model, experiment='historical').where(data['year']<=1860, drop=True).mean().item()-preindustrial_t
    if first_decade_t >= .8:
      print(f'{model} historical hot')
      linewidth=1.3
    elif first_decade_t <=-.6: 
      print(f'{model} historical cold')
      linewidth=1.3
    else:
      linewidth=.5
    chart.plot([data.sel(model=model, experiment = data.experiment.isin(['ssp245', 'historical']))], alpha=1, color=color, linewidth=linewidth)
  
  chart.show()
  chart.save()

  
  # Temperature rise for models ordered by TCR
  models = models.sort_values(by='tcr')
  for model in models['model']:
    try:
      m = data.sel(model=model, experiment='ssp245')

      t = m.where((data['year'] >=2090) & (data['year'] <= 2100), drop=True).mean().item() - preindustrial_t # some models go up to 2200
      tcr = models.loc[models['model'] == model, 'tcr'].values[0]
      print(f"tcr: {tcr:.1f} +{t:.1f}° {model}")
    except:
      pass #print(f'missing {model}')

# monthly: 'monthly_maximum_near_surface_air_temperature', 'tasmax', 'frequency': 'monthly'
def maxTemperature():
  variable = 'max_temperature'; global DATADIR; DATADIR = DATADIR + variable + '/'
  
  #models = md['daily_models']
  models = list(md["all_models"])

  datastore = downloader.DownloaderCopernicus(DATADIR)
  unavailable_experiments = datastore.download(
    models[0:0], 
    ['historical', 'ssp245'], 
    variable='daily_maximum_near_surface_air_temperature', 
    frequency='daily', 
    area=md['area']['cz'], mark_failing_scenarios=mark_failing_scenarios, skip_failing_scenarios=mark_failing_scenarios, forecast_from=forecast_from)
  
  aggregate(var='tasmax')
  
  data = cleanUpData(loadAggregated())
  data = models_with_all_experiments(data, dont_count_historical=True)

  data = data['tasmax']
  quantile_ranges = quantiles(data, (.1, .5, .9))
  maxes = {'Madrid': 35}

  #print(data)
  #print(quantile_ranges[0])

  model_set = set(data.model.values.flat)

  chart = visualizations.Charter(variable=variable,
    title=f'Maximal temperature (in Czechia) projections ({len(model_set)} CMIP6 models)', 
    ylabel='Max Temperature (°C)',
    reference_lines=[preindustrial_temp(quantile_ranges[1]),40]
    )
  
  chart.plot([quantile_ranges[0], quantile_ranges[-1]], ranges='quantile', labels=scenarios['to-visualize'], models=model_set)
  chart.plot(quantile_ranges[1:2], labels=scenarios['to-visualize'], models=model_set)

  chart.show()
  chart.save()

def tropicDaysBuckets():
  variable = 'max_temperature'; global DATADIR; DATADIR = DATADIR + variable + '/'
  
  models = list(md["all_models"])

  datastore = downloader.DownloaderCopernicus(DATADIR)
  unavailable_experiments = datastore.download(
    models[0:0], 
    ['historical', 'ssp245'], 
    variable='daily_maximum_near_surface_air_temperature', 
    frequency='daily', 
    area=md['area']['cz'], mark_failing_scenarios=mark_failing_scenarios, skip_failing_scenarios=mark_failing_scenarios, forecast_from=forecast_from)
  
  aggregate(var='tasmax', stacked=True)
  
  data = cleanUpData(loadAggregated())

  data = models_with_all_experiments(data, dont_count_historical=True)

  model_count = len(set(data.model.values.flat))
  models = data.model.values.flat
  data = data.median(dim='model').max(dim='experiment')      

  chart = visualizations.Charter(variable=variable, models=models,
    title=f'Tropic days (in Czechia) projection ({model_count} CMIP6 models)', 
    subtitle="When no decline of emissions till 2050 (ssp245 scenario)", 
    ylabel='Tropic days annualy',
    marker=forecast_from)
    
  chart.stack(data)
  chart.show()
  chart.save()

def discovery():
  variable = 'discovery'; global DATADIR; DATADIR = DATADIR + variable + '/'
  #models = ["CIESM", "CMCC-CM2-SR5", "FGOALS-g3", "NorESM2-MM", "MPI-ESM-1-2-HAM", "INM-CM4-8", "TaiESM1", "HadGEM3-GC31-MM", "NorESM2-LM", "MIROC-ES2L", "CAS-ESM2-0", "BCC-CSM2-MR", "ACCESS-CM2", "NESM3", "E3SM-1-0", "INM-CM5-0", "KACE-1-0-G", "FGOALS-f3-L", "CNRM-ESM2-1", "MRI-ESM2-0", "CESM2-WACCM-FV2", "CanESM5", "MPI-ESM1-2-HR", "CNRM-CM6-1", "EC-Earth3", "IITM-ESM", "MIROC6", "GISS-E2-2-G", "EC-Earth3-Veg", "CESM2", "CNRM-CM6-1-HR", "SAM0-UNICON", "GISS-E2-1-H", "MPI-ESM1-2-L", "AWI-CM-1-1-MR", "MCM-UA-1-0", "GFDL-CM4", "ACCESS-ESM1-5", "HadGEM3-GC31-LL", "CAMS-CSM1-0", "UKESM1-0-LL", "MPI-ESM1-2-LR", "FIO-ESM-2-0", "NorCPM1", "CanESM5-CanOE", "GFDL-ESM4", "IPSL-CM6A-LR", "BCC-ESM1", "CESM2-WACCM"]
  #models = md['smaller_models4testing']
  
  datastore = downloader.DownloaderCopernicus(DATADIR)
  unavailable_experiments = datastore.download(
    ['canesm5'], 
    ['ssp245'], 
    variable='surface_temperature', frequency='monthly',
    #variable='daily_maximum_near_surface_air_temperature', frequency='daily',
    #area=md['area']['cz'],
    start=2020,
    forecast_from=2020,
    end=2030,
    skip_failing_scenarios=True, mark_failing_scenarios=True, 
    fileformat='zip' #zip tgz netcdf grib
    )
  aggregate(var='tas')
  data = loadAggregated()
  
  chart = visualizations.Charter(variable=variable)
  #chart.plot(data, what={'experiment': None})
  
  #data = data['tas']
  #quantile_ranges = quantiles(data, (.1, .5, .9))
  chart.plot([data])
  
  #chart.plotDiscovery(data, what={'experiment':'ssp245'})
  #chart.plot(data, what={'experiment':'historical'})
  chart.show()
  return data

def load_observed_temperature():
  # OBSERVATIONS from https://climate.metoffice.cloud/current_warming.html
  observations = [pd.read_csv(f'data/{observation}.csv') for observation in ['gmt_HadCRUT5', 'gmt_NOAAGlobalTemp', 'gmt_Berkeley Earth']]
  observations = pd.concat(observations)
  observation = observations[['Year', observations.columns[1]]].groupby('Year').mean()
  return observation[observation.index <= 2023]

def history():
  # not working yet: https://web.archive.org/web/20240516185454/https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=form
  unavailable_experiments = downloader.reanalysis()


# COMPUTATION

def quantiles(data, quantiles):
  quantilized = []
  for q in quantiles:
    quantilized.append(data.quantile(q, dim='model'))
  
  return quantilized

  quantilized = xr.concat([data.quantile(q, dim='model') for q in quantiles], dim='quantile')
  quantilized['quantile'] = quantiles  # name coordinates

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
    #<variable_id>_<table_id>_<source_id>_<experiment_id>_<variant_label>_<grid_label>_<time_range>.nc
    model, experiment, variant, grid, time = filename.split('_')[2:7]
    da_yr.to_netcdf(path=f'{DATADIR}cmip6_agg_{exp}_{mod}_{variant}_{grid}_{time}.nc')
    #da_yr.to_netcdf(path=f'{DATADIR}cmip6_agg_{exp}_{mod}_{str(da_yr.year[0].values)}.nc')

  #except OSError as e: print(f"\n{RED}Error loading model:{RESET} {type(e).__name__}: {e}")
  except Exception as e: print(f"\n{RED}Error aggregating {filename}:{RESET} {type(e).__name__}: {e}"); traceback.print_exc()

def aggregate(stacked=None, var='tas'):
  dataFiles = list()
  for i in glob(f'{DATADIR}{var}*.nc'):
      dataFiles.append(os.path.basename(i))
  for filename in dataFiles:
    model, experiment, variant, grid, time = filename.split('_')[2:7]
    try:
      candidate_files = [f for f in os.listdir(DATADIR) if f.startswith(f'cmip6_agg_{experiment}_{model}_{variant}_{grid}_{time}')] 
      # NOTE it expects the same filename strucutre, which seems to be followed, but might be worth checking for final run (or regenerating all)
      if reaggregate or not len(candidate_files):
        print('.', end='')
        geog_agg(filename, var=var, buckets=stacked)

    except Exception as e: print(f"Error in {filename}: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)
  print()

def loadAggregated(models=None, experiments=None, unavailable_experiments=None, wildcard=''):
  filename_pattern = f'{DATADIR}cmip6_agg_*{wildcard}*.nc'
  filenames = glob(os.path.join(DATADIR, filename_pattern))
  modelex = {}
  for filename in filenames: 
    filename = filename.split('/')[-1]
    
    model, experiment, variant, grid, time = filename.split('_')[2:7]
    key = f'{model}_{experiment}_{time}'
    if not key in modelex:
      modelex[key] = set()
    else:
      print(f'{YELLOW}duplicate{RESET} {key}: {set(variant)|modelex[key]}')

    modelex[key] |= {variant}

  print('Opening aggregations')
  try:
    data_ds = None
    data_ds = xr.open_mfdataset(filename_pattern, combine='nested', concat_dim='model') # when problems with loading # data_ds = xr.open_mfdataset(f'{DATADIR}cmip6_agg_*.nc')
    data_ds.load()

    return data_ds

    

    
  
    for i in glob(filename_pattern):
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
    '''
    for model in data.model.values.flat:
      for experiment in data.experiment.values.flat:
        year_range = np.arange(1850, 2101)
        ds = data.sel(model=model, experiment=experiment)
        years = ds['year'].values
        missing_years = np.setdiff1d(year_range, years)
        unique_years, counts = np.unique(years, return_counts=True)
        duplicate_years = unique_years[counts > 1]

        if missing_years.size > 0:
            print(f"{YELLOW}Missing years:{RESET} {missing_years} in {model} {experiment}")
        if duplicate_years.size > 0:
            print(f"{YELLOW}Duplicate years:{RESET} {duplicate_years} in {model} {experiment}")
    '''
    # merging data series of different periods for the same model
    #models = data.model.values
    #if len(models) > len(set(models)):
    data = data.groupby('model').mean('model')

    return data

  except Exception as e: print(f"Error: {type(e).__name__}: {e}"); traceback.print_exc()  

def normalize(data):
  
  model_mean = data.sel(experiment='historical').sel(year=slice(forecast_from-20, forecast_from-1)).mean(dim='year')

  global_mean = model_mean.mean(dim='model').item()

  normalization_offset = model_mean - global_mean
  normalization_offset_expanded = normalization_offset.expand_dims(dim={'year': data.year}).transpose('model', 'year')

  normalized_data = data - normalization_offset_expanded

  #print(f"NORMALIZED {normalized_data.sel(experiment='ssp126', model=normalized_data.model.values.flat[0], year=slice(2014,2015))}")

  return normalized_data


  #raise(NotImplementedError)

def models_with_all_experiments(data, dont_count_historical=False, drop_experiments=None):

  if drop_experiments:
    data = data.sel(experiment =~ data.experiment.isin(drop_experiments))

  experiments = set(data.experiment.values.flat)
  if dont_count_historical:
    experiments = experiments - {'historical'} 
  
  models_by_experiment = {experiment: data.sel(experiment=experiment).dropna(dim='model', how='all').model.values for experiment in experiments}
  available = [models[1] for models in models_by_experiment.items()]
  intersection = set(available[0]).intersection(*available[1:]) if available else []

  data = data.sel(model = data.model.isin(list(intersection)))
  data = data.dropna(dim='model', how='all')
  
  remained = set(data.model.values.flat)

  print(f"\n{len(remained)} remained") #: {' '.join(remained)}
  for experiment in models_by_experiment.items():
    print(f"{len(experiment[1])}⨉ {experiment[0]}, except: {sorted(set(experiment[1])-remained)}")

  return data


def preindustrial_temp(data):
  return data.sel(year=slice(1850, 1900)).mean(dim='year').mean(dim='experiment').item()

md = util.loadMD('model_md')

# RUN the function defined in the 'run' at the top
try:
  result = main()
  #result = globals()[run]()
except Exception as e: print(f"\nError: {type(e).__name__}: {e}"); traceback.print_exc()