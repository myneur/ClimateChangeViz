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
    'ssp245': "3° = no decline till 2050"},
  'out-of-focus': {
    'ssp370': "4° = 2× emissions in 2100",
    'ssp534os': "peak at 2040, then steeper decline",
    'ssp585': "5° = 3× emissions in 2075"}
  }
forecast_from = 2015 # Forecasts are actually from 2014. Hindcast untill 2018 or 2019?

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

DATADIR = os.path.expanduser(f'~/Downloads/ClimateData/') # DOWNLOAD LOCATION (most models have hundreds MB globally)
datastore = None
#datastore = downloader.DownloaderCopernicus(DATADIR, skip_failing_scenarios=True, mark_failing_scenarios=True)
#mark_failing_scenarios = True to save unavailable experiments not to retry downloading again and again. Clean it in 'metadata/status.json'. 

def main():
  GlobalTemperature()
  #GlobalTemperature(drop_experiments=['ssp119'])
  #return maxTemperature(frequency='monthly')
  #maxTemperature(frequency='daily')
  #tropicDaysBuckets()
  #return discovery() # with open('ClimateProjections.py', 'r') as f: exec(f.read())

# VISUALIZATIONS

experiments = list(scenarios['to-visualize'].keys())

def GlobalTemperature(drop_experiments=None):
    variable = 'temperature'; 
    global DATADIR; DATADIR = os.path.join(DATADIR, variable, '')
    try:
        models = pd.read_csv('metadata/models.csv')
        observed_t = load_observed_temperature()
        
        if datastore:
            datastore.DATADIR = DATADIR
            datastore.download(models['model'].values, experiments, forecast_from=forecast_from)
        
        aggregate(var='tas')
        data = loadAggregated()
        
        data = data['tas']
        data = cleanUpData(data)
        data = normalize(data)

        classify_models(data, models)

        data = models_with_all_experiments(data, drop_experiments=drop_experiments, dont_count_historical=True)
        
        preindustrial_t = preindustrial_temp(data)

        quantile_ranges = quantiles(data, (.1, .5, .9))
        #preindustrial_t = preindustrial_temp(quantile_ranges[1])

        model_set = set(data.model.values.flat)

        chart = visualizations.Charter(
            title=f'Global temperature projections ({len(model_set)} CMIP6 models)', 
            zero=preindustrial_t, yticks=[0, 1.5, 2, 3], ylimit=[-1,4], reference_lines=[0, 2], yformat=lambda y, i: f"{'+' if y-preindustrial_t>0 else ''}{y-preindustrial_t:.1f} °C" 
            )

        chart.scatter(observed_t + preindustrial_t, label='measurements') # the observations are already relative to 1850-1900 preindustrial average

        chart.plot([quantile_ranges[0], quantile_ranges[-1]], ranges=True, labels=scenarios['to-visualize'], models=model_set)
        chart.plot(quantile_ranges[1:2], labels=scenarios['to-visualize'], models=model_set)


        chart.show()
        chart.save(tag=f'{variable}_{len(model_set)}')
        
        return data
    except OSError as e: print(f"{RED}Error: {type(e).__name__}: {e}{RESET}") 
    except Exception as e: print(f"{RED}Error: {type(e).__name__}: {e}{RESET}"); traceback.print_exc(limit=10)


def classify_models(data, models):
    data = data.sel(experiment = data.experiment.isin(['ssp245', 'historical'])) #data = models_with_all_experiments(data, keep_experiments=['ssp245', 'historical'], dont_count_historical=False)
    model_set = set(data.model.values.flat)
    preindustrial_t = preindustrial_temp(data)

    not_hot_models = models[models['tcr'] <= 2.2]
    likely_models = not_hot_models[(not_hot_models['tcr'] >= 1.4)]
    not_hot_ecs_models = models[(models['ecs'] <= 4.5)]

    not_hot_data = data.sel(model = data.model.isin(not_hot_models['model'].values))
    #chart.plot(quantiles(not_hot_data, [.5]), alpha=.6)

    likely_data = data.sel(model = data.model.isin(likely_models['model'].values))
    quantile_ranges = quantiles(likely_data, (.1, .5, .9))


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


    final_t = list(map(lambda quantile: quantile.sel(year=slice(2090, 2100+1)).mean().item(), quantile_ranges))
    print("GRAND FINALE: ", final_t)


    #final_t_all = quantile_ranges[1].sel(experiment='ssp245').where(data['year'] > 2090, drop=True).mean().item() - preindustrial_t
    #final_t_likely = likely_t[0].sel(experiment='ssp245').where(data['year'] > 2090, drop=True).mean().item() - preindustrial_t
    #print(f"\nALL models {final_t_likely:.2f} > LIKELY models {final_t_all:.2f} ssp245\n")
    #chart.rightContext([final_t_likely, final_t_all])

    # 3. plot labels explaining the model slection or selections means(s) at 2100 to the right edge of the chart

    hot_models = models[(models['tcr'] > 2.2)]['model'].values

    chart = visualizations.Charter(title=f'Global temperature projections ({len(set(data.model.values.flat))} CMIP6 models)') #, reference_lines=[0, 2]
    chart.plot(quantile_ranges, alpha=1)
    chart.annotate(final_t)

    for model in data.model.values.flat:
      if model in hot_models:
        color = 'red' 
      elif model in likely_models['model'].values: 
        color = 'green'
      else:
        color = 'blue'

      first_decade_t = data.sel(model=model, experiment='historical').where(data['year']<=1860, drop=True).mean().item()
      if first_decade_t >= .8:
        print(f'{model} historical hot')
        linewidth=1.3
      elif first_decade_t <=-.6: 
        print(f'{model} historical cold')
        linewidth=1.3
      else:
        linewidth=.5
      chart.plot([data.sel(model=model)], alpha=.4, color=color, linewidth=linewidth)
    
    chart.show()
    chart.save(tag=f'all_classified')

  
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
def maxTemperature(frequency='daily'):
    variable = 'max_temperature'; global DATADIR; DATADIR = os.path.join(DATADIR, variable+'_'+frequency, '')
    try:
        models = pd.read_csv('metadata/models.csv')

        observed_max_t=[]
        for file in glob('data/Czechia/*.xlsx'):
            observations = pd.read_excel(file, sheet_name='teplota maximální', header=3)
            max_t = observations.iloc[:, :2].copy()
            max_t['Max'] = observations.iloc[:, 2:].max(axis=1)
            max_t = max_t.rename(columns={'rok': 'Year', 'měsíc': 'Month'})
            max_t = max_t.groupby('Year').max()
            max_t = max_t.drop(columns=['Month'])
            observed_max_t.append(max_t)
        observed_max_t = pd.concat(observed_max_t)
        observed_max_t = observed_max_t.groupby(observed_max_t.index)['Max'].max()


        if datastore:
            datastore.DATADIR = DATADIR
            datastore.download(models['model'].values, ['ssp245', 'historical'], variable='tasmax', forecast_from=forecast_from, frequency=frequency,
                area=md['area']['cz'])    

        aggregate(var='tasmax')
        data = loadAggregated(wildcard='tasmax_')

        data = cleanUpData(data)
        #data = models_with_all_experiments(data, dont_count_historical=True)
        data = data['tasmax']
        #data = normalize(data)
        
        quantile_ranges = quantiles(data, (.1, .5, .9))
        maxes = {'Madrid': 35}

        model_set = set(data.sel(experiment='ssp245').dropna(dim='model', how='all').model.values.flat)

        chart = visualizations.Charter(
          title=f'Maximal temperature (in Czechia) projections ({len(model_set)} CMIP6 models)', 
          yticks = [30, 35, 40, 45],
          ylabel='Max Temperature (°C)', yformat=lambda x, pos: f'{x:.0f} °C',
          reference_lines=[preindustrial_temp(quantile_ranges[1]),40]
          )

        chart.scatter(observed_max_t, label='measurements') # expects year as index
        
        chart.plot([quantile_ranges[0], quantile_ranges[-1]], ranges='quantile', labels=scenarios['to-visualize'], models=model_set)
        chart.plot(quantile_ranges[1:2], labels=scenarios['to-visualize'], models=model_set)

        chart.show()
        chart.save(tag=f'{variable}_{len(model_set)}')


        chart = visualizations.Charter(title=f'Maximum temperature projections ({len(model_set)} CMIP6 models)')
        for model in model_set:
          chart.plot([data.sel(model=model, experiment = data.experiment.isin(['ssp245', 'historical']))], alpha=.4, linewidth=.5)
        chart.show()
        chart.save(tag=f'all_{variable}_{len(model_set)}')

    except OSError as e: print(f"{RED}Error: {type(e).__name__}: {e}{RESET}") 
    except Exception as e: print(f"{RED}Error: {type(e).__name__}: {e}{RESET}"); traceback.print_exc()

def tropicDaysBuckets():
    variable = 'max_temperature'; global DATADIR; DATADIR = os.path.join(DATADIR, variable+'_'+'daily', '')
    try:
        models = pd.read_csv('metadata/models.csv')

        max_by_place = [pd.read_excel(file, sheet_name='teplota maximální', header=3) for file in glob('data/Czechia/*.xlsx')]
        max_by_place = pd.concat(max_by_place)
        max_by_place = max_by_place.rename(columns={'rok': 'Year', 'měsíc': 'Month'})
        
        days = max_by_place.columns[2:]
        max_daily = max_by_place.groupby(['Year', 'Month']).agg({day: 'max' for day in days})
        max_daily['tropic_days'] = (max_daily[days] >= 30).sum(axis=1)
        observed_tropic_days_annually = max_daily.groupby('Year')['tropic_days'].sum()
            
        if datastore:
            datastore.DATADIR = DATADIR
            datastore.download(models['model'].values, ['ssp245', 'historical'], variable='tasmax', forecast_from=forecast_from,frequency='daily', #variable=f'daily_maximum_near_surface_air_temperature', 
                area=md['area']['cz'])
      
        aggregate(var='tasmax', stacked=True) 
        data = loadAggregated(wildcard='tasmaxbuckets_')

        data = cleanUpData(data)
        #data = normalize(data)
        data = models_with_all_experiments(data, dont_count_historical=True)

        model_set = set(data.sel(experiment='ssp245').dropna(dim='model', how='all').model.values.flat)

        data = data.median(dim='model').max(dim='experiment')      

        chart = visualizations.Charter(
            title=f'Tropic days (in Czechia) projection ({len(model_set)} CMIP6 models)', 
            subtitle="When no decline of emissions till 2050 (ssp245 scenario)", 
            ylabel='Tropic days annually',
            marker=forecast_from)
        
        chart.stack(data)
        chart.scatter(observed_tropic_days_annually, label='Observed 30+ °C') # expects year as index
        chart.show()
        chart.save(tag=f'tropic_days_{len(model_set)}')

    except OSError as e: print(f"{RED}Error: {type(e).__name__}: {e}{RESET}") 
    except Exception as e: print(f"{RED}Error: {type(e).__name__}: {e}{RESET}"); traceback.print_exc()

def tropic_months(observed_maxes, threshold=30):
    # months whose temperature reached threshold
    tropic_months = []
    for row_idx in range(len(observed_maxes)):
        months_exceeding = []
        for col in observed_maxes.columns:
            if observed_maxes.iloc[row_idx][col] >= threshold:
                months_exceeding.append(col)
        if months_exceeding:
            print(f'Row {observed_maxes.index[row_idx]}:', ', '.join(months_exceeding))
            tropic_months.append([observed_maxes.index[row_idx], months_exceeding])

    for m in tropic_months:
        if m[0][1]<6 or m[0][1]>8: 
         print(f'{m[0][0]} {m[0][1]}. {m[1]}')

    months = set([m[0][1] for m in tropic_months])
    print(months)
    return months

def discovery():
    variable = 'discovery'; global DATADIR; DATADIR = DATADIR + variable + '/'

    datastore = downloader.DownloaderCopernicus(DATADIR, skip_failing_scenarios=False, mark_failing_scenarios=True)
    datastore.download(
      ['canesm5'], ['ssp245'], 
      variable='surface_temperature', frequency='monthly', #variable='daily_maximum_near_surface_air_temperature', frequency='daily',#area=md['area']['cz'],
      start=2020,forecast_from=2020,end=2030)
    
    aggregate(var='tas')
    data = loadAggregated()
    
    #data=data.sel(experiment='ssp126')
    print(data)

    chart = visualizations.Charter()
    #chart.plot(data, what={'experiment': None})

    data = data['tas']
    #quantile_ranges = quantiles(data, (.1, .5, .9))
    #chart.plot([data])

    chart.plotDiscovery(data, what={'experiment':'ssp245'})
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
def aggregate_file(filename, var='tas', buckets=None, area=None):
    var_aggregated = var if not buckets else var+'buckets'
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
            if 'max' in var:
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
        model, experiment, run, grid, time = filename.split('_')[2:7] #<variable_id>_<table_id>_<source_id>_<experiment_id>_<variant_label>_<grid_label>_<time_range>.nc
        da_yr.to_netcdf(path=os.path.join(DATADIR, f'agg_{var_aggregated}_{model}_{experiment}_{run}_{grid}_{time}.nc'))  #da_yr.to_netcdf(path=f'{DATADIR}cmip6_agg_{exp}_{mod}_{str(da_yr.year[0].values)}.nc')

    #except OSError as e: print(f"\n{RED}Error loading model:{RESET} {type(e).__name__}: {e}")
    except Exception as e: print(f"\n{RED}Error aggregating {filename}:{RESET} {type(e).__name__}: {e}"); traceback.print_exc()

def aggregate(stacked=None, var='tas'):
    dataFiles = list()
    var_aggregated = var if not stacked else var+'buckets'
    for i in glob(f'{DATADIR}{var}*.nc'):
        dataFiles.append(os.path.basename(i))
    for filename in dataFiles:
      model, experiment, run, grid, time = filename.split('_')[2:7]
      try:
          candidate_files = [f for f in os.listdir(DATADIR) if f.startswith(f'agg_{var_aggregated}_{model}_{experiment}_{run}_{grid}_{time}')] 
          # NOTE it expects the same filename strucutre, which seems to be followed, but might be worth checking for final run (or regenerating all)
          if reaggregate or not len(candidate_files):
              print('.', end='')
              aggregate_file(filename, var=var, buckets=stacked)

      except Exception as e: print(f"{RED}Error in {filename}: {type(e).__name__}: {e}{RESET}"); traceback.print_exc(limit=1)
    print()

def loadAggregated(models=None, experiments=None, unavailable_experiments=None, wildcard=''):
    filename_pattern = os.path.join(DATADIR, f'agg_*{wildcard}*.nc')
    
    filenames = glob(filename_pattern)
    duplicites = {}
    for filename in filenames: 
        filename = filename.split('/')[-1]
        
        var, model, experiment, variant, grid, time = filename.split('_')[1:7]
        key = f'{var}_{model}_{experiment}_{time}'
        if not key in duplicites:
            duplicites[key] = set()
        else:
            print(f'{YELLOW}duplicate{RESET} {key}: {set([variant])|duplicites[key]}')

        duplicites[key] |= {variant}

    print('Opening aggregations')
    data_ds = None
    data_ds = xr.open_mfdataset(filename_pattern, combine='nested', concat_dim='model') # when problems with loading # data_ds = xr.open_mfdataset(f'{DATADIR}cmip6_agg_*.nc')
    data_ds.load()


    return data_ds

    '''for i in glob(filename_pattern):
        filename = os.path.abspath(i) # os.path.basename(i)
        print(filename)
        new_ds = xr.open_dataset(filename)
        if data_ds is None:
            data_ds = new_ds
        else:
            #data_ds = xr.combine_by_coords([data_ds, new_ds])
            #data_ds = xr.combine_nested([data_ds, new_ds], concat_dim=['experiment', 'model', 'bins', 'year'], combine_attrs='override')
            data_ds = xr.combine_nested([data_ds, new_ds], concat_dim=['model'])


    not_read = set(models)-set(data_ds.model.values.flat) if data_ds else print("Nothing read at all")
    if not_read: print("\nNOT read: '" + ' '.join(map(str, not_read)) +"'")

    #print(len(set(data_ds.sel(experiment='ssp126').model.values.flat)))
    #print(sorted(set(data_ds.sel(experiment='ssp126').model.values.flat)))

    return data_ds'''

def cleanUpData(data):
  # removing historical data before 2014, because some models can include them despite request
  try:
    def filter_years(ds):
      if 'historical' in ds['experiment']:
        ds = ds.sel(year=ds['year'] < forecast_from)
      return ds
    data = data.groupby('experiment').map(filter_years) #, squeeze=True
    
    ''' # CHECK FOR MISSING YEARS
    raise(NotImplementedError)
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
  except Exception as e: 
    print(f"{RED}Error: {type(e).__name__}: {e}{RESET}"); 
    traceback.print_exc(limit=1)  
    print(data)

  return data

def normalize(data):
    try:
      model_mean = data.sel(experiment='historical').sel(year=slice(forecast_from-20, forecast_from-1)).mean(dim='year')

      global_mean = model_mean.mean(dim='model').item()

      normalization_offset = model_mean - global_mean
      normalization_offset_expanded = normalization_offset.expand_dims(dim={'year': data.year}).transpose('model', 'year')

      data = data - normalization_offset_expanded

      #print(f"NORMALIZED {normalized_data.sel(experiment='ssp126', model=normalized_data.model.values.flat[0], year=slice(2014,2015))}")

    except Exception as e: 
        print(f"{RED}Error: {type(e).__name__}: {e}{RESET}"); 
        traceback.print_exc()
        print(data)

    return data

def models_with_all_experiments(data, dont_count_historical=False, drop_experiments=None, keep_experiments=None):

    if drop_experiments:
        data = data.sel(experiment =~ data.experiment.isin(drop_experiments))
    if keep_experiments:
        data = data.sel(experiment = data.experiment.isin(drop_experiments))


    experiments = set(data.experiment.values.flat)
    if dont_count_historical:
        experiments = experiments - {'historical'} 
    
    models_by_experiment = {experiment: set(data.sel(experiment=experiment).dropna(dim='model', how='all').model.values.flat) for experiment in experiments}
    models_available_by_experiment = [models[1] for models in models_by_experiment.items()]
    
    intersection = set.intersection(*models_available_by_experiment)
    union = set.union(*models_available_by_experiment)

    print(f'{len(intersection)}/{len(union)} models in all experiments')

    data = data.sel(model = data.model.isin(list(intersection)))
    data = data.dropna(dim='model', how='all')
    
    remained = set(data.model.values.flat)
    print(f"\n{len(remained)} remained: {remained}") #: {' '.join(remained)}

    for experiment in models_by_experiment.items():
        print(f"{len(experiment[1])}⨉ {experiment[0]}, except: {sorted(union-set(experiment[1]))}")
        #print(experiment[1])

    return data


def preindustrial_temp(data):
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        preindustrial_period = data.sel(experiment = 'historical').sel(year=slice(1850, 1900))
        
        if 'model' in data.dims: # quantiles not count yet
            preindustrial_period = preindustrial_period.quantile(.5, dim='model')
        
        return preindustrial_period.mean(dim='year').item()
    
    else: # dataframe
        return data.loc[1850:1900].mean()


md = util.loadMD('model_md')

# RUN the function defined in the 'run' at the top
try:
  result = main()
  #result = globals()[run]()
except Exception as e: print(f"\nError: {type(e).__name__}: {e}"); traceback.print_exc()