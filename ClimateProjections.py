# About visualized models: https://confluence.ecmwf.int/display/CKB/CMIP6%3A+Global+climate+projections#CMIP6:Globalclimateprojections-Models

# Data-sets: cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6 | Model availability: cds.climate.copernicus.eu | aims2.llnl.gov

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
forecast_from = 2015

from glob import glob
from pathlib import Path
import os
import sys
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
import calculations as calc

import util
from util import debug
import traceback

RED = '\033[91m'; RESET = "\033[0m"; YELLOW = '\033[33m'

# UNCOMENT WHAT TO DOWNLOAD, COMPUTE AND VISUALIZE:

DATADIR = os.path.expanduser(f'~/Downloads/ClimateData/') # DOWNLOAD LOCATION (most models have hundreds MB globally)
if 'esgf' in sys.argv:
    if 'wget' in sys.argv:
        datastore = downloader.DownloaderESGF(DATADIR, method='wget')
    else:
        datastore = downloader.DownloaderESGF(DATADIR, method='request') 
else:
    datastore = downloader.DownloaderCopernicus(DATADIR) #mark_failing_scenarios = True to save unavailable experiments not to retry downloading again and again. Clean it in 'metadata/status.json'. 

experiments = list(scenarios['to-visualize'].keys())

def main():
    if 'max' in sys.argv:
        MaxTemperature()
        TropicDaysBuckets()
    else:
        GlobalTemperature()

# VISUALIZATIONS

def GlobalTemperature(drop_experiments=None):
    try:
        models = pd.read_csv('metadata/models.csv')
        observed_t = calc.observed_temperature()
        
        datastore.set('tas', 'mon') # temperature above surface
        if not 'preview' in sys.argv: 
            datastore.download(models['model'].values, experiments, forecast_from=forecast_from)

        aggregate(var='tas')
        data = loadAggregated()
        
        data = data['tas']
        data = calc.cleanup_data(data)
        data = calc.normalize(data)
        data_all = data

        calc.classify_models(data, models, observed_t)

        experiment_set = experiments
        for experiment_to_drop in [None, 'ssp119']:
            
            if experiment_to_drop: experiments.remove(experiment_to_drop)
            data = calc.models_experiments_intersection(data_all, keep_experiments=experiment_set, dont_count_historical=True)

            calc.classify_models(data, models, observed_t)
            
            preindustrial_t = calc.preindustrial_temp(data)

            quantile_ranges = [data.quantile(q, dim='model') for q in (.1, .5, .9)]
            #preindustrial_t = calc.preindustrial_temp(quantile_ranges[1])

            model_set = set(data.model.values.flat)

            chart = visualizations.Charter(
                title=f'Global temperature change projections ({len(model_set)} CMIP6 models)', 
                #ylabel='Difference from pre-industrial era',
                zero=preindustrial_t, yticks=[0, 1.5, 2, 3], ylimit=[-1,4], reference_lines=[0, 2], 
                yformat=lambda y, i: f"{'+' if y-preindustrial_t>0 else ''}{y-preindustrial_t:.1f} °C" 
                )

            chart.scatter(observed_t + preindustrial_t, label='measurements') # the observations are already relative to 1850-1900 preindustrial average

            chart.plot([quantile_ranges[0], quantile_ranges[-1]], ranges=True, labels=scenarios['to-visualize'], models=model_set)
            chart.plot(quantile_ranges[1:2], labels=scenarios['to-visualize'], models=model_set)
            chart.annotate_forecast(y=preindustrial_t)

            chart.show()
            chart.save(tag=f'tas_{len(model_set)}_{"+".join(experiment_set)}')
            
        return data
    except OSError as e: print(f"{RED}Error: {type(e).__name__}: {e}{RESET}") 
    except Exception as e: print(f"{RED}Error: {type(e).__name__}: {e}{RESET}"); traceback.print_exc(limit=10)


# monthly: 'monthly_maximum_near_surface_air_temperature', 'tasmax', 'frequency': 'monthly'
def MaxTemperature(frequency='day'):
    try:
        models = pd.read_csv('metadata/models.csv')

        if 'eu' in sys.argv:
            models = models[models['Europe-accuracy'] == 1]
        model_names = list(models['model'].values)

        observed_max_t = calc.observed_max_temperature()

        datastore.set('tasmax', frequency, area=md['area']['cz']) # temperature above surface max
        if not 'preview' in sys.argv: 
            datastore.download(model_names, ['ssp245', 'historical'], forecast_from=forecast_from)    

        aggregate(var='tasmax')
        data = loadAggregated(wildcard='tasmax_', models=model_names)

        data = calc.cleanup_data(data)
        #data = calc.models_experiments_intersection(data, dont_count_historical=True)
        data = data['tasmax']
        #data = calc.normalize(data) # TO REVIEW: should we normalize max the same way like avg? 

        calc.classify_models(data, models, observed_max_t)
        
        quantile_ranges = [data.quantile(q, dim='model') for q in (.1, .5, .9)]

        maxes = {'Madrid': 35}

        #model_set = set(data.sel(experiment='ssp245').dropna(dim='model', how='all').model.values.flat)
        model_set = set(data.sel(experiment='ssp245').dropna(dim='model', how='all').model.values.flat)

        chart = visualizations.Charter(
          title=f'Maximal temperature (in Czechia) projections ({len(model_set)} CMIP6 models)', 
          yticks = [30, 35, 40, 45],
          ylabel='Max Temperature', yformat=lambda x, pos: f'{x:.0f} °C',
          reference_lines=[calc.preindustrial_temp(quantile_ranges[1]),40]
          )

        chart.scatter(observed_max_t, label='measurements') # expects year as index
        
        chart.plot([quantile_ranges[0], quantile_ranges[-1]], ranges='quantile', labels=scenarios['to-visualize'], models=model_set)
        chart.plot(quantile_ranges[1:2], labels=scenarios['to-visualize'], models=model_set)

        chart.show()
        chart.save(tag=f'tasmax_{len(model_set)}')

        chart = visualizations.Charter(title=f'Maximum temperature projections ({len(model_set)} CMIP6 models)')
        for model in model_set:
            chart.plot([data.sel(model=model, experiment = data.experiment.isin(['ssp245', 'historical']))], alpha=.4, linewidth=.5)
        chart.show()
        chart.save(tag=f'all_tasmax_{len(model_set)}')

    except OSError as e: print(f"{RED}Error: {type(e).__name__}: {e}{RESET}") 
    except Exception as e: print(f"{RED}Error: {type(e).__name__}: {e}{RESET}"); traceback.print_exc()


def TropicDaysBuckets():
    try:
        models = pd.read_csv('metadata/models.csv')
        
        #models = models[models['Europe-accuracy'] == 1]
        model_names = list(models['model'].values)

        observed_tropic_days_annually = calc.observed_tropic_days()
        
        datastore.set('tasmax', 'day', area=md['area']['cz']) # temperature above surface max
        if not 'preview' in sys.argv: 
            datastore.download(model_names, ['ssp245', 'historical'],  forecast_from=forecast_from) #variable=f'daily_maximum_near_surface_air_temperature')
      
        aggregate(var='tasmax', stacked=True) 
        data = loadAggregated(wildcard='tasmaxbuckets_', models=model_names)
        #data = loadAggregated(wildcard='tasmaxbuckets_')

        data = calc.cleanup_data(data)
        #data = calc.normalize(data) # TO REVIEW: should we normalize max the same way like avg? 
        #data = calc.models_experiments_intersection(data, dont_count_historical=True)

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

        tropic_days = data.sum(dim='bins')

        #chart = visualizations.Charter(title=f'Tropic days ({len(model_set)} CMIP6 models)') 
        #chart.scatter(observed_tropic_days_annually, label='measurements')
        #chart.plot([tropic_days], series=None, alpha=1, linewidth=1, color=palette[1])
        #chart.show()

    except OSError as e: print(f"{RED}Error: {type(e).__name__}: {e}{RESET}") 
    except Exception as e: print(f"{RED}Error: {type(e).__name__}: {e}{RESET}"); traceback.print_exc()

def aggregate(stacked=None, var='tas'):
    dataFiles = list()
    var_aggregated = var if not stacked else var+'buckets'
    for i in glob(f'{datastore.DATADIR}{var}*.nc'):
        dataFiles.append(os.path.basename(i))
    for filename in dataFiles:
      model, experiment, run, grid, time = filename.split('_')[2:7]
      try:
          candidate_files = [f for f in os.listdir(datastore.DATADIR) if f.startswith(f'agg_{var_aggregated}_{model}_{experiment}_{run}_{grid}_{time}')] 
          # NOTE it expects the same filename strucutre, which seems to be followed, but might be worth checking for final run (or regenerating all)
          if not len(candidate_files):
              calc.aggregate_model(filename, datastore.DATADIR, var=var, buckets=stacked, area=datastore.area, verbose=True)
              #data = aggregate_file(filename, var=var, buckets=stacked)

      except Exception as e: print(f"{RED}Error in {filename}: {type(e).__name__}: {e}{RESET}"); traceback.print_exc(limit=1)
    print()

def loadAggregated(models=None, experiments=None, unavailable_experiments=None, wildcard=''):
    filename_pattern = os.path.join(datastore.DATADIR, f'agg_*{wildcard}*.nc')
    
    pathnames = glob(filename_pattern)

    if models:
        pathnames = [name for name in pathnames if any(model in name for model in models)]

    duplicites = {}
    for pathname in pathnames: 
        filename = pathname.split('/')[-1]
        
        var, model, experiment, variant, grid, time = filename.split('_')[1:7]
        key = f'{var}_{model}_{experiment}_{time}'
        if not key in duplicites:
            duplicites[key] = set()
        else:
            print(f'{YELLOW}duplicate{RESET} {key}: {set([variant])|duplicites[key]}')

        duplicites[key] |= {variant}

    print('Opening aggregations')
    data_ds = None
    data_ds = xr.open_mfdataset(pathnames, combine='nested', concat_dim='model') # when problems with loading # data_ds = xr.open_mfdataset(f'{datastore.DATADIR}cmip6_agg_*.nc')
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

md = util.loadMD('model_md')

# RUN the function defined in the 'run' at the top
if __name__ == "__main__":
    try:
      result = main()
      #result = globals()[run]()
    except Exception as e: print(f"\nError: {type(e).__name__}: {e}"); traceback.print_exc()


# with open('debug-snippet.py', 'r') as f: exec(f.read())