# TODO
# unziping to different folders not to interfere
# 0. aggregated models by satte as series to show how big is the coverage of the models
# 1. max temp: means of summer months afternoon instad of yearly max to get more realistic projections?
# 2. plot range of forecast to the right edge of the chart
# API keys from cds.climate.copernicus.eu must be in ~/.cdsapirc

# Colab notebook: ecmwf-projects.github.io/copernicus-training-c3s/projections-cmip6.html
# Data-sets: cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6 | aims2.llnl.gov
# Temperature extremes: 
# ecmwf-projects.github.io/copernicus-training-c3s/reanalysis-temp-record.html
# github.com/faktaoklimatu/data-analysis/blob/master/notebooks/teplotni-extremy-cr.ipynb


from glob import glob
from pathlib import Path
import os
from os.path import basename

# Data
import numpy as np
import xarray as xr
import pandas as pd

# Viz
import matplotlib
import matplotlib.path as mpath
import matplotlib.pyplot as plt

import datetime
import bisect
import cftime
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

# Utils
from downloader import download
import util
from util import debug
import traceback

# SETUP #
# What to analyze:

#variable = 'temperature'
variable = 'max_temperature'
stacked = True


experiments = ['historical', 'ssp119', 'ssp126', 'ssp245'] # removed for now: 'ssp534os', 'ssp570', 'ssp585'
colors = ['black','#3DB5AF','#61A3D2','#EE7F00', '#E34D21']
scenarios = {
    'historical': "hindcast", 
    'ssp119': "1.5°: carbon neutral in 2050", 
    'ssp126': "2°: carbon neutral in 2075",
    'ssp245': "3° no decline till ½ millenia",
    'ssp534os': "peak at 2040, then steeper decline",
    'ssp570': "4° 2× emissions in 2100",
    'ssp585': "5° 3× emissions in 2075"
    }
variables = {
  'temperature': {'request': 'near_surface_air_temperature', 'dataset': 'tas'},
  'max_temperature': {'request': 'daily_maximum_near_surface_air_temperature', 'dataset': 'tasmax'}
}

md = util.loadMD('model_md')

current_year = datetime.datetime.now().year
forecast_from = 2015 # hidcast data is not available beyond 2014 anyway for most models
DATADIR = f'/Users/myneur/Downloads/ClimateData/{variable}/'


models = md['validated_models'] # models to be downloaded – when empty, only already downloaded files will be visualized
# mark_failing_scenarios=True saves unavailable experiments not to retry downloading. Clean it in metadata/status.json
if variable == 'temperature':
  download(models, experiments, DATADIR, mark_failing_scenarios=True, forecast_from=forecast_from)
else:
  models = md["validated_max_temp"]
  models = []
  experiments = ['historical', 'ssp126', 'ssp245']
  colors = ['black','#61A3D2','#EE7F00', '#E34D21']
  experiments = ['historical', 'ssp245']
  #download(models, experiments, DATADIR, variable=variables[variable]['request'], area=md['area']['cz'], frequency='monthly', mark_failing_scenarios=True, forecast_from=forecast_from)
  download(models, experiments, DATADIR, variable=variables[variable]['request'], area=md['area']['cz'], frequency='daily', mark_failing_scenarios=True, forecast_from=forecast_from)

cmip6_nc = list()
cmip6_nc_rel = glob(f'{DATADIR}tas*.nc')
for i in cmip6_nc_rel:
    cmip6_nc.append(os.path.basename(i))

K = 273.15
def geog_agg(fn, buckets=False, area=None):
  try:
    ds = xr.open_dataset(f'{DATADIR}{fn}')

    # Context
    exp = ds.attrs['experiment_id']
    mod = ds.attrs['source_id']

    # Fixing inconsistent naming
    if 'lat' in ds.coords: lat, lon = 'lat', 'lon' 
    else: lat, lon = 'latitude', 'longitude'
    
    # Narrow to selected variable
    var = 'tasmax' if 'tasmax' in ds else 'tas'
    da = ds[var] 
    if 'height' in da.coords:
      da = da.drop_vars('height')

    if area: 
      if len(area)>3:
        da.sel({lat: slice(area[0], area[2]), lon: slice(area[1], area[3])})# N-S # W-E
      else:
        da.sel({lat: lat_value, lon: lon_value}, method='nearest')
    
    # Aggregate in spatially 
    if var == 'tasmax':
      da_agg = da.max(['lat', 'lon'])
    else:
      # Weight according to the latitude
      weights = np.cos(np.deg2rad(da.lat))
      weights.name = "weights"
      da_weighted = da.weighted(weights)
      da_agg = da_weighted.mean(['lat', 'lon'])

    # Aggregate in time
    if var == 'tasmax':
      if buckets:
        # single count
        #da_yr = (da_agg >= (30+K)).resample(time='YE').sum(dim='time') #.resample(time='ME') 
        
        # buckets
        #bucket = xr.Dataset({'bucket': (('30-35', '35+'), 
        #    [(da_agg >= 30) & (da_agg < 35).resample(time='A').sum(dim='time'), 
        #    (da_agg >= 35).resample(time='A').sum(dim='time')])})
        # OR
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
    return da_yr
  except Exception as e:
    print(f"{fn}\nError: {type(e).__name__}: {e}")
    traceback.print_exc(limit=1)


def aggregate_all():
  for filename in cmip6_nc:
    model, experiment = filename.split('_')[2:4]
    try:
      candidate_files = [f for f in os.listdir(DATADIR) if f.endswith('.nc') and f.startswith(f'cmip6_agg_{experiment}_{model}')] # TODO: multiple files for multiple years can exist
      if not len(candidate_files):
        print(f'aggregating {model} {experiment}')
        geog_agg(filename, buckets=stacked)
    except Exception as e:
      print(f'- failed {filename}')
      print(f"Error: {type(e).__name__}: {e}")
      traceback.print_exc(limit=1)

aggregate_all()
print('opening aggregations')
#nodata = []
#for model in models: 
#  print(model)
try:
  #data_ds = xr.open_mfdataset(f'{DATADIR}cmip6_agg_*{model}*.nc', combine='nested', concat_dim='model')

  data_ds = xr.open_mfdataset(f'{DATADIR}cmip6_agg_*.nc', combine='nested', concat_dim='model')
  data_ds.load()

  # removing historical data before 2014, because some models can include them despite request
  def filter_years(ds):
    if 'historical' in ds['experiment']:
      ds = ds.sel(year=ds['year'] < forecast_from)
    return ds

  data_ds_filtered = data_ds.groupby('experiment').map(filter_years) #, squeeze=True

  if not stacked:
    data = data_ds_filtered[variables[variable]['dataset']]
    data_90 = data.quantile(0.9, dim='model')
    data_10 = data.quantile(0.1, dim='model')
    data_50 = data.quantile(0.5, dim='model')

    preindustrial_temp = data_50.sel(year=slice(1850, 1900)).mean(dim='year').mean(dim='experiment').item()

  else:
    data = data_ds

  models_read = set(data.model.values.flat)
  model_count = len(models_read)
  not_read = set(models)-models_read
  print("\nNOT read: " + ' '.join(map(str, not_read)))
except Exception as e:
  print(f"\nError: {type(e).__name__}: {e}")
  traceback.print_exc(limit=1)


def xaxis_climatic(ax):
  ax.set(xlim=(1850, 2100))
  plt.subplots_adjust(left=.08, right=.97, top=0.95, bottom=0.15)
  ax.yaxis.label.set_size(14)

  xticks_major = [1850, 2000, 2015, 2050, 2075, 2100]
  bisect.insort(xticks_major, current_year)
  xticks_minor = [1900, 1945, 1970, 1995, 2020, 2045, 2070, 2095]
  xtickvals_minor = ['Industrial Era', 'Baby Boomers', '+1 gen', '+2 gen', '+3 gen', '+4 gen', '+5 gen', '+6 gen']

  ax.set_xticks(xticks_major) 
  ax.set_xticklabels(xticks_major)
  ax.set_xticks(xticks_minor, minor=True)  
  ax.set_xticklabels(xtickvals_minor, minor=True, rotation=45, va='bottom', ha='right',  fontstyle='italic', color='#b2b2b2', fontsize=9)
  ax.xaxis.set_tick_params(which='minor', pad=70, color="white")

  ax.axvline(x=current_year, color='lightgray', linewidth=.5)

def chart(what='mean', zero=None, reference_lines=None):
  try:
    fig, ax = plt.subplots(1, 1)
    if variable == 'temperature':
      ax.set(title=f'Global temperature projections ({model_count} CMIP6 models)', ylabel='Temperature')  
    else:
      ax.set(title=f'Maximal temperature (in Czechia) projections ({model_count} CMIP6 models)', ylabel='Max Temperature (°C)')  

    # SCALE
    if zero and not (np.isnan(zero) and not np.isinf(zero)):
      if zero:
        yticks = [0, 1.5, 2, 3, 4]
        plt.gca().set_yticks([val + zero for val in yticks])
        ax.set_ylim([-1 +zero, 4 + zero])
        plt.gca().set_yticklabels([f'{"+" if val > 0 else ""}{val:.1f} °C' for val in yticks])
    else:
      if False and variable == 'max_temperature':
        ax.set_ylim([34, 40])

    xaxis_climatic(ax)

    if reference_lines: 
      if not zero:
        zero = 0
      ax.axhline(y=zero+reference_lines[0], color='#717174') # base
      for ref in reference_lines[1:]:
        ax.axhline(y=zero+ref, color='#E34D21', linewidth=.5)
      plt.grid(axis='y')

    ax.axvline(x=current_year, color='lightgray', linewidth=.5)

    
    # DATA
    if what == 'mean':
      legend = [scenarios[s] for s in data_50.experiment.values]
      for i in np.arange(len(experiments)):
        try:
          ax.plot(data_50.year, data_50[i,:], color=f'{colors[i%len(colors)]}', label=f'{legend[i]}', linewidth=1.3)
          ax.fill_between(data_50.year, data_90[i,:], data_10[i,:], alpha=0.05, color=f'{colors[i]}')
        except Exception as e: print(f"Error: {type(e).__name__}: {e}")
    else:
      years = data.coords['year'].values
      legend = data.model.values
      for i, model in enumerate(data.coords['model'].values):
        try:
          ax.plot(years, data.sel(model=model).values.squeeze(), color=f'{colors[i%len(colors)]}', label=model, linewidth=1.3)
        except Exception as e: print(f"Error: {type(e).__name__}: {e}")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', frameon=False)


    # CONTEXT
    context = "models: " + ' '.join(map(str, models_read)) # +'CMIP6 projections. Averages by 50th quantile. Ranges by 10-90th quantile.'
    plt.text(0.5, 0.005, context, horizontalalignment='center', color='#cccccc', fontsize=6, transform=plt.gcf().transFigure)
    print(context)

    # OUTPUT
    fig.savefig(f'charts/chart_{variable}_{len(set(data.model.values.flat))}m.png')
    fig.savefig(f'charts/chart_{variable}_{len(set(data.model.values.flat))}m.svg')
    plt.show()
  except Exception as e:
    print(f"Visualization\nError: {type(e).__name__}: {e}")
    traceback.print_exc(limit=1)



def chartstacked(what='series'):
  try:
    fig, ax = plt.subplots(1, 1)
    ax.set(title=f'Local tropic days projection ({model_count} CMIP6 models)', ylabel='Tropic days annualy')  

    xaxis_climatic(ax)

    years = data.year.values #years = data.coords['year'].values

    # one value, no buckets
    #tasmax_max = data.tasmax.max(dim='experiment').mean(dim='model')
    #plt.bar(years, tasmax_max.squeeze().values)

    bucket_values = data.bucket.squeeze()  
    bins = data.bins.values
    #bins = sorted(bins, reverse=True)
    bottom = np.zeros(len(years)) 
    bucket_sums = bucket_values.mean(dim='model').max(dim='experiment')
    palette = ["#FCED8D", "#FF9F47", "#E04B25", "#7A0B0A", "#330024"][2:]
    colors = plt.cm.hot(range(len(bins)))
    for i, bin_label in enumerate(bins):
        bin_values = bucket_sums.sel(bins=bin_label).values
        ax.bar(years, bin_values, label=f'{bin_label} °C', bottom=bottom, color=palette[i])
        bottom += bin_values

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', frameon=False)

    # CONTEXT
    context = "models: " + ' '.join(map(str, models_read)) # +'CMIP6 projections. Averages by 50th quantile. Ranges by 10-90th quantile.'
    plt.text(0.5, 0.005, context, horizontalalignment='center', color='#cccccc', fontsize=6, transform=plt.gcf().transFigure)
    print(context)

    # OUTPUT
    fig.savefig(f'charts/chart_{variable}_{len(set(data.model.values.flat))}m.png')
    #fig.savefig(f'charts/chart_{variable}_{len(set(data.model.values.flat))}m.svg')
    plt.show()
  except Exception as e:
    print(f"- failed viz\nError: {type(e).__name__}: {e}")
    traceback.print_exc(limit=1)

  

if variable == 'temperature':
  chart(zero=preindustrial_temp, reference_lines=[0, 2])
  #chart(what='series')
  #chart()
else: 
  if stacked:
    chartstacked(what='series')
  else:
    chart(reference_lines=[preindustrial_temp, 32])
