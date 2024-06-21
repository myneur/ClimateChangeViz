# TODO
# unziping to different folders not to interfere
# 0. aggregated models by satte as series to show how big is the coverage of the models
# 1. max temp: means of Jun-Jul afternoon instad of yearly max to get more realistic projections?
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
import cftime
import urllib3
urllib3.disable_warnings() # Disable warnings for data download via API

# Libraries for working with multi-dimensional arrays
import numpy as np
import xarray as xr
import pandas as pd

# Libraries for plotting and visualising data
import matplotlib
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

from downloader import download

import util
from util import debug

variable = 'temperature'
#variable = 'max_temperature'

experiments = ['historical', 'ssp119', 'ssp126', 'ssp245'] # removed for now: 'ssp534os', 'ssp570', 'ssp585'
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

forecast_from = 2015 # hidcast data is not available beyond 2014 anyway
#DATADIR = f'/Users/myneur/Downloads/ClimateData/validated/'
DATADIR = f'/Users/myneur/Downloads/ClimateData/{variable}/'


models = md['validated_models']
if variable == 'temperature':
  download(models, experiments, DATADIR, mark_failing_scenarios=False, forecast_from=forecast_from)
else:
  download(models, experiments, DATADIR, variable=variables[variable]['request'], area=md['area']['cz'], frequency='monthly', mark_failing_scenarios=True, forecast_from=forecast_from)

cmip6_nc = list()
cmip6_nc_rel = glob(f'{DATADIR}tas*.nc')
for i in cmip6_nc_rel:
    cmip6_nc.append(os.path.basename(i))


# Function to aggregate in geographical lat lon dimensions

def aggregate_geo_time(fn): # TODO change naming not ot read the file to check for experiment and model
  ncfile = f'{DATADIR}{fn}'
  ds = xr.open_dataset(ncfile, decode_times=False)
  experiment = ds.attrs['experiment_id']
  model = ds.attrs['source_id']

  # Converting dates to make it robust for incompatible date formats.
  time_units = ds['time'].attrs['units']
  time_data = ds['time'].values
  dates = cftime.num2date(time_data, units=time_units, calendar='360_day')

  # Chosen variable to visualize
  da = ds[variables[variable]['dataset']] # 'tas' or 'tasmax'

  # Spatial aggregation
  weights = np.cos(np.deg2rad(da.lat))
  weights.name = "weights"
  da_weighted = da.weighted(weights)
  da_agg = da_weighted.mean(['lat', 'lon'])

  # Reassigning dates after conversion to support various date formats # instead of da_agg.groupby('time.year').mean()
  years = [date.year for date in dates]
  da_agg = da_agg.assign_coords(year=('time', years))
  da_yr = da_agg.groupby('year').mean(dim='time')

  # Conversion from Kelvin to Celsius
  da_yr = da_yr - 273.15

  # Additional data dimensions (to later combine data from multiple models & experiments)
  da_yr = da_yr.assign_coords(model=model)
  da_yr = da_yr.expand_dims('model')
  da_yr = da_yr.assign_coords(experiment=experiment)
  da_yr = da_yr.expand_dims('experiment')

  # Saving aggregated data for a visualisation
  ncaggregated = f'{DATADIR}cmip6_agg_{experiment}_{model}_{str(da_yr.year[0].values)}.nc'  
  da_yr.to_netcdf(path=ncaggregated)

def aggregate_all():
  for filename in cmip6_nc:
    model, experiment = filename.split('_')[2:4]
    try:
      candidate_files = [f for f in os.listdir(DATADIR) if f.endswith('.nc') and f.startswith(f'cmip6_agg_{experiment}_{model}')] # TODO: multiple files for multiple years can exist
      if not len(candidate_files):
        print(f'aggregating {model} {experiment}')
        aggregate_geo_time(filename)
    except Exception as e:
      print(f'- failed {filename}')
      print(f"Error: {type(e).__name__}: {e}")

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

  data = data_ds_filtered[variables[variable]['dataset']]

  data_90 = data.quantile(0.9, dim='model')
  data_10 = data.quantile(0.1, dim='model')
  data_50 = data.quantile(0.5, dim='model')

  preindustrial_temp = data_50.sel(year=slice(1850, 1900)).mean(dim='year').mean(dim='experiment').item()

  models_read = set(data.model.values.flat)
  model_count = len(models_read)
  print(model_count)
  print(models_read)
  print('CMIP6 projections. Averages by 50th quantile. Ranges by 10-90th quantile.')

  colors = ['black','#3DB5AF','#61A3D2','#EE7F00', '#E34D21']


  def chart(zero=0, reference_lines=None):
    fig, ax = plt.subplots(1, 1, figsize = (16, 8))
    if variable == 'temperature':
      ax.set(title=f'Global temperature projections ({model_count} CMIP6 models)', ylabel='Temperature near surface (°C)')  
    else:
      ax.set(title=f'Maximumal temperature (in Czechia) projections ({model_count} CMIP6 models)', ylabel='Max Temperature near surface (°C)')  

    ax.set(xlim=(1850, 2100))
    plt.subplots_adjust(left=.08, right=.97, top=0.95, bottom=0.15)
    ax.yaxis.label.set_size(14)

    # SCALE
    if zero:
      diff_temp = [round(val - zero, 1) for val in plt.gca().get_yticks()]
      yticks = [0, 1.5, 2, 3, 4]
      plt.gca().set_yticks([val + zero for val in yticks])
      plt.tick_params(axis='y', colors='#717174')

      ax.set_ylim([-1 +zero, 4 + zero])

      plt.gca().set_yticklabels([f'{"+" if val > 0 else ""}{val:.1f} °C' for val in yticks])
    else:
      ax.set_ylim([28, 34])
    plt.tick_params(axis='x', colors='#717174')

    xticks_major = [1850, 2000, 2015, 2050, 2075, 2100]
    xtickvals_major = ['1850', '2000', '2015', '2050', '2075', '2100']
    xticks_minor = [1900, 1945, 1970, 1995, 2020, 2045, 2070, 2095]
    xtickvals_minor = ['Industrial Era', 'Baby Boomers', '+1 gen', '+2 gen', '+3 gen', '+4 gen', '+5 gen', '+6 gen']

    ax.set_xticks(xticks_major)  
    ax.set_xticklabels(xtickvals_major)
    ax.set_xticks(xticks_minor, minor=True)  
    ax.set_xticklabels(xtickvals_minor, minor=True, rotation=45, va='bottom', ha='right',  fontstyle='italic', color='#b2b2b2', fontsize=9)
    ax.xaxis.set_tick_params(which='minor', pad=70, color="white")


    if reference_lines: 
      ax.axhline(y=zero, color='#717174') # base
      for ref in reference_lines:
        ax.axhline(y=zero+ref, color='#E34D21', linestyle='--', alpha=.5, linewidth=.5)
      plt.grid(axis='y', linestyle=(0, (3, 9)), alpha=.5)

    # LEGEND
    legend = [scenarios[s] for s in data_50.experiment.values]

    # DATA
    for i in np.arange(len(experiments)):
      try:
        # AVERAGES
        ax.plot(data_50.year, data_50[i,:], color=f'{colors[i]}', label=f'{legend[i]}', linewidth=1.3)
        ax.fill_between(data_50.year, data_90[i,:], data_10[i,:], alpha=0.05, color=f'{colors[i]}')
      except:
        pass
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', frameon=False)


    # OUTPUT
    fig.savefig(f'charts/chart_{variable}_{len(set(data.model.values.flat))}m.png')
    plt.show()

  if variable == 'temperature':
    chart(zero=preindustrial_temp)
  else: 
    chart()
except Exception as e:
  print(f'- failed {model}')
  print(f"Error: {type(e).__name__}: {e}")

#print(nodata)