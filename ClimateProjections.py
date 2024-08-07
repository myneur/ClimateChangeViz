# About visualized models: https://confluence.ecmwf.int/display/CKB/CMIP6%3A+Global+climate+projections#CMIP6:Globalclimateprojections-Models

# Aggregations based on colab notebook: ecmwf-projects.github.io/copernicus-training-c3s/projections-cmip6.html

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
forecast_from = 2015 # Forecasts are actually from 2014. Hindcast untill 2018 or 2019?

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

  #return Discovery() # with open('ClimateProjections.py', 'r') as f: exec(f.read())


# VISUALIZATIONS

def GlobalTemperature(drop_experiments=None):
    try:
        models = pd.read_csv('metadata/models.csv')
        observed_t = load_observed_temperature()
        
        datastore.set('tas', 'mon') # temperature above surface
        if not 'preview' in sys.argv: 
            datastore.download(models['model'].values, experiments, forecast_from=forecast_from)

        aggregate(var='tas')
        data = loadAggregated()
        
        data = data['tas']
        data = cleanUpData(data)
        data = normalize(data)
        data_all = data

        classify_models(data, models, observed_t)

        experiment_set = experiments
        for experiment_to_drop in [None, 'ssp119']:
            
            if experiment_to_drop: experiments.remove(experiment_to_drop)
            data = models_with_all_experiments(data_all, keep_experiments=experiment_set, dont_count_historical=True)

            classify_models(data, models, observed_t)
            
            preindustrial_t = preindustrial_temp(data)

            quantile_ranges = quantiles(data, (.1, .5, .9))
            #preindustrial_t = preindustrial_temp(quantile_ranges[1])

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


def classify_models(data, models, observed_t):
    #data = data.sel(experiment = data.experiment.isin(['ssp245', 'historical'])) 
    data = models_with_all_experiments(data, keep_experiments=['ssp245', 'historical'], dont_count_historical=True)
    model_set = set(data.model.values.flat)
    preindustrial_t = preindustrial_temp(data)
    data = data - preindustrial_t

    # Model classes
    not_hot_models = models[models['tcr'] <= 2.2]
    likely_models = not_hot_models[(not_hot_models['tcr'] >= 1.4)]
    likely_models_ecs = models[(models['ecs'] <= 4.5) & (models['ecs'] >= 1.5)]
    hot_models = models[models['tcr'] > 2.2]
    
    # Data by classes
    not_hot_data = data.sel(model = data.model.isin(not_hot_models['model'].values))
    likely_data = data.sel(model = data.model.isin(likely_models['model'].values))
    likely_data_ecs = data.sel(model = data.model.isin(likely_models_ecs['model'].values))
    hot_data = data.sel(model = data.model.isin(hot_models['model'].values))
    
    m1 = models[models['model'].isin(model_set)]
    print(f"+{m1['tcr'].mean():.2f}° ⌀2100: ALL {len(m1)}× ")
    m2 = models[models['model'].isin(not_hot_data.model.values.flat)]
    print(f"+{m2['tcr'].mean():.2f}° ⌀2100: NOT HOT TCR {len(m2)}× ")
    m3 = models[models['model'].isin(likely_data.model.values.flat)]
    print(f"+{m3['tcr'].mean():.2f}° ⌀2100: LIKELY {len(m3)}× ")
    m4 = models[models['model'].isin(likely_data_ecs.model.values.flat)]
    print(f"+{m4['tcr'].mean():.2f}° ⌀2100: NOT HOT ECS {len(m4)}× ")

    # 2100 temperatures

    #final_t_all = quantile_ranges[1].sel(experiment='ssp245').where(data['year'] > 2090, drop=True).mean().item() - preindustrial_t
    #final_t_likely = likely_t[0].sel(experiment='ssp245').where(data['year'] > 2090, drop=True).mean().item() - preindustrial_t
    #print(f"\nALL models {final_t_likely:.2f} > LIKELY models {final_t_all:.2f} ssp245\n")
    #chart.rightContext([final_t_likely, final_t_all])

    likely_range = quantiles(likely_data, (.1, .9))
    hot_range = quantiles(hot_data, (.1, .9))
    final_t = list(map(lambda quantile: quantile.sel(year=slice(2090, 2100+1)).mean().item(), likely_range))
    final_t_hot = list(map(lambda quantile: quantile.sel(year=slice(2090, 2100+1)).mean().item(), hot_range))
    
    print("GRAND FINALE (likely): ", final_t)

    # Comparing Temperature rise for models sorted by TCR

    models = models.sort_values(by='tcr')
    for model in models['model']:
        try:
            m = data.sel(model=model, experiment='ssp245')

            t = m.where((data['year'] >=2090) & (data['year'] <= 2100), drop=True).mean().item() # some models go up to 2200
            tcr = models.loc[models['model'] == model, 'tcr'].values[0]
            print(f"tcr: {tcr:.1f} +{t:.1f}° {model}")
        except:
            pass #print(f'missing {model}')

    # Charts

    chart = visualizations.Charter(
        title=f'Global temperature change projections ({len(set(data.model.values.flat))} CMIP6 models)',
        #ylabel='Difference from pre-industrial era',
        yticks=[0, 1.5, 2, 3, 4], ylimit=[-1,5], reference_lines=[0, 2], 
        yformat=lambda y, i: f"{'+' if y>0 else ''}{y:.1f} °C" 
        ) 

    palette = chart.palette['coldhot']

    chart.scatter(observed_t, label='measurements')

    # Likely range

    chart.plot(likely_range, alpha=1, linewidth=1, color=palette[1])
    chart.annotate(final_t, 'likely', palette[1], offset=4)
    chart.annotate(final_t_hot, 'hot\nmodels', palette[-1], offset=2, align='top')

    hot_model_set = hot_models['model'].values
    likely_model_set = likely_models['model'].values

    # All models
    
    alpha = .3
    for model in data.model.values.flat:
        if model in hot_model_set: 
            color = palette[-1]
        elif model in likely_model_set: 
            color = palette[1]
            alpha = .2
        else: 
            color = palette[0]

        first_decade_t = data.sel(model=model, experiment='historical').where(data['year']<=1860, drop=True).mean().item()
        if first_decade_t >= .8: print(f'{model} historical hot')
        elif first_decade_t <=-.6: print(f'{model} historical cold')
        
        chart.plot([data.sel(model=model)], alpha=alpha, color=color, linewidth=.5)

    # Annotations

    chart.add_legend([[scenarios['to-visualize']['ssp245'], palette[1]]])
    chart.annotate_forecast()
    
    chart.show()
    chart.save(tag=f'all_classified')


# monthly: 'monthly_maximum_near_surface_air_temperature', 'tasmax', 'frequency': 'monthly'
def MaxTemperature(frequency='day'):
    try:
        models = pd.read_csv('metadata/models.csv')

        #models = models[models['Europe-accuracy'] == 1]
        model_names = list(models['model'].values)

        observed_max_t = load_observed_max_temperature()

        datastore.set('tasmax', frequency, area=md['area']['cz']) # temperature above surface max
        if not 'preview' in sys.argv: 
            datastore.download(model_names, ['ssp245', 'historical'], forecast_from=forecast_from)    

        aggregate(var='tasmax')
        data = loadAggregated(wildcard='tasmax_', models=model_names)

        data = cleanUpData(data)
        #data = models_with_all_experiments(data, dont_count_historical=True)
        data = data['tasmax']
        data = normalize(data) # TO REVIEW: should we normalize max the same way like avg? 
        
        quantile_ranges = quantiles(data, (.1, .5, .9))
        maxes = {'Madrid': 35}

        #model_set = set(data.sel(experiment='ssp245').dropna(dim='model', how='all').model.values.flat)
        model_set = set(data.sel(experiment='ssp245').dropna(dim='model', how='all').model.values.flat)

        chart = visualizations.Charter(
          title=f'Maximal temperature (in Czechia) projections ({len(model_set)} CMIP6 models)', 
          yticks = [30, 35, 40, 45],
          ylabel='Max Temperature', yformat=lambda x, pos: f'{x:.0f} °C',
          reference_lines=[preindustrial_temp(quantile_ranges[1]),40]
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

def load_observed_max_temperature():
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
    return observed_max_t.groupby(observed_max_t.index)['Max'].max()

def load_observed_tropic_days():
    max_by_place = [pd.read_excel(file, sheet_name='teplota maximální', header=3) for file in glob('data/Czechia/*.xlsx')]
    max_by_place = pd.concat(max_by_place)
    max_by_place = max_by_place.rename(columns={'rok': 'Year', 'měsíc': 'Month'})
    
    days = max_by_place.columns[2:]
    max_daily = max_by_place.groupby(['Year', 'Month']).agg({day: 'max' for day in days})
    max_daily['tropic_days'] = (max_daily[days] >= 30).sum(axis=1)
    observed_tropic_days_annually = max_daily.groupby('Year')['tropic_days'].sum()
    return observed_tropic_days_annually

def TropicDaysBuckets():
    try:
        models = pd.read_csv('metadata/models.csv')
        
        #models = models[models['Europe-accuracy'] == 1]
        model_names = list(models['model'].values)

        observed_tropic_days_annually = load_observed_tropic_days()
        
        datastore.set('tasmax', 'day', area=md['area']['cz']) # temperature above surface max
        if not 'preview' in sys.argv: 
            datastore.download(model_names, ['ssp245', 'historical'],  forecast_from=forecast_from) #variable=f'daily_maximum_near_surface_air_temperature')
      
        aggregate(var='tasmax', stacked=True) 
        data = loadAggregated(wildcard='tasmaxbuckets_', models=model_names)
        #data = loadAggregated(wildcard='tasmaxbuckets_')

        data = cleanUpData(data)
        #data = normalize(data) # TO REVIEW: should we normalize max the same way like avg? 
        #data = models_with_all_experiments(data, dont_count_historical=True)

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

def Discovery():
    datastore = downloader.DownloaderCopernicus(DATADIR)
    datastore.set('tas', 'mon') # temperature above surface
    datastore.download(
      ['canesm5'], ['ssp245'], 
      variable='surface_temperature', frequency='mon', #variable='daily_maximum_near_surface_air_temperature', frequency='day',#area=md['area']['cz'],
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


# COMPUTATION

def quantiles(data, quantiles):
  quantilized = []
  for q in quantiles:
    quantilized.append(data.quantile(q, dim='model'))
  
  return quantilized

  quantilized = xr.concat([data.quantile(q, dim='model') for q in quantiles], dim='quantile')
  quantilized['quantile'] = quantiles  # name coordinates

def create_buckets(data):
  t30 = ((data >= (30+K)) & (data < (35+K))).resample(time='YE').sum(dim='time')
  t35 = (data >= (35+K)).resample(time='YE').sum(dim='time') # t35 = ((data >= (35+K)) & (data < np.inf)).resample(time='YE').sum(dim='time')
  buckets = xr.Dataset(
    {'bucket': (('bins', 'time'), [t30, t35])},
    coords={'bins': ['30-35', '35+'],'time': t30.time})

  buckets = buckets.assign_coords(year=buckets['time'].dt.year)
  buckets = buckets.drop_vars('time')
  buckets = buckets.rename({'time': 'year'})

  return buckets

def model_coverage(data, lat, lon, tolerance=None):
    area = [data[lat].max().item(), data[lon].min().item(), data[lat].min().item(), data[lon].max().item()]
    if tolerance:
        area = [area[0]+tolerance[0], area[1]-tolerance[1], area[2]-tolerance[0], area[3]+tolerance[1]]
    return area

def add_params(data, lat, lon, area):
    coverage = model_coverage(data, lat, lon)
    points = [data[lat].size, data[lon].size]
    resolution = [(coverage[0]-coverage[2])/points[0], (coverage[3]-coverage[1])/points[1]]
    data.attrs['coverage'] = coverage
    data.attrs['points'] = points
    data.attrs['resolution'] = resolution
    if area:
        # grow area so points nearer than half resolution will fit in
        areaTolerance = model_coverage(data, lat, lon, tolerance=[resolution[0]/2, resolution[1]/2])
        data.attrs['areaTolerance'] = areaTolerance
    return data 


K = 273.15 # Kelvins
def aggregate_file(filename, var='tas', buckets=None, area=None, verbose=False): # [N,W,S,E] area
    var_aggregated = var if not buckets else var+'buckets'
    try:
        ds = xr.open_dataset(f'{datastore.DATADIR}{filename}')

        exp = ds.attrs['experiment_id']
        mod = ds.attrs['source_id']

        # Fixing inconsistent naming
        if 'lat' in ds.coords: lat, lon = 'lat', 'lon' 
        else: lat, lon = 'latitude', 'longitude'
        
        # Narrow to selected variable
        data = ds[var] 
        if 'height' in data.coords:
            data = data.drop_vars('height')

        # Model spatial details about 
        data = add_params(data, lat, lon, area)
        
        # filter within area
        #print(data.attrs)
        if area: 
            # area growed by half resolution
            if len(area)>3:
                tolerance = data.attrs['areaTolerance']
                cover = data.attrs['coverage']
                if(cover[0]>tolerance[0] or cover[1]<tolerance[1] or cover[2]<tolerance[2] or cover[3]>tolerance[3]):
                    print(cover, area)
                    print((cover[0]>area[0] , cover[1]<area[1] , cover[2]<area[2] , cover[3]>area[3]))
                    data = data.sel({lat: slice(area[2], area[0]), lon: slice(area[1], area[3])})# S-N # W-E
            else:
                data = data.sel({lat: lat_value, lon: lon_value}, method='nearest')
            data.attrs['coverage_constrained'] = model_coverage(data, lat, lon)

        if verbose: print('>' if area else '.', end='')
        
        # AGGREGATE SPATIALLY
        
        # MAX
        if var == 'tasmax':
            global_agg = data.max([lat, lon])
        
        # AVG
        else:
            # Weight as longitude gird shrinks with latitude
            weights = np.cos(np.deg2rad(data[lat]))
            weights.name = "weights"
            data_weighted = data.weighted(weights)
            global_agg = data_weighted.mean([lat, lon])

        # AGGREGATE TIME

        if buckets:
            year_agg = create_buckets(global_agg)
        else:
            # MAX
            if 'max' in var:
                year_agg = global_agg.groupby('time.year').max()

            # AVG
            else: 
                year_agg = global_agg.groupby('time.year').mean()

            year_agg = year_agg - K # °C
        
        # CONTEXT
        year_agg = year_agg.assign_coords(model=mod)
        year_agg = year_agg.expand_dims('model')
        year_agg = year_agg.assign_coords(experiment=exp)
        year_agg = year_agg.expand_dims('experiment')
        for attr in data.attrs.keys():
            year_agg.attrs[attr] = data.attrs[attr]

        # SAVE
        model, experiment, run, grid, time = filename.split('.')[0].split('_')[2:7] #<variable_id>_<table_id>_<source_id>_<experiment_id>_<variant_label>_<grid_label>_<time_range>.nc
        year_agg.to_netcdf(path=os.path.join(datastore.DATADIR, f'agg_{var_aggregated}_{model}_{experiment}_{run}_{grid}_{time}.nc'))  #year_agg.to_netcdf(path=f'{datastore.DATADIR}cmip6_agg_{exp}_{mod}_{str(year_agg.year[0].values)}.nc')

        return year_agg

    #except OSError as e: print(f"\n{RED}Error loading model:{RESET} {type(e).__name__}: {e}")
    except Exception as e: print(f"\n{RED}Error aggregating {filename}:{RESET} {type(e).__name__}: {e}"); traceback.print_exc()

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
              aggregate_file(filename, var=var, buckets=stacked, area=datastore.area, verbose=True)
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

def cleanUpData(data, start=1850, end=2100):
    # removing historical data beyond requested period, because some models can include them despite request
    try:
        print(f'Cropping years')
        def filter_years(ds):
            if 'historical' in ds['experiment']:
                ds = ds.sel(year=ds['year'] < forecast_from)
            else:
                ds = ds.sel(year=ds['year'] <= end)
            return ds
        data = data.groupby('experiment').map(filter_years) #, squeeze=True

        data = data.groupby('model').mean('model')

        print(f"Models with incomplete coverage from {len(data['model'].values)} models and {len(data['experiment'].values)} experiments:")
        model_limits = []
        for model in set(data['model'].values):
            for experiment in set(data['experiment'].values):
                try:
                    model_data = data.sel(model=model, experiment=experiment)
                    if isinstance(data, xr.Dataset):
                        var = list(model_data.data_vars)[0]
                        model_data = model_data[var]
                    
                    mask = ~model_data.isnull()
                    if 'bins' in model_data.dims:
                        mask = mask.any(dim='bins')  

                    valid_years = model_data['year'][mask].values
                    model_min, model_max = valid_years.min().item(), valid_years.max().item()

                    min_year, max_year = (1850, 2014) if experiment == 'historical' else (2015, 2100-1) # let's be tolerant for the last missing year
                    
                    if model_min > min_year or model_max < max_year:
                        data = data.where(~((data.model == model) & (data.experiment == experiment)), drop=True)
                        model_limits.append([model, experiment, model_min, model_max])

                except (ValueError, AttributeError, KeyError) as e:
                    data = data.where(~((data.model == model) & (data.experiment == experiment)), drop=True)
                    model_limits.append([model, experiment, None, None])

                except Exception as e:
                    print(f"{RED}Error model {model}, {experiment}: {type(e).__name__}: {e}{RESET}"); 
                    model_limits.append([model, experiment, None, None])
                    traceback.print_exc(limit=1)  
                    print(model_data)
        
        print(YELLOW, end='')
        model_limits = pd.DataFrame(model_limits, columns=['model', 'experiment', 'start', 'end'])
        print(model_limits[model_limits['start'].notna() & model_limits['end'].notna()])
        print(RESET, end='')

    except Exception as e: 
        print(f"{RED}Error: {type(e).__name__}: {e}{RESET}")
        traceback.print_exc(limit=1)  
        print(data)

    return data

def normalize(data, measurements=None, period=20):
    try:
        
        # it will drop models without historical experiment

        if measurements: # normalize to overlapping period if we have measurements
            last_measurement = measurements.index[-1]
            overlap = measurements[measurements.index >= forecast_from]
            measured_mean = overlap.mean()
            print(f'{len(overlap)} years overlap since {last_measurement} to {measurements.index[-1]}')

            model_mean = data.sel(experiment='historical').sel(year=slice(last_measurement-len(overlap), last_measurement+1)).mean(dim='year')
            #global_mean = model_mean.mean(dim='model').item()
            normalization_offset = model_mean - measured_mean

        else: # normalize to last 2 decades of hindcast if we don't have measurements
            model_mean = data.sel(experiment='historical').sel(year=slice(forecast_from-period, forecast_from)).mean(dim='year')
            global_mean = model_mean.mean(dim='model').item()
            normalization_offset = model_mean - global_mean
        
        normalization_offset_expanded_to_all_years = normalization_offset.expand_dims(dim={'year': data.year}).transpose('model', 'year')
        data = data - normalization_offset_expanded_to_all_years

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
        data = data.sel(experiment = data.experiment.isin(keep_experiments))


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
if __name__ == "__main__":
    try:
      result = main()
      #result = globals()[run]()
    except Exception as e: print(f"\nError: {type(e).__name__}: {e}"); traceback.print_exc()