import visualizations
from ClimateProjections import scenarios, forecast_from

import pandas as pd
import xarray as xr

from glob import glob

import traceback

K = 273.15 # Kelvins
RED = '\033[91m'; RESET = "\033[0m"; YELLOW = '\033[33m'

# Aggregations based on colab notebook: ecmwf-projects.github.io/copernicus-training-c3s/projections-cmip6.html
def aggregate_model(filename, path, var='tas', buckets=None, area=None, verbose=False): # [N,W,S,E] area
    var_aggregated = var if not buckets else var+'buckets'
    try:
        ds = xr.open_dataset(os.path.join(path,filename))

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
        data = add_model_context(data, lat, lon, area)
        
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
        year_agg.to_netcdf(path=os.path.join(path, f'agg_{var_aggregated}_{model}_{experiment}_{run}_{grid}_{time}.nc'))  #year_agg.to_netcdf(path=f'{path}cmip6_agg_{exp}_{mod}_{str(year_agg.year[0].values)}.nc')

        return year_agg

    #except OSError as e: print(f"\n{RED}Error loading model:{RESET} {type(e).__name__}: {e}")
    except Exception as e: print(f"\n{RED}Error aggregating {filename}:{RESET} {type(e).__name__}: {e}"); traceback.print_exc()


def cleanup_data(data, start=1850, end=2100):
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

def models_experiments_intersection(data, dont_count_historical=False, drop_experiments=None, keep_experiments=None):

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

def add_model_context(data, lat, lon, area):
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

def classify_models(data, models, observed_t):
    #data = data.sel(experiment = data.experiment.isin(['ssp245', 'historical'])) 
    data = models_experiments_intersection(data, keep_experiments=['ssp245', 'historical'], dont_count_historical=True)
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

    likely_range = [likely_data.quantile(q, dim='model') for q in (.1, .9)]
    hot_range = [hot_data.quantile(q, dim='model') for q in (.1, .9)]
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

def observed_temperature():
  # OBSERVATIONS from https://climate.metoffice.cloud/current_warming.html
  observations = [pd.read_csv(f'data/{observation}.csv') for observation in ['gmt_HadCRUT5', 'gmt_NOAAGlobalTemp', 'gmt_Berkeley Earth']]
  observations = pd.concat(observations)
  observation = observations[['Year', observations.columns[1]]].groupby('Year').mean()
  return observation[observation.index <= 2023]

def observed_max_temperature():
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

def observed_tropic_days():
    max_by_place = [pd.read_excel(file, sheet_name='teplota maximální', header=3) for file in glob('data/Czechia/*.xlsx')]
    max_by_place = pd.concat(max_by_place)
    max_by_place = max_by_place.rename(columns={'rok': 'Year', 'měsíc': 'Month'})
    
    days = max_by_place.columns[2:]
    max_daily = max_by_place.groupby(['Year', 'Month']).agg({day: 'max' for day in days})
    max_daily['tropic_days'] = (max_daily[days] >= 30).sum(axis=1)
    observed_tropic_days_annually = max_daily.groupby('Year')['tropic_days'].sum()
    return observed_tropic_days_annually