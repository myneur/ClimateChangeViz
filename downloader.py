import cdsapi
import util
import os
import urllib3 
urllib3.disable_warnings() # Disable warnings for data download via API

status = util.loadMD('status')

c = cdsapi.Client()

def metadata(models, experiments, date=2014): 
  metadata = []
  for model in models: 
    for experiment in experiments:
      date = f'{date}-01-01/{date}-12-31'
      metadata = (c.retrieve('projections-cmip6', {'format': 'zip','temporal_resolution': 'monthly','experiment': 'historical','level': 'single_levels','variable': 'tas','model': model,'date': date }))
      #metadata.append(c.retrieve('projections-cmip6', {'format': 'zip','temporal_resolution': 'monthly','experiment': 'historical','level': 'single_levels','variable': 'tas','model': model,'date': date }))
      
      metadata_json = metadata.download() 
      with open(metadata_json, 'r') as f:
          data = json.load(f)
      print(data['models'])
      print(data['experiments'])
      print(data['date_ranges'])

def download(models, experiments, DATADIR, variable='near_surface_air_temperature', frequency='monthly', area=None, mark_failing_scenarios=False, forecast_from=2015): # WIP
  unavailable_experiments = status['unavailable_experiments'][variable]  
  separator = '='*60
  print(f'\n\nRequesting {variable} {frequency} {experiments} for {models}\n{separator}\n')
  for experiment in experiments:
    if experiment == 'historical':
      start = 1850
      till = forecast_from-1      
    else:
      start = forecast_from
      till = 2100

    date = f'{start}-01-01/{till}-12-31'
    for model in models:
      if experiment not in unavailable_experiments or not (model in unavailable_experiments[experiment]):          
        try:
          filename = f'{DATADIR}cmip6_monthly_{start}-{till}_{experiment}_{model}.zip'
          if not os.path.isfile(filename):
            params = {'format': 'zip',
              'temporal_resolution': frequency,
              'experiment': f'{experiment}',
              'level': 'single_levels',
              'variable': variable,
              'model': f'{model}',
              'date': date
              }
            if frequency == 'daily':
              params['month'] = ['06', '07', '08']

            if area: params['area'] = area
            print(f'REQUESTING: {experiment} from {model} for {date}')
            c.retrieve('projections-cmip6', params, filename)
            util.unzip(filename, DATADIR)
          else:
            print(f'REUSING: {experiment} for {model}')
        except Exception as e:
          print(f'\nUNAVAILABLE experiment {experiment} for {model}')
          print(f"\nError:\n––––––\n{type(e).__name__}: {e}\n––––––\n")
          if not experiment in unavailable_experiments: 
            unavailable_experiments[experiment] = []
          unavailable_experiments[experiment].append(model)
      else:
        print(f'\nSKIPPING UNAVAILABLE experiment {experiment} for {model}')

  if unavailable_experiments:
    print(f"\nUNAVAILABLE:")
    for experiment in unavailable_experiments.keys():
      print (f"{experiment}: {' '.join(unavailable_experiments[experiment])}")
    print("\n")
  
  if(mark_failing_scenarios):
    status['unavailable_experiments'][variable] = unavailable_experiments
    util.saveMD(status, 'status') 

  return unavailable_experiments

def reanalysis():
  c.retrieve('reanalysis-era5-single-levels', {
    'product_type': 'reanalysis', 
    'variable': '2m_temperature'
    #'year': list(range(1910,1918+1)),
    #'area': [51, 12, 48, 18]
    })
