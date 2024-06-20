import cdsapi
import util
import os

c = cdsapi.Client()

unavailable_experiments = util.loadMD('unavailable_experiments')

def download(models, experiments, DATADIR, forecast_from=2015, save_failing_scenarios=False): # WIP
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
            print(f'REQUESTING: {experiment} from {model} for {date}')
            c.retrieve('projections-cmip6', 
              {'format': 'zip',
              'temporal_resolution': 'monthly',
              'experiment': f'{experiment}',
              'level': 'single_levels',
              'variable': 'near_surface_air_temperature',
              'model': f'{model}',
              'date': date
              }, 
              filename)
            util.unzip(filename)
          else:
            print(f'REUSING: {experiment} for {model}')
        except Exception as e:
          print(f'\nUNAVAILABLE experiment {experiment} for {model}')
          print(f"Error: {type(e).__name__}: {e}")
          if not experiment in unavailable_experiments: 
            unavailable_experiments[experiment] = []
          unavailable_experiments[experiment].append(model)
      else:
        print(f'\nSKIPPING UNAVAILABLE experiment {experiment} for {model}')

  if unavailable_experiments:
    print(f"\nUNAVAILABLE:\n{unavailable_experiments}")
  
  if(save_failing_scenarios):
    util.saveMD(unavailable_experiments, 'unavailable_experiments') 