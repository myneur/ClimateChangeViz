import cdsapi
from pyesgf.logon import LogonManager
from pyesgf.search import SearchConnection

import util
import os
import fnmatch
import glob
import re
import requests
import subprocess
from dotenv import load_dotenv
import urllib3 
urllib3.disable_warnings() # Disable warnings for data download via API

BLUE = "\033[34m" #BLUE = '\033[94m' #CYAN = '\033[96m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
RED = '\033[91m'
GREY = "\033[47;30m"
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
RESET = "\033[0m"

class DownloaderCopernicus:
  def __init__(self, DATADIR):
    self.DATADIR = DATADIR
    self.status = util.loadMD('status')

    self.c = cdsapi.Client() # Doc: https://cds.climate.copernicus.eu/toolbox/doc/how-to/1_how_to_retrieve_data/1_how_to_retrieve_data.html

  def download(self, models, experiments, variable='near_surface_air_temperature', frequency='monthly', area=None, mark_failing_scenarios=False, skip_failing_scenarios=False, forecast_from=2015, start=1850, end=2100, fileformat='zip'): 

    unavailable_experiments = self.status['unavailable_experiments'][variable] if skip_failing_scenarios else {}

    for experiment in experiments:
      if experiment == 'historical':
        end = forecast_from-1      
      else:
        start = forecast_from
      date = f'{start}-01-01/{end}-12-31'

      separator = '='*60
      print(f'\n\nRequesting {variable} {frequency} {experiments} for {models} {start}-{end}\n{separator}\n')

      for model in models:
        if not skip_failing_scenarios or (experiment not in unavailable_experiments or not (model in unavailable_experiments[experiment])):
          try:
            filename = f'{self.DATADIR}cmip6_monthly_{start}-{end}_{experiment}_{model}.{fileformat}'
            if not os.path.isfile(filename):
              params = {'format': fileformat,
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
              
              self.c.retrieve('projections-cmip6', params, filename)
              if fileformat == 'zip':
                util.unzip(filename, self.DATADIR)
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
      self.status['unavailable_experiments'][variable] = unavailable_experiments
      util.saveMD(self.status, 'status') 

    return unavailable_experiments



  def reanalysis(self):
    self.c.retrieve('reanalysis-era5-single-levels', {
      'product_type': 'reanalysis', 
      'variable': '2m_temperature'
      #'year': list(range(1910,1918+1)),
      #'area': [51, 12, 48, 18]
      })

  def metadata(self, models, experiments, date=2014): 
    metadata = []
    for model in models: 
      for experiment in experiments:
        date = f'{date}-01-01/{date}-12-31'
        metadata = (self.c.retrieve('projections-cmip6', {'format': 'zip','temporal_resolution': 'monthly','experiment': 'historical','level': 'single_levels','variable': 'tas','model': model,'date': date }))
        #metadata.append(c.retrieve('projections-cmip6', {'format': 'zip','temporal_resolution': 'monthly','experiment': 'historical','level': 'single_levels','variable': 'tas','model': model,'date': date }))
        
        metadata_json = metadata.download() 
        with open(metadata_json, 'r') as f:
            data = json.load(f)
        print(data['models'])
        print(data['experiments'])
        print(data['date_ranges'])

class DownloaderESGF:
  def __init__(self, DATADIR):
    self.DATADIR = DATADIR
    load_dotenv(dotenv_path=os.path.expanduser('~/.esgfenv'))
    self.lm = LogonManager()
    self.servers = ['esgf-data.dkrz.de', 'esg-dn1.nsc.liu.se', 'esgf-node.ipsl.upmc.fr', 'esgf-node.llnl.gov', 'esgf-data1.llnl.gov', 'esgf.nci.org.au', 'esgf-node.ornl.gov', 'esgf.ceda.ac.uk', 'esgf-data04.diasjp.net']
    # web search https://aims2.llnl.gov/search

    if not self.lm.is_logged_on():
        self.login()

    self.connection = SearchConnection(f'https://{self.servers[-2]}/esg-search', distrib=False)

    # https://esgf.github.io/esg-search/ESGF_Search_RESTful_API.html

  def login(self):
    user = os.getenv('ESGF_OPENID')
    print(f'Logging-in as {user}')
    self.lm.logon_with_openid(openid=user, password=os.getenv('ESGF_PASSWORD'), bootstrap=False)

  def logoff(self):
    self.lm.logoff()


  def download(self, models, experiments, variable='tas', frequency='mon'):
    print(f"\033[47;30m Downloading {models} {experiments} \033[0m")
    existing_files = [os.path.basename(file) for file in self.list_files('*.nc')]

    for model in models:
      for experiment in experiments:
            if not self.file_in_list(existing_files, f'{variable}*_{model}_{experiment}*.nc'):
                results = self.connection.new_context(source_id=model, experiment_id=experiment, variable='tas', frequency='mon').search()
                if(len(results)):
                    # 'context', 'dataset_id', 'download_url', 'file_context', 'globus_url', 'gridftp_url', 'index_node', 'json', 'las_url', 'number_of_files', 'opendap_url', 'urls'
                    print(f'Found {model} {experiment}: {len(results)}×')
                    
                    results = sorted(results, key=self.splitByNums, reverse=True) # latest release at the top
                    print(f'{BLUE}⬇{RESET} downloading {results[0].dataset_id}')
                    
                    files = results[0].file_context().search()
                    print("Downloadable from:")
                    print([file.download_url for file in files])

                    #self.downloadDs(results[0], model, experiment, variable)
                    self.downloadRq(results[0], model, experiment, variable)
                else:
                    print(f'❌ missing {model} {experiment}')
            else:
                print(f'✅ exists {model} {experiment}')
    return 

  def downloadRq(self, ds, model, experiment, variable):
    files = ds.file_context().search()
    for file in files:
        url = file.download_url
        filename = url.split('/')[-1]
        print(f'{BLUE}⬇{RESET} downloading {url} to {filename}')
        if False:
            print(f'✅ exists {model} {experiment}')
        response = requests.get(url, stream=True)   #line 54: requests.exceptions.ConnectionError: ('Connection aborted.', TimeoutError(60, 'Operation timed out')) < urllib3.exceptions.ProtocolError: ('Connection aborted.', TimeoutError(60, 'Operation timed out')) < TimeoutError: [Errno 60] Operation timed out
        if response.status_code == 200:
            with open(self.DATADIR + filename, 'wb') as f:
                progress = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress += 1
                    if progress%100 == 0:
                        print('.', end='')
                print()
            print(f'✅ Downloaded {model} {experiment}')
        else:
            print(f'❌ download failed {model} {experiment}: Code: {response.status_code}')

  def downloadDs(self, ds, model, experiment, variable):
    fc = ds.file_context()
    wget_script_content = fc.get_download_script()
    script_path = self.DATADIR + "download-{}.sh".format(os.getpid())
    #file_handle, script_path = tempfile.mkstemp(suffix='.sh', prefix='download-')
    try:
        with open(script_path, "w") as writer:
            writer.write(wget_script_content)

        os.chmod(script_path, 0o750)
        download_dir = os.path.dirname(script_path)
        subprocess.check_output("{}".format(script_path), cwd=download_dir)
        
        files = self.list_files(f'{variable}*{model}*{experiment}*.nc')
        if files:
            removed = 0
            for file in files:
                if os.path.getsize(file) == 0:
                    removed += 1
                    os.remove(file)
            if removed: 
                print(f'❌ download failed {model} {experiment}')
            else:
                print(f'✅ Downloaded {model} {experiment}')
        else:
            print(f'❌ download failed {model} {experiment}')
    except Exception as e: 
        print(f"❌ download failed {model} {experiment}: {type(e).__name__}: {e}")#; traceback.print_exc(limit=1)

  def list_files(self, pattern):
    return glob.glob(os.path.join(self.DATADIR, pattern))

  def file_in_list(self, files, pattern):
    return [file for file in files if fnmatch.fnmatch(file, pattern)]

  def splitByNums(self, ds):
    return [int(part) if part.isdigit() else part for part in re.split('(\d+)', ds.dataset_id)]


class Downloader:
  def __init__(self, DATADIR, mark_failing_scenarios=False, skip_failing_scenarios=False, forecast_from=None, start=None, end=None, fileformat='zip'): 
    self.DATADIR = DATADIR
    self.mark_failing_scenarios=mark_failing_scenarios
    self.skip_failing_scenarios=skip_failing_scenarios
    self.fileformat=fileformat

    if forecast_from or start or end:
      if not start: start = 1850
      if not end: end = 2100
      if not forecast_from: forecast_from = 2015

    self.forecast_from=forecast_from
    self.start=start
    self.end=end
    

    self.status = util.loadMD('status')

  def download(models, experiments, variable='near_surface_air_temperature', frequency='monthly', area=None): 
    unavailable_experiments = self.status['unavailable_experiments'][variable] if self.skip_failing_scenarios else {}

    for experiment in experiments:
      if forecast_from:
        if experiment == 'historical':
          end = forecast_from-1      
        else:
          start = forecast_from
        date = f'{start}-01-01/{end}-12-31'

      separator = '='*60
      print(f'\n\nRequesting {variable} {frequency} {experiments} for {models} {start}-{end}\n{separator}\n')

      for model in models:
        if not self.skip_failing_scenarios or (experiment not in unavailable_experiments or not (model in unavailable_experiments[experiment])):
          filename = f'{self.DATADIR}cmip6_{frequency}_{start}-{end}_{experiment}_{model}.{fileformat}'
          raise NotImplementedError('TODO added {frequency} to download filename') # StopIteration InterruptedError FileNotFoundError


def main():
  datastore = DownloaderESGF()
  #datastore.logoff()
  #datastore.login()
  results = datastore.download(['IPSL-CM5A2-INCA', 'CESM2', 'HadGEM3-GC31-MM', 'EC-Earth3-Veg-LR', 'KIOST-ESM'], ['ssp245', 'ssp126', 'ssp119', 'historical'][1:2])

if __name__ == "__main__":
    main()