import util
import os
import json
import fnmatch
import glob
import re
import requests
import time
import subprocess
import ssl
import OpenSSL
from dotenv import load_dotenv
import urllib3 
urllib3.disable_warnings() # Disable warnings for data download via API
import traceback

BLUE = "\033[34m" #BLUE = '\033[94m' #CYAN = '\033[96m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
RED = '\033[91m'
GREY = "\033[47;30m"
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
RESET = "\033[0m"

class Downloader:
    def __init__(self, DATADIR, mark_failing_scenarios=False, skip_failing_scenarios=False, forecast_from=None, start=None, end=None, fileformat='zip'): 
        self.DATADIR = DATADIR
        os.makedirs(DATADIR, exist_ok=True)
        self.fileformat=fileformat
        self.skip_failing_scenarios = skip_failing_scenarios
        self.mark_failing_scenarios = mark_failing_scenarios
        self.status = util.loadMD('status')
    '''
        if forecast_from or start or end:
        if not start: start = 1850
        if not end: end = 2100
        if not forecast_from: forecast_from = 2015

        self.forecast_from=forecast_from
        self.start=start
        self.end=end
        '''

    def list_files(self, pattern): 
        return glob.glob(os.path.join(self.DATADIR, pattern))

    '''
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
                  filename = os.path.join(self.DATADIR, f'cmip6_{frequency}_{start}-{end}_{experiment}_{model}.{fileformat}')
                  raise NotImplementedError('TODO added {frequency} to download filename') # StopIteration InterruptedError FileNotFoundError'''

    def get_certificate_issuer(self, url):
        import socket
        from urllib.parse import urlparse

        if not url.startswith("http"): url = "https://"+ url
        hostname = urlparse(url).hostname # import idna;hostname = idna.encode(hostname).decode()
        conn = ssl.create_default_context().wrap_socket(socket.socket(socket.AF_INET), server_hostname=hostname)
        conn.settimeout(3.0)

        try:
            conn.connect((hostname, 443))
            cert = conn.getpeercert()
        finally:
            conn.close()
        return(cert)

class DownloaderCopernicus(Downloader):
    def __init__(self, DATADIR, fileformat='zip', mark_failing_scenarios=False, skip_failing_scenarios=False):
        super().__init__(DATADIR, fileformat=fileformat, mark_failing_scenarios=mark_failing_scenarios, skip_failing_scenarios=skip_failing_scenarios)
        import cdsapi
        self.c = cdsapi.Client() # Doc: https://cds.climate.copernicus.eu/toolbox/doc/how-to/1_how_to_retrieve_data/1_how_to_retrieve_data.html


    def download(self, models, experiments, variable='near_surface_air_temperature', frequency='monthly', area=None, forecast_from=2015, start=1850, end=2100): 

        unavailable_experiments = self.status['unavailable_experiments'][variable] if self.skip_failing_scenarios else {}
        print(f"\n\n{BLUE}{BOLD}Requesting {variable} {frequency} {models} {experiments} {start}-{end}\n{'='*60}\n{RESET}")


        for experiment in experiments:
            if experiment == 'historical':
                end = forecast_from-1      
            else:
                start = forecast_from
            date = f'{start}-01-01/{end}-12-31'

            for model in models:
                var = 'tasmax' if 'max' in variable else 'tas'
                if not self.skip_failing_scenarios or (experiment not in unavailable_experiments or not (model in unavailable_experiments[experiment])):
                    try:
                        files = self.list_files(f'*{var}*_{model}_{experiment}*')
                        filename = os.path.join(self.DATADIR, f'{var}_A{frequency[:3]}_{model}_{experiment}_{start}-{end}.{self.fileformat}')

                        if not files:
                            params = {'format': self.fileformat,
                                'temporal_resolution': frequency,
                                'experiment': f'{experiment}',
                                'level': 'single_levels',
                                'variable': variable,
                                'model': f'{model}',
                                'date': date}
                            if area: params['area'] = area
                            #if frequency == 'daily': params['month'] = ['O4', '05' '06', '07', '08', '09']
                            
                            print(f'{BLUE}REQUESTING: {model} {experiment} for {date}{RESET}')
                            
                            self.c.retrieve('projections-cmip6', params, filename)
                            if self.fileformat == 'zip':
                                util.unzip(filename, self.DATADIR)
                                os.remove(os.path.join(self.DATADIR, filename))
                            print(f'{GREEN}DOWNLOADED: {model} {experiment} {RESET}')
                        else:
                            print(f'{GREEN}REUSING: {model} {experiment} {RESET}')
                    except Exception as e:
                        print(f'\n{RED}UNAVAILABLE {model} {experiment}{RESET}')
                        print(f"\n{type(e).__name__}: {e}")
                        if not experiment in unavailable_experiments: 
                            unavailable_experiments[experiment] = []
                        unavailable_experiments[experiment].append(model)
                else:
                    print(f'{RED}SKIPPING UNAVAILABLE {model} {experiment}{RESET}')

        if unavailable_experiments:
            print(f"\nUNAVAILABLE:")
            for experiment in unavailable_experiments.keys(): print (f"{experiment}: {' '.join(unavailable_experiments[experiment])}\n") 
        
        if(self.mark_failing_scenarios):
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

class DownloaderESGF(Downloader):
    servers = [
        'esgf-data.dkrz.de', 
        'esgf-node.llnl.gov', 
        'esgf.ceda.ac.uk', 
        'esg-dn1.nsc.liu.se', 
        'esgf-node.ipsl.upmc.fr', 
        'esgf.nci.org.au', 
        'esgf-node.ornl.gov', 
        'esgf-data04.diasjp.net'] 

    def __init__(self, DATADIR, server=0, method='request', fileformat='nc', mark_failing_scenarios=True, skip_failing_scenarios=True):
        super().__init__(DATADIR, fileformat=fileformat, mark_failing_scenarios=mark_failing_scenarios, skip_failing_scenarios=skip_failing_scenarios)
        
        from pyesgf.logon import LogonManager
        from pyesgf.search import SearchConnection
        load_dotenv(dotenv_path=os.path.expanduser('~/.esgfenv'))
        self.lm = LogonManager()
        self.current_server = server%len(self.servers)
        self.downloadMethod = method

        if not self.lm.is_logged_on():
            self.login()

        self.connection = SearchConnection(f'https://{DownloaderESGF.servers[self.current_server]}/esg-search', distrib=True)

        # https://esgf.github.io/esg-search/ESGF_Search_RESTful_API.html

        self.max_tries = 5
        self.retry_delay = 10
        #os.environ['ESGF_PYCLIENT_NO_FACETS_STAR_WARNING'] = '1'

    def login(self):
        user = os.getenv('ESGF_OPENID')
        print(f'Logging-in as {UNDERLINE}{user}{RESET}')
        try:
            self.lm.logon_with_openid(openid=user, password=os.getenv('ESGF_PASSWORD'), bootstrap=False)
        except OpenSSL.SSL.Error as e:
            issuer = self.get_certificate_issuer(user)['issuer']
            print(f"{RED}Your python does not trust the certificate of the Data Service.\nCertificate Issuer: O:{issuer[1][0][1]} CN: {issuer[2][0][1]}{RESET}")
            raise(e)

    def logoff(self):
        self.lm.logoff()


    def download(self, models, experiments, variable='tas', frequency='mon'):
        print(f"{BLUE}Downloading {BOLD}{models} {experiments}{RESET}")
        existing_files = [os.path.basename(file) for file in self.list_files('*.nc')]
        downloaded = False
        for model in models:
            for experiment in experiments:
                if not self.file_in_list(existing_files, f'{variable}*_{model}_{experiment}*.nc'):
                    for attempt in range(self.max_tries):
                        try:
                            print(f'Try {attempt} {model} {experiment}', end='\r')
                            results = []
                            facets='variant_label,version,data_node'
                            context = self.connection.new_context(facets=facets, project='CMIP6', source_id=model, experiment_id=experiment, variable='tas', frequency='mon')
                            [print(f'{facet} {counts}') for facet, counts in context.facet_counts.items()]
                            results = context.search()
                            break
                        except requests.exceptions.Timeout as e:
                            if attempt < self.max_tries:
                              print(f"Timeout. Retrying search in {self.retry_delay} s:\n{type(e).__name__}: {e}")
                              time.sleep(self.retry_delay)
                            else:
                              print(f'❌ download search failed {model} {experiment}: Timeout'); 
                              break
                        except (requests.exceptions.RequestException, Exception) as e:
                            print(f'❌ download search failed {model} {experiment}: {type(e).__name__}: {e}'); traceback.print_exc()
                            break
                      
                    if(len(results)):
                        print(f'Found {model} {experiment}: {len(results)}×')

                        print(f'{BLUE}⬇{RESET} downloading {results[0].dataset_id}')

                        #results = sorted(results, key=self.splitByNums, reverse=True) 

                        for result in results:
                            # print(r.urls['THREDDS'][0][0].split('/')[2])
                            #print(dir(r.context)) #'connection', 'constrain', 'facet_constraints', 'facet_counts', 'facets', 'fields', 'freetext_constraint', 'geospatial_constraint', 'get_download_script', 'get_facet_options', 'hit_count', 'latest', 'replica', 'search', 'search_type', 'shards', 'temporal_constraint', 'timestamp_range'
                            #print(r.json)
                            print(f"{UNDERLINE}{result.dataset_id.split('|')[1]}{RESET} {result.number_of_files} files")

                        #[print(f'{UNDERLINE}{node}{RESET}: {counts} facets') for node, counts in context.facet_counts['data_node'].items()]

                        #result = results[0]

                        #for result in results: # TODO pick just one variant per server
                        if results[0]:
                            result = results[0]
                            server = result.dataset_id.split('|')[1]
                            try:
                                if self.downloadMethod == 'request':
                                    context = result.file_context()
                                    #context.facet 'data_node' 'index_node' 'data_specs_version' 'nominal_resolution'
                                    [print(f'{facet} {counts}') for facet, counts in context.facet_counts.items() if counts]
                                    
                                    for file in context.search(facets='variant_label,version,data_node'):
                                        #for facet, counts, in context.facet_counts.items():print(f'facet :{facet}');for value, count in counts.items():print(f'{value}: {count}')
                                        #'context', 'download_url', 'file_id', 'filename', 'index_node', 'json',  'size', 'tracking_id', 'urls'
                                        downloaded = self.downloadUrl(file.download_url)
                                    
                                    if downloaded:
                                        break
                                else:
                                  if self.downloadWget(result, model, experiment, variable): 
                                    downloaded = True
                                    break
                            #TODO when we group by server, we should switch server here #except requests.exceptions.ConnectionError as e: print(f"❌ server {server} Timeout: {type(e).__name__}: {e}")
                                #ConnectionError: HTTPConnectionPool(host='{data_node}', port=80): Max retries exceeded with url: /thredds/fileServer/{filepathname} (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at {object}>: Failed to establish a new connection: [Errno 60] Operation timed out'))
                                
                            except (requests.exceptions.TooManyRedirects, requests.exceptions.RequestException, Exception) as e:

                                print(f"❌ server {server} failed: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)

                        # 'number_of_files', 'las_url', 'urls', 'context',  'opendap_url', 'globus_url', 'gridftp_url', 'index_node', 'json', 
                    else:
                        print(f'❌ missing {model} {experiment}')
                else:
                    print(f'✅ exists {model} {experiment}')
                    return True
            print(f"{'✅ Downloaded' if downloaded else '❌ Download failed'} {model} {experiment}")
        return downloaded

    def downloadRequest(self, url):
        filename = url.split('/')[-1]
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            if 'Content-Length' in response.headers:
                size = float(response.headers['Content-Length'])
                print(f"{size/1000000:.2f} MB")
            with open(os.path.join(self.DATADIR, filename), 'wb') as f:
                progress = 0
                chunk = 8192
                for data in response.iter_content(chunk_size=chunk):
                    f.write(data)
                    progress += chunk
                    if progress%(100*chunk) == 0:
                        #print('.', end='')
                        if size > 0:
                          print(f"Downloaded: {int(progress/size)}%", end='\r')
            return True
        return False

    def downloadUrl(self, url):
        print(f'{BLUE}⬇{RESET} downloading {url}')
        for attempt in range(self.max_tries):
            try:
                return self.downloadRequest(url)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < self.max_tries:
                    print(f"Timeout. Retrying download in {self.retry_delay} s:\n{type(e).__name__}: {e}")
                    time.sleep(self.retry_delay)
                    self.downloadUrl(url)
                else:
                    print(f'❌ max retries')
                    return False

        return False

    def downloadWget(self, ds, model, experiment, variable):
        fc = ds.file_context()
        wget_script_content = fc.get_download_script()
        script_path = os.path.join(self.DATADIR, "download-{}.sh".format(os.getpid()))
        #file_handle, script_path = tempfile.mkstemp(suffix='.sh', prefix='download-')
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
            if not removed: 
                return True
        return False

    def list_files(self, pattern):
      return glob.glob(os.path.join(self.DATADIR, pattern))

    def file_in_list(self, files, pattern):
      return [file for file in files if fnmatch.fnmatch(file, pattern)]

    def splitByNums(self, ds):
      return [int(part) if part.isdigit() else part for part in re.split('(\d+)', ds.dataset_id)]


def trusted_ca():
    import certifi
    trusted = []
    with open(certifi.where(), 'r') as f:
        for line in f:
            if line.startswith('# Issuer:'):
                trusted.append(line.strip())
    return trusted


    return dict(x[0] for x in cert['issuer'])

def show_server_certification_issuers(url):
    trusted_CA = trusted_ca()
    print(url)
    issuer = get_certificate_issuer(url)
    print(f"Issuer of {url}: {issuer}")
    print("Trusted CA:")
    for ca in trusted_CA: 
        if issuer['organizationName'] in ca or issuer['commonName'] in ca:
            print(ca)

def main():
    wanted = ['tas_Amon_CESM2-WACCM_ssp126_r1i1p1f1_gn_201501-206412.nc']
    # 'tas_Amon_CIESM_historical_r3i1p1f1_gr_185001-201412.nc'
    try:
        #show_server_certification_issuers(DownloaderESGF.servers[0])
        datastore = DownloaderESGF(os.path.expanduser(f'~/Downloads/ClimateData/discovery/'), method='request', server=0)
        #datastore = DownloaderCopernicus(os.path.expanduser(f'~/Downloads/ClimateData/temperature/'), skip_failing_scenarios=False)
        models = ["CAMS-CSM1-0","CNRM-ESM2-1","CanESM5","CanESM5-1","EC-Earth3","EC-Earth3-Veg-LR","EC-Earth3-Veg","FGOALS-g3","GFDL-ESM4","GISS-E2-1-G","GISS-E2-1-H","IPSL-CM6A-LR","MIROC-ES2H","MIROC-ES2L","MIROC6","MPI-ESM1-2-LR","MRI-ESM2-0","UKESM1-0-LL"]
        results = datastore.download(models[5:6], ['ssp126'])
    except OpenSSL.SSL.Error: pass


if __name__ == "__main__":
    main()