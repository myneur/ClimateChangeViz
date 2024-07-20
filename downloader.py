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

def run_once(f):
    def wrapper(self, *args, **kwargs):
        if not getattr(self, '_decorated', False):
            f(self, *args, **kwargs)
            self._decorated = True
    return wrapper

class Downloader:
    def __init__(self, DATADIR, mark_failing_scenarios=False, skip_failing_scenarios=False, forecast_from=None, start=None, end=None, fileformat='zip'): 
        self._parent = self.DATADIR = DATADIR
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

    @run_once
    def setup(self):
        os.makedirs(self.DATADIR, exist_ok=True)

    def set(self, variable, frequency, area=None):
        self.variable = variable
        self.frequency = frequency
        self.area = area
        subfolder = f'{variable}_{frequency}'
        if area:
            subfolder += f"_{'_'.join(map(str, area))}"
        self.DATADIR = os.path.join(self.DATADIR, subfolder, '')

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = None     

    def login(self):
        import cdsapi
        self.client = cdsapi.Client() # Doc: https://cds.climate.copernicus.eu/toolbox/doc/how-to/1_how_to_retrieve_data/1_how_to_retrieve_data.html

    def download(self, models, experiments, forecast_from=2015, start=1850, end=2100): 
        self.setup()
        variable = self.variable
        frequency=self.frequency
        area = self.area
        
        if not self.client: self.login()

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
                            #if frequency == 'day': params['month'] = ['O4', '05' '06', '07', '08', '09']
                            
                            print(f'{BLUE}REQUESTING: {model} {experiment} for {date}{RESET}')
                            
                            self.client.retrieve('projections-cmip6', params, filename)
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

    def reanalysis(self): # retrieve historical measurementss
        raise NotImplementedError
        self.client.retrieve('reanalysis-era5-single-levels', {
            'product_type': 'reanalysis', 
            'variable': '2m_temperature'
            #'year': list(range(1910,1918+1)),
            #'area': [51, 12, 48, 18]
            })


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

    def __init__(self, DATADIR, server=0, method='wget', fileformat='nc', mark_failing_scenarios=False, skip_failing_scenarios=False):
        super().__init__(DATADIR, fileformat=fileformat, mark_failing_scenarios=mark_failing_scenarios, skip_failing_scenarios=skip_failing_scenarios)
        self.max_tries = 5
        self.retry_delay = 10
        self.current_server = server%len(self.servers)
        self.downloadMethod = method

    @run_once
    def setup(self):
        super().setup()
        from pyesgf.logon import LogonManager
        from pyesgf.search import SearchConnection
        load_dotenv(dotenv_path=os.path.expanduser('~/.esgfenv'))
        self.lm = LogonManager()
        self.connection = SearchConnection(f'https://{DownloaderESGF.servers[self.current_server]}/esg-search', distrib=True)
        # https://esgf.github.io/esg-search/ESGF_Search_RESTful_API.html
        #os.environ['ESGF_PYCLIENT_NO_FACETS_STAR_WARNING'] = '1'
        self.trustCertificate()
        
    def trustCertificate(self): # certificates we trust on top of the system-wise
        os.environ['REQUESTS_CA_BUNDLE'] = 'trusted-certificates.pem'

    def login(self):
        user = os.getenv('ESGF_OPENID')
        print(f'Logging-in as {UNDERLINE}{user}{RESET}')
        try:
            self.lm.logon_with_openid(openid=user, password=os.getenv('ESGF_PASSWORD'), bootstrap=True)
        except OpenSSL.SSL.Error as e:
            issuer = self.get_certificate_issuer(user)['issuer']
            print(f"{RED}Your python does not trust the certificate of the Data Service.\nCertificate Issuer: O:{issuer[1][0][1]} CN: {issuer[2][0][1]}{RESET}")
            print("1. check the issuer of the certificate is trustworthy. E. g. By checking your browser trusts it when you visit the domain above.")
            print("2. add the certificate into trusted ones by the following terminal shell command:")
            url = user.split('/')[2]
            print(f"echo | openssl s_client -connect {url}:443 -servername {url} 2>/dev/null | openssl x509 >> trusted-certificates.pem")
            raise(e)

    def logoff(self):
        self.lm.logoff()


    def search(self, model, experiment, forecast_from=None, area=None, variable=None, frequency=None):
        
        self.setup()
        variable = variable if variable else self.variable
        frequency = frequency if frequency else self.frequency
        area = area if area else self.area
        if not self.lm.is_logged_on(): self.login()

        print(f'Searching for {variable} {frequency} {model} {experiment} {area}')

        for attempt in range(self.max_tries):
            try:
                print(f'Try {attempt} {model} {experiment}', end='\r')
                results = []
                params = {
                    'facets': 'data_node,variant_label,version', 
                    'retracted': False, 'latest': True, # latest don't include retracted    
                    'variable':variable, 'frequency': frequency, 'project': 'CMIP6'}
                if model: params['source_id'] = model
                else: params['facets'] += ',source_id'
                if experiment: params['experiment_id'] = experiment
                else: params['facets'] += ',experiment_id'

                if area:
                    #raise NotImplementedError("Area constraint is not implemented by DownloaderESGF yet, use DownloaderCopernicus")
                    print("Warnging: Area constraint is not implemented by DownloaderESGF yet, use DownloaderCopernicus if possible.")
                    params['bbox'] = area #bbox=[W,S,E,N] for ESGF [N,W,S,E] for Copernicus

                if forecast_from: # not implemented yet
                    pass
                    #print(f'{RED}Datum constraint not implemented for ESGF yet{RESET}')
                    #start="2100-12-30T23:23:59Z", to_timestamp="2200-01-01T00:00:00Z",
                    
                context = self.connection.new_context(**params)
                print()
                # Show what variants are available
                # [print(f'{facet} {counts}') for facet, counts in context.facet_counts.items()]
                print(', '.join([f'{facet}: {counts}' for facet, counts in context.facet_counts.items() if facet == 'data_node']))
                
                print(f'Found {context.hit_count}× {model} {experiment}')
                
                return context

            except requests.exceptions.Timeout as e:
                if attempt < self.max_tries:
                    print(f"Timeout. Retrying search in {self.retry_delay} s:\n{type(e).__name__}: {e}")
                    time.sleep(self.retry_delay)
                else:
                    print(f'❌ download search failed {model} {experiment}: Timeout'); 
                    return {'hit_count': 0}
            except (requests.exceptions.RequestException, Exception) as e:
                print(f'❌ download search failed {model} {experiment}: {type(e).__name__}: {e}'); traceback.print_exc()
                return {'hit_count': 0}

    def models_available_for(self, experiment):
        context = self.search(None, [experiment])
        if context.hit_count:
            for facet, counts in context.facet_counts.items():
                if counts and facet in ('source_id'):
                    return counts
                    
        else: return {}


    def download(self, models, experiments, forecast_from=None, area=None):
        # TODO forecast_from & area not implemented yet
        variable = self.variable
        frequency = self.frequency
        area = self.area

        print(f"{BLUE}Downloading {BOLD}{models} {experiments}{RESET}")
        existing_files = [os.path.basename(file) for file in self.list_files('*.nc')]
        
        for model in models:
            for experiment in experiments:
                downloaded = False
                if not self.file_in_list(existing_files, f'{variable}*_{model}_{experiment}*.nc'):
                    
                    context = self.search(model, experiment, forecast_from=forecast_from, area=area)  
                    # [print(f'{facet} {counts}') for facet, counts in context.facet_counts.items()]
                    
                    if(context.hit_count):
                        results = context.search()
                        print(f'{BLUE}⬇{RESET} downloading {results[0].dataset_id}')
                        #results = sorted(results, key=self.splitByNums, reverse=True) 

                        #for result in results: print(f"{UNDERLINE}{result.dataset_id.split('|')[1]}{RESET} {result.number_of_files} files")
                            # print(r.urls['THREDDS'][0][0].split('/')[2])
                            # print(dir(r.context)) #'connection', 'constrain', 'facet_constraints', 'facet_counts', 'facets', 'fields', 'freetext_constraint', 'geospatial_constraint', 'get_download_script', 'get_facet_options', 'hit_count', 'latest', 'replica', 'search', 'search_type', 'shards', 'temporal_constraint', 'timestamp_range'
                            #print(r.json)
                            
                        # [print(f'{UNDERLINE}{node}{RESET}: {counts} facets') for node, counts in context.facet_counts['data_node'].items()]

                        #for result in results: # TODO pick just one variant per server
                        if results[0]:
                            result = results[0]
                            server = result.dataset_id.split('|')[1]
                            try:
                                if self.downloadMethod == 'request':
                                    context = result.file_context()
                                    # context.facet 'data_node' 'index_node' 'data_specs_version' 'nominal_resolution'
                                    #[print(f'{facet} {counts}') for facet, counts in context.facet_counts.items() if counts]
                                    
                                    for file in context.search(facets='variant_label,version,data_node'):
                                        # for facet, counts, in context.facet_counts.items():print(f'facet :{facet}');for value, count in counts.items():print(f'{value}: {count}')
                                        # 'context', 'download_url', 'file_id', 'filename', 'index_node', 'json',  'size', 'tracking_id', 'urls'
                                        downloaded = self.downloadUrl(file.download_url)
                                    
                                    if downloaded:
                                        break
                                else:
                                  if self.downloadWget(result, model, experiment, variable): 
                                    downloaded = True
                                    break
                            # TODO when we group by server, we should switch server here #except requests.exceptions.ConnectionError as e: print(f"❌ server {server} Timeout: {type(e).__name__}: {e}")
                                # ConnectionError: HTTPConnectionPool(host='{data_node}', port=80): Max retries exceeded with url: /thredds/fileServer/{filepathname} (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at {object}>: Failed to establish a new connection: [Errno 60] Operation timed out'))
                                
                            except (requests.exceptions.TooManyRedirects, requests.exceptions.RequestException, Exception) as e:
                                print(f"❌ server {server} failed: {type(e).__name__}: {e}"); traceback.print_exc(limit=1)

                        # 'number_of_files', 'las_url', 'urls', 'context',  'opendap_url', 'globus_url', 'gridftp_url', 'index_node', 'json', 
                    else:
                        print(f'❌ missing {model} {experiment}')
                        continue
                else:
                    print(f'✅ exists {model} {experiment}')
                    continue
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
                print(file, os.path.getsize)
                if os.path.getsize(file) == 0:  # crashed wget leaves empty file that would prevent downloading the next time
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
    #return dict(x[0] for x in cert['issuer'])

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
        #datastore = DownloaderCopernicus(os.path.expanduser(f'~/Downloads/ClimateData/discovery/'))
        datastore = DownloaderESGF(os.path.expanduser(f'~/Downloads/ClimateData/discovery/'), method='wget') #wget|request
        datastore.set(
            #area=[51, 12, 48, 18], # not supported yet
            'tasmax', 'day') # temperature above surface max
        
        os.environ['ESGF_PYCLIENT_NO_FACETS_STAR_WARNING'] = '1'

        print(datastore.models_available_for('ssp245'))

        #datastore.download(["EC-Earth3-CC"], ['ssp245'])
        #results = datastore.download(models, ['ssp126'])

    except Exception as e:
        print(f"\n{type(e).__name__}: {e}"); traceback.print_exc(limit=1)

if __name__ == "__main__":
    main()