# with open('ClimateProjections.py', 'r') as f: exec(f.read())
# with open('debug-snippet.py', 'r') as f: exec(f.read())
# debug()

import re
import os
import subprocess
import requests
from dotenv import load_dotenv

import fnmatch
import glob

from pyesgf.logon import LogonManager
from pyesgf.search import SearchConnection

BLUE = "\033[34m"
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
BLUEBG = "\033[44m"
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
GREY = "\033[47;30m"
HEADER = '\033[95m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
RESET = "\033[0m"

class DownloaderESGF:
    def __init__(self):
        load_dotenv(dotenv_path=os.path.expanduser('~/.esgfenv'))
        self.lm = LogonManager()
        self.servers = ['esgf-data.dkrz.de', 'esg-dn1.nsc.liu.se', 'esgf-node.ipsl.upmc.fr', 'esgf-node.llnl.gov', 'esgf-data1.llnl.gov', 'esgf.nci.org.au', 'esgf-node.ornl.gov', 'esgf.ceda.ac.uk', 'esgf-data04.diasjp.net']
        # web search https://aims2.llnl.gov/search

        if not self.lm.is_logged_on():
            self.login()

        self.connection = SearchConnection(f'https://{self.servers[-2]}/esg-search', distrib=False)

        # https://esgf.github.io/esg-search/ESGF_Search_RESTful_API.html

        self.DATADIR = os.path.expanduser('~/Downloads/ClimateAnalysis/discovery/')

    def login(self):
        user = os.getenv('ESGF_OPENID')
        print(f'Logging-in as {user}')
        self.lm.logon_with_openid(openid=user, password=os.getenv('ESGF_PASSWORD'), bootstrap=False)

    def logoff(self):
        self.lm.logoff()

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

    def download(self, models, experiments, variable='tas', frequency='mon'):
        print(f"\033[47;30m Downloading {models} {experiments} \033[0m")
        existing_files = [os.path.basename(file) for file in self.list_files('*.nc')]

        for experiment in experiments:
            for model in models:            
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

missing119 = ["CanESM5-CanOE", "KACE-1-0-G", "FIO-ESM-2-0", "INM-CM5-0", "NESM3", "ACCESS-CM2", "CNRM-CM6-1-HR", "BCC-CSM2-MR", "FGOALS-f3-L", "FGOALS-g3", "IITM-ESM", "HadGEM3-GC31-MM", "NorESM2-LM", "MPI-ESM1-2-LR", "MCM-UA-1-0", "HadGEM3-GC31-LL"]
missing126 = ["FGOALS-f3-L", "FGOALS-g3"]
missing245 = ["CanESM5", "FGOALS-f3-L", "FGOALS-g3", "HadGEM3-GC31-MM", "MCM-UA-1-0"]

in_nature = ["CanESM5", "GFDL-CM4", "FGOALS-f3-L", "HadGEM3-GC31-LL", "MCM-UA-1-0"]
in_copernicus = ['GFDL-ESM4', 'NorESM2-MM', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 'IITM-ESM', 'TaiESM1', 'NESM3', 'INM-CM4-8', 'KACE-1-0-G', 'CanESM5-CanOE', 'MPI-ESM1-2-LR', 'AWI-CM-1-1-MR', 'INM-CM5-0', 'MCM-UA-1-0', 'NorESM2-LM', 'ACCESS-CM2', 'MRI-ESM2-0', 'FGOALS-f3-L', 'FIO-ESM-2-0', 'MIROC-ES2L', 'FGOALS-g3', 'CNRM-CM6-1', 'MIROC6', 'CNRM-CM6-1-HR', 'IPSL-CM6A-LR', 'CNRM-ESM2-1', 'UKESM1-0-LL', 'HadGEM3-GC31-LL']
not_in_copernicus = ['GISS-E2-1-G', 'GFDL-CM4', 'MPI-ESM1-2-HR', 'CIESM', 'EC-Earth3-Veg', 'EC-Earth3']
already_merged_from_esgf = ['ACCESS-ESM1-5', 'CanESM5', 'CESM2', 'CESM2-WACCM', 'CMCC-ESM2', 'CMCC-CM2-SR5']


datastore = DownloaderESGF()
#datastore.logoff()
#datastore.login()
results = datastore.download(not_in_copernicus, ['ssp126', 'ssp245', 'ssp119', 'historical'])