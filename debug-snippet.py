# with open('ClimateProjections.py', 'r') as f: exec(f.read())
# with open('debug-snippet.py', 'r') as f: exec(f.read())
# debug()

import re
import os

import fnmatch
import glob

import xarray as xr


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

ds = xr.open_dataset(['tas_Amon_UKESM1-0-LL_ssp245_r13i1p1f2_gn_205001-210012_v20190507.nc', 'tas_Amon_UKESM1-0-LL_ssp245_r1i1p1f2_gn_205001-210012_v20190507.nc'])
ds = ds['tas']
import visualisations

chart = visualisations.Charter()
chart.plot([ds], series='model')
###

import requests
custom_ca_bundle_path = '/metadata/ca-bundle.crt'

# Making a request with the custom CA bundle
response = requests.get('https://esg-dn1.nsc.liu.se/esgf-idp/openid/petr.meissner@gmail.com', verify=custom_ca_bundle_path)
print(response.content)

### Modifying Your Code: In your specific case, you could modify the `logon_with_openid` function to use the custom CA bundle:

import os
import pyesgf.logon

def login_with_custom_ca():
    user = 'your_openid'
    password = os.getenv('ESGF_PASSWORD')
    custom_ca_bundle_path = '/path/to/your/custom/ca-bundle.crt'

    lm = pyesgf.logon.LogonManager()
    lm.logon_with_openid(openid=user, password=password, ca_cert_custom=custom_ca_bundle_path, bootstrap=False)

