# Data-sets: cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6

# Colab notebook: ecmwf-projects.github.io/copernicus-training-c3s/projections-cmip6.html

# API keys from cds.climate.copernicus.eu must be in ~/.cdsapirc

import cdsapi
import zipfile
import os
import matplotlib.pyplot as plt
import xarray as xr

years = range(2024, 2100+1)
months = range(1,12+1)
model = 'ec_earth3'
strategy = 'ssp1_1_9'
metric = 'near_surface_air_temperature'
area = [15, 50, 14, 51] # CZ

data_folder = 'data'

def testAvailability(strategies, models, years, months, coords): 
    years = range(*years)
    months = range(*months)

    failed = []
    passed = []
    
    c = cdsapi.Client()
    package = 'data/test.zip'
    for strategy in strategies: 
        for model in models:
            print(f'\nTRYING: {strategy} {model}')
            try:        
                c.retrieve('projections-cmip6', { 'temporal_resolution': 'monthly', 'experiment': strategy, 'variable': metric,'model': model, 'year': list([str(i) for i in years]), 'month': list([str(i) for i in months]), 'area': coords, 'format': 'zip'}, package)
                passed.append([strategy, model])
            except Exception as e: 
                failed.append([strategy, model, e])
                print('- failed')
                #print(e)
    
    print('\nMissing:')
    [print(f) for f in failed]

    print('\nAvailable:')
    [print(f) for f in passed]
    

def main():
    testAvailability(['ssp1_1_9', 'ssp1_2_6', 'ssp2_4_5', 'ssp4_3_4', 'ssp5_3_4os'], ['ec_earth3', 'ipsl_cm6a_lr', 'miroc_es2l', 'ukesm1_0_ll'], (2024, 2025), (7, 8), [12, 48, 19, 51]) #SSP5-3.4OS = high first, steeper later to the same point   
    return 
    
    testAvailability(['ssp1_1_9'], ['ipsl_cm6a_lr'], (2024, 2025), (7, 8), [12, 48, 19, 51]) #SSP5-3.4OS = high first, steeper later to the same point


    if not (os.path.exists(data_folder) and os.path.isdir(data_folder)):
        loadData()
    data = xr.open_dataset(getFilename(), engine='netcdf4')  # or engine='h5netcdf'
    averaged_data = averageData(data)
    plot(averaged_data)
    return data

def loadData():
    # credentials in ~/.cdsapirc 
    c = cdsapi.Client()
    package = 'data_climate.zip'
    c.retrieve('projections-cmip6', { 'temporal_resolution': 'monthly', 'experiment': strategy, 'variable': metric,'model': model, 'year': list([str(i) for i in years]), 'month': list([str(i) for i in months]), 'area': area, 'format': 'zip'}, package)
    with zipfile.ZipFile(package, 'r') as zip_ref:
        zip_ref.extractall(data_folder)
    os.remove(package)

def getFilename():
    nc_files = [f for f in os.listdir(data_folder) if f.endswith('.nc')]
    if len(nc_files) > 1:
        print(f"Warning: Found multiple netcdf files, using the first one: {nc_files[0]}")
    return (os.path.join(data_folder, nc_files[0]))

def averageData(data):
    average_temp = data['tas'].mean(dim=('lat', 'lon')) - 273.15 # K to °C
    return average_temp.resample(time='AS-JAN').mean('time')

def june_july_mean(data, dim):
  da_filtered = data.where((data.time.dt.month == 6) | (data.time.dt.month == 7))
  return da_filtered.mean(dim=dim)

def averageSummer(data):
    return data.resample(time=june_july_mean, dim="time")

def plot(series):
    plt.plot(series.time, series.values)
    plt.xlabel('Time')
    plt.ylabel('Global Average Temperature (K | °C)')
    plt.title('CMIP6 Global Temperature Rise Prediction (SSP1-1.9)')
    plt.show()

main()

