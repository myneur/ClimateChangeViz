# Climate Change Visualizations
Projections of global temperature or local max temperature according to CMIP IPCC models.

## Global temperature projections (CMIP6 models)
![Global temperature projections (CMIP6 models)](charts/latest.png)

### Work in progress
Currently just a subset of models is read correctly. Despite the trend is representative, the real values might differ. 
Averages by 50th quantile. Ranges by 10-90th quantile.

## Maximal temperature (in Czechia) projections (CMIP6 models)
![Local temperature max projections (CMIP6 models)](charts/latest_max.png)

### Work in progress (exploration)
This is just an exploration. The max temperature projections seems to be varying a lot and predictions are available in less number of models. 
Alternative is to explore averages of daily forecasts for summer months.
Averages by 50th quantile. Ranges by 10-90th quantile.


# How to run it
1. install python libraries in requirements.txt
2. register on (https://cds.climate.copernicus.eu/) and get API keys in the user profile
3. put API keys to ~/.cdsapirc
4. change DATADIR where data are downloaded
5. Run python ClimateProjections.py

Tested just on python3.
