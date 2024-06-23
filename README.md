# Climate Change Visualizations
Projections of global temperature or local max temperature according to CMIP IPCC models.

## Global temperature projections (CMIP6 models)
![Global temperature projections (CMIP6 models)](charts/latest.png)
Averages by 50th quantile. Ranges by 10-90th quantile.

### Work in progress
Currently just a subset of models is read correctly. Despite the trend is representative, the real values might differ. 

## Maximal temperature (in Czechia) projections (CMIP6 models)
![Local temperature max projections (CMIP6 models)](charts/latest_max.png)
Averages by 50th quantile. Ranges by 10-90th quantile.

## Tropic days annualy (in Czechia) projections (CMIP6 models)
![Local tropic days in summer months with the max temperature over 30 °C projections (CMIP6 models)](charts/latest_tropic.png)

### Work in progress (exploration)
This an exploration so far. The max temperature projections seems to be varying a lot and predictions are available in less number of models. 
Alternative is to explore averages of daily forecasts for summer months.

# How to run it
1. install python libraries in `requirements.txt`
2. register on (https://cds.climate.copernicus.eu/) and get an API key in your user profile
3. put your API key to `~/.cdsapirc` as [in this format](https://cds.climate.copernicus.eu/api-how-to)
4. change `DATADIR` to point where to download model data
5. Run `python ClimateProjections.py`

Tested only on python3.
