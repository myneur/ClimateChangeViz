# Climate Change Visualizations
Projections of global temperature or local max temperature according to CMIP IPCC models.

## Global temperature projections (CMIP6 models)
Coupled Model Intercompari-son Project, phase 6 (CMIP6), run by the World Climate Research Programme

Thare are currently 46 models available (mid 2024). 
- 28 models have both ssp126 and ssp245 
- 8 models have ssp119, ssp126 and ssp245 



### Most complete set of models covering most scenarios (so far)
![Global temperature projections (CMIP6 models)](charts/latest_most_complete.png)

### Most models covering 1.5 °C scenario (ssp119)
![Global temperature projections (CMIP6 models)](charts/latest_ssp119.png)

### Context (Work in progress)
This is an exploration so far. There are models considered hot and outliers and couple of more models. These are not distinguished here yet. 
Models with TCR (Transient Climate Response) 1.4-2.2 are deemed more likely in AR6. 
Averages by 50th quantile. Ranges by 10-90th quantile.

[Hot models problem, Nature](https://www.nature.com/articles/d41586-022-01192-2.epdf)

## Maximal temperature (in Czechia) projections (CMIP6 models)
![Local temperature max projections (CMIP6 models)](charts/latest_max.png)
Averages by 50th quantile. Ranges by 10-90th quantile.

## Tropic days annualy (in Czechia) projections (CMIP6 models)
![Local tropic days in summer months with the max temperature over 30 °C projections (CMIP6 models)](charts/latest_tropic.png)

### Context (Work in progress, exploration)
This is an early exploration of what data is available so far. The max temperature projections seem to be varying more and daily predictions are available in less number of models. 

# How to run it
1. install python libraries in `requirements.txt`
2. register on (https://cds.climate.copernicus.eu/) and get an API key in your user profile
3. put your API key to `~/.cdsapirc` as [in this format](https://cds.climate.copernicus.eu/api-how-to)
4. change `DATADIR` to point where to download model data
5. Run `python ClimateProjections.py`

Tested only on python3.
