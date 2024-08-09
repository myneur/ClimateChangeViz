# Climate Change Visualizations

What are the latest projections of climate change?

How likely are they?

Let's visualize the latest knowledge of all CMIP6 models and their variability.

## Global temperature projections

### The widest selection of models covering the main scenarios where emissions won't grow
![Global temperature projections (CMIP6 models)](charts/latest_most_complete.svg)


### Models covering also 1.5 °C scenario (ssp119) of the Paris agreement
This scenario now seems out of our reach.

![Global temperature projections (CMIP6 models)](charts/latest_ssp119.svg)

### Context
Averages by the 50th quantile. Ranges by the 10-90th quantile. Normalized to the last 20 years of hindcast. 

Models availability (Aug '24) for the whole period: 

> 18 models for all 3 scenarios: `CAMS-CSM1-0, CanESM5, CanESM5-1, CNRM-ESM2-1, EC-Earth3, EC-Earth3-Veg, EC-Earth3-Veg-LR, FGOALS-g3, GFDL-ESM4, GISS-E2-1-G, GISS-E2-1-H, IPSL-CM6A-LR, MIROC-ES2H, MIROC-ES2L, MIROC6, MPI-ESM1-2-LR, MRI-ESM2-0, UKESM1-0-LL`

> 47 models for ssp126 and ssp245: `Previous 18 + ACCESS-CM2, ACCESS-ESM1-5, AWI-CM-1-1-MR, BCC-CSM2-MR, CanESM5-CanOE, CAS-ESM2-0, CESM2, CESM2-WACCM, CIESM, CMCC-CM2-SR5, CMCC-ESM2, CNRM-CM6-1, CNRM-CM6-1-HR, FGOALS-f3-L, FIO-ESM-2-0, GISS-E2-1-G-CC, GISS-E2-2-G, HadGEM3-GC31-LL, IITM-ESM, INM-CM4-8, INM-CM5-0, KACE-1-0-G, KIOST-ESM, MCM-UA-1-0, MPI-ESM1-2-HR, NESM3, NorESM2-LM, NorESM2-MM, TaiESM1`

> 48 models for at least one of the scenarios: `Previous 47 + EC-Earth3-CC`


## Variance in climate models

To interpret the variance among the CMIP models, the Intergovernmental Panel on Climate Change (IPCC)_ assesses the accuracy of the models by applying statistics to determine which model projections are more consistent with lines of evidence. They combined models by weighting them by their likelihood into an ‘__assessed warming__’ projection in their Sixth Assessment Report (AR6).

Currently, the biggest debate is about so-called _'hot models'_ that predict the highest warming. In the IPCC 'assessed warming', they are considered to be less likely, because they less accurately reproduce historical temperatures over time.

Alternatively, a subset of models with __'TCR'__ between 1.4-2.2 ºC can be used to produce very close results, as described in the [Nature.com article](http://doi.org/10.1038/d41586-022-01192-2). _Nature_, is one of the top scientific journals recognized for its rigorous peer-review process involving multiple rounds of evaluation by experts in the field.

![Global temperature projections by model (CMIP6 models)](charts/models_classified.svg)


- __TCR: Transient Climate Response__ is the amount of global warming in the year in which atmospheric CO2 concentrations have finally doubled after having steadily increased by 1% every year.

- __IPCC: Intergovernmental Panel on Climate Change__ is an international body established by the _United Nations Environment Programme (UNEP)_ and the World Meteorological Organization (WMO) to assess the scientific, technical, and socio-economic information relevant to understanding the risk of human-induced climate change.
- - The IPCC's assessments are based on the work of thousands of scientists from around the world and are widely regarded as the most comprehensive source of information on climate change.
The IPCC does not conduct its own research but synthesizes existing scientific literature, including that made by CMIP. Its reports go through a rigorous process of review and revision involving experts to ensure accuracy, transparency. They provide policymakers with regular scientific assessments of the current state of knowledge about climate change.

## Future Scenarios

> The largest source of uncertainty in global temperatures in decades to come is the volume of future greenhouse-gas emissions, which are largely under human control.

The scenarios visualized:

1. Reaching worldwide carbon neutrality around 2050 => projected 1.5° global warming – probably out of our reach (ssp119)
2. Carbon neutrality around 2075 => projected 2° global warming – will we make it? (ssp126)
3. No decline of emissions until 2050 => projected 3° global warming (ssp245)

There are even more pessimistic scenarios like "Double emissions in 2100 => projected 4° global warming (ssp570)" that could be visualized.


## Maximum temperature projections (WIP)

### Maximum temperature in Czechia
![Local temperature max projections (CMIP6 models)](charts/latest_max.svg)

### Tropic days annualy in Czechia
![Local tropic days in the summer months with max temperatures over 30 °C projections (CMIP6 models)](charts/latest_tropic.svg)

### Context: Work In Progress

This is an early discovery. The max temperature aggregation is yet to be reviewed. It seems that the model resolution lowers the maximum temperatures a bit and more complex approach would be needed for more accurate values, not just the trend. 

For example a [CLIMRISK](https://www.climrisk.cz/mapa-cr/) uses a statistical interpolation and avaluates model historical accuracy. 

## Where are the projections from?

The most widely recognized projections are from the __Coupled Model Intercomparison Project, phase 6 (CMIP6)__, run by the _World Climate Research Programme_.

- __CMIP__ is a collaborative framework designed to improve our understanding of climate change through the systematic comparison of climate models. Their work is publicly shared with anyone for use or review.

- CMIP shares __around 50 distinct climate models__ (Aug 2024) created by [49 modelling groups](https://wcrp-cmip.github.io/CMIP6_CVs/docs/CMIP6_institution_id.html) from __17 countries all across the world__ from Europe, America, Asia, Australia and Oceania.

### Data Sources
Model data from [_Copernicus Climate Data Store_]((https://cds.climate.copernicus.eu/api-how-to)) or [_Earth System Grid Federation Data Portal_](https://aims2.llnl.gov/).

Measurement data from [_Met Office_](https://climate.metoffice.cloud/current_warming.html) and [_Czech Hydrometeorological Institute_](https://www.chmi.cz/historicka-data/pocasi/denni-data/data-ze-stanic-site-RBCN)


# How to run it
1. install python libraries in `requirements.txt`
2. Register on Data Store: 
    1. Copernicus (https://cds.climate.copernicus.eu/):
    	- get an API key in your user profile
    	- put your API key to `~/.cdsapirc` as [in this format](https://cds.climate.copernicus.eu/api-how-to)
    2. ESGF (https://esgf.github.io/nodes.html) – any node:
    	- put the open-id you'll get after the registration to `ESGF_OPENID=***` & `ESGF_PASSWORD=***` lines into `~/.esgfenv`
4. change `DATADIR` to the folder point where to download model data
	- Toggle DownloaderCopernicus/DownloaderESGF to get missing models on either of them
5. Run `python3 ClimateProjections.py`
	- `preview` to skip download and visualize already downloaded data
	- `max` to visualize the local maximum temperature instead of the global average
	- `esgf` to download from the ESGF instead of Copernicus data store. `wget` to use their script instead the direct download.
	- Update `model` `experiment` in the code when needed. 

Tested only on python3.

## Downloading

Downloading is tricky, because not all models are available on all servers. Multiple download methods might be necessary to use when the method is unable to download a particular model. 

- DownloaderCopernicus is easier to start with, but more models are missing there. 
- DownloaderESGF has more models, but can be hard to tame (run with `esgf` parameter). It might be necessary to add trusted certificates (an error suggests how) or try using their scripts with some dependencies (run with `wget` parameter). 

Also, sometimes a server timeouts, so it might be necessary to wait hours before retrying. Somebody is welcome to contribute by making it more robust.
Plus at this point the ESGF python library does not limit data by geographic coordinates, which makes it cumbersome for huge daily models. The program visualizes just the aggregations starting with `agg*`, so the big source files can be deleted once aggregated.

If you break the download in the process, remove unfinished files not to have holes in the data. One serie is sometimes split into multiple files, so all files with the last ssp experiment and model names are recommended to be deleted in such case to make downloader redownload it. Watch for yellow warnings of duplicates or incomplete data. 

All models have tens of Gigs. It takes time.