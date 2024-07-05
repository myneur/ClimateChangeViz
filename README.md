# Climate Change Visualizations

What are the latest projections of climate change?
How likely are they?

Let's visualize the latest knowledge of all CMIP6 models and their variability.

## Global temperature projections

### Models covering most scenarios
![Global temperature projections (CMIP6 models)](charts/latest_most_complete.png)

### Models covering 1.5 °C scenario (ssp119)
![Global temperature projections (CMIP6 models)](charts/latest_ssp119.png)

## Maximal temperature (in Czechia) projections
![Local temperature max projections (CMIP6 models)](charts/latest_max.png)

## Tropic days annualy (in Czechia) projections
![Local tropic days in the summer months with max temperatures over 30 °C projections (CMIP6 models)](charts/latest_tropic.png)

## Context: Work in progress
Averages by the 50th quantile. Ranges by the 10-90th quantile.

Model data from [_Copernicus Climate Data Store_ ]((https://cds.climate.copernicus.eu/api-how-to)) and [_Earth System Grid Federation Data Portal_](https://aims2.llnl.gov/).
Measurement data from [_Met Office_](https://climate.metoffice.cloud/current_warming.html)

This is yet to be done.

## Where are the projections from?

The most widely recognized projections are from the __Coupled Model Intercomparison Project, phase 6 (CMIP6)__, run by the _World Climate Research Programme_.

- __CMIP__ is a collaborative framework designed to improve our understanding of climate change through the systematic comparison of climate models. Their work is publicly shared with anyone for use or review.

- CMIP shares __around 40 distinct climate models__ (mid 2024) created by [49 modelling groups](https://wcrp-cmip.github.io/CMIP6_CVs/docs/CMIP6_institution_id.html) from __17 countries all across the world__ from Europe, America, Asia, Australia and Oceania.

## Variance in climate models

To interpret the variance among the CMIP models, the Intergovernmental Panel on Climate Change (IPCC)_ assesses the accuracy of the models by applying statistics to determine which model projections are more consistent with lines of evidence. They combined models by weighting them by their likelihood into an ‘__assessed warming__’ projection in their Sixth Assessment Report (AR6).

Currently, the biggest debate is about so-called _'hot models'_ that predict the highest warming. In the IPCC 'assessed warming', they are considered to be less likely, because they less accurately reproduce historical temperatures over time.

Alternatively, a subset of models with __'TCR'__ between 1.4-2.2 ºC can be used to produce very close results, as described in the [Nature.com article](http://doi.org/10.1038/d41586-022-01192-2). _Nature_, is one of the top scientific journals recognized for its rigorous peer-review process involving multiple rounds of evaluation by experts in the field.

- __TCR: Transient Climate Response__ is the amount of global warming in the year in which atmospheric CO2 concentrations have finally doubled after having steadily increased by 1% every year.

- __IPCC: Intergovernmental Panel on Climate Change__ is an international body established by the _United Nations Environment Programme (UNEP)_ and the World Meteorological Organization (WMO) to assess the scientific, technical, and socio-economic information relevant to understanding the risk of human-induced climate change.
- - The IPCC's assessments are based on the work of thousands of scientists from around the world and are widely regarded as the most comprehensive source of information on climate change.
The IPCC does not conduct its own research but synthesizes existing scientific literature, including that made by CMIP. Its reports go through a rigorous process of review and revision involving experts to ensure accuracy, transparency. They provide policymakers with regular scientific assessments of the current state of knowledge about climate change.

## Future Scenarios

> The largest source of uncertainty in global temperatures in decades to come is the volume of future greenhouse-gas emissions, which are largely under human control.

The scenarios visualized:

1. Reaching worldwide carbon neutrality around 2050 => projected 1.5° global warming – probably out of our reach (ssp119)
2. Carbon neutrality around 2075 => projected 2° global warming – will we make it? (ssp126)
3. No decline of emissions until half of millenia => projected 3° global warming (ssp245)
4. Double emissions in 2100 => projected 4° global warming (ssp570)


# How to run it
1. install python libraries in `requirements.txt`
2. register on (https://cds.climate.copernicus.eu/) and get an API key in your user profile
3. put your API key to `~/.cdsapirc` as [in this format](https://cds.climate.copernicus.eu/api-how-to)
4. change `DATADIR` to the folder point where to download model data
5. Run `python ClimateProjections.py`

Tested only on python3.