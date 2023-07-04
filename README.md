# Long-term impacts of fire on forest carbon in the Sierra Nevada mountains.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Requirements
- Python 3.9+
- Packages in requirements.txt file. Can be installed via `pip install -r requirements.txt`.

## Project Summary

We use GEDI and Landsat data to look at long-term post-fire recovery of forests in Sierra Nevada. The project report containing all the background information and the summary of the analysis performed can be found at in the reports folder.

## Data availability
This project uses the following data sources:

| Data source                                           | Availability                                                                  | Sensors                                     | Data                | Date range      | No. observations (used) |
|-------------------------------------------------------|-------------------------------------------------------------------------------|---------------------------------------------|---------------------|-----------------|-------------------------|
| GEDI Level 4A - Aboveground Biomass Density                                  | [public](https://daac.ornl.gov/GEDI/guides/GEDI_L4A_AGB_Density_V2_1.html)                      | Space-borne LiDAR                           | Full-waveform LiDAR | 2019-2022       | ~9 Million                |
| Landsat 5, 7 and 8, Surface Reflectance Collection 2, Tier 1 | [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/landsat)                       | Space-borne multispectral imagery at 30m resolution | Optical bands + NDVI    | 1984-2022 | -                       |  
| NASA SRTM Digital Elevation 30m | [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003)                       | Space-borne radar | Topography: elevation, aspect, slope    | 2000  | -                       |
| USFS Landscape Change Monitoring System v2021.7 | [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/USFS_GTAC_LCMS_v2021-7)                       | Space-borne multispectral imagery |  Land cover    | 1985-2022  | -                       |
| Monitoring Trends in Burn Severity (MTBS) Burn Severity Images | [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/USFS_GTAC_MTBS_annual_burn_severity_mosaics_v1)                       | Space-borne multispectral imagery | Burn severity   | 1984-2022  | -                       |
| MTBS Burned Area Boundaries | [MTBS Direct Download](https://www.mtbs.gov/direct-download)                       | Shapefiles | Boundaries of fires larger than 1000 acres    | 1984-2022  | -                       |


**Note on data access**: In general, the data used in this project can be divided into three categories:
* MTBS Fire boundaries can be directly downloaded from the MTBS website - https://www.mtbs.gov/direct-download.
* Earth Engine data - requires [Earth Engine authentication and initialization](https://developers.google.com/earth-engine/guides/auth). The code should be able to download the necessary data upon execution.
* GEDI footprints - we worked with more than 9 million Level 4A GEDI footprints, and that data is too large to be saved here on GitHub. To obtain the data, we recommend following the instructions specified in this repo: https://github.com/ameliaholcomb/biomass-recovery/blob/main/README.md#data-download-and-setup or just working with NASA download directly.


## Project Organization
```
├── LICENSE
├── Makefile           <- Makefile with commands like `make init` or `make test_environment`
├── README.md          <- The top-level README for developers using this project.
|
├── notebooks          <- Jupyter notebooks.
│   ├── exploratory    <- Notebooks for initial exploration. Lots of half-baked code lives here, with lots of analysis that didn't make their way into the report.
│
│
├── requirements.txt   <- File containing all the required python packages. Use pip install -r requirements.txt to install them.
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── data               <- This directory mostly contains relevant shape files we used for California and Sierra Nevada. Does not contain large data like GEDI.
├── reports/figures    <- Directory that contains our results as figures.
└── src                <- Python source code.
   ├── __init__.py     <- Makes fire-regen a Python module
   │
   ├── data           <- Functions to download and process data. This includes logic for matching GEDI shots to raster data.
      ├── ee               <- Functions to fetch Google Earth Engine data.
   ├── processing     <- Functions to do main analysis.
      ├── control          <- Framework for running placebo tests on controls.
      ├── recent_fires     <- Logic to find coincident GEDI shots - i.e. shots that are in close proximity to each other.
      ├── regen            <- Uses random forest counterfactual to generate AGBD control values for all fires.
      ├── rf               <- Random Forest training, and splitting data to avoid spatial auto-correlation.
   ├── utils           <- Basic helper functions.
   └── visualization  <- Functions to plot and visualize data.
```

