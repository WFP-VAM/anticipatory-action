# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python (Pixi)
#     language: python
#     name: pixi-kernel-python3
# ---

# %% [markdown]
# # Loading Seasonal Forecasts from CDS
#
# This notebook demonstrates how to load **seasonal forecasts** from the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/how-to-api):
# - Connect to the CDS API
# - Request seasonal forecast data for a specific country and time period
# - Visualize the data and compute zonal statistics
#
# **Note:** For training purposes, we only load **one lead time (24h)** and **one year** to minimize runtime and data size. In practice, you can fetch **all lead times** and **multiple years**.
#
# Additionally, you could compute **Standardized Precipitation Index (SPI)** or **drought probabilities** from the downloaded data.

# %% [markdown]
# ## 1. Environment Setup
#
# *You only need to run this cell once. If the library is already installed in your environment, you can skip this step.*
#
# Install required packages. Ensure `cdsapi` is available for CDS API access.

# %%
# !pixi add cdsapi

# %% [markdown]
# ## 2. Import Libraries
# We use `cdsapi` for data retrieval, `xarray` for handling NetCDF files, and `odc.geo` for geospatial operations.

# %%
import os

if os.getcwd().split("\\")[-1] != "anticipatory-action":
    os.chdir("..")
print(os.getcwd())

import cdsapi
import xarray as xr
import pandas as pd
from odc.geo.xr import xr_reproject
from config.params import Params
from hip.analysis.aoi.analysis_area import AnalysisArea

# %% [markdown]
# ## 3. Define Parameters
# - `country`: ISO code of the country
# - `issue`: Month of forecast issuance
# - `start_year`, `end_year`: Time range (we use only one year for speed)
# - `data_path`, `output_path`: Paths for input/output

# %%
country = "ISO"  # Replace with ISO code of the country
issue = 12  # Forecast issue month (December)
start_year = 1981
end_year = 2025  # For training, we only use the last year

data_path = "."  # anticipatory-action directory
output_path = "."

# %% [markdown]
# ## 4. Define Area of Interest (AOI)
# We use `AnalysisArea` to get administrative boundaries and resolution.

# %%
area = AnalysisArea.from_admin_boundaries(
    iso3=country,
    admin_level=0,
    resolution=0.25,
)
shp = area.get_dataset([area.BASE_AREA_DATASET])

# %% [markdown]
# ## 5. Connect to CDS API
# Initialize the CDS API client.

# %%
client = cdsapi.Client()

# %% [markdown]
# ## 6. Prepare Data Request
# We request **seasonal-original-single-levels** dataset from ECMWF system 51.
#
# **Important:**
# - We only request **one lead time (24h)** and **one year** for speed.
# - You can extend this to all lead times and years.

# %%
dataset = "seasonal-original-single-levels"
request = {
    "originating_centre": "ecmwf",
    "system": "51",
    "variable": ["total_precipitation"],
    "year": end_year,  # Uncomment this to load all years: [str(y) for y in range(start_year, end_year + 1)]
    "month": [str(issue)],
    "day": ["01"],
    "leadtime_hour": "24",  # Could be [str(lt) for lt in range(24, 5184, 24)] for all leadtimes
    "data_format": "netcdf",
    "area": [area.bbox[-1], area.bbox[0], area.bbox[1], area.bbox[2]],
}
target = f"{data_path}/data/zmb/zarr/2022/{issue}/ecmwf_cds_24h.nc"

# %% [markdown]
# ## 7. Retrieve Data
# This step downloads the NetCDF file from CDS in `{data_path}/data/zmb/zarr/{end_year}/{issue}/ecmwf_cds_24h.nc`

# %%
client.retrieve(dataset, request, target)

# %% [markdown]
# ## 8. Load and Inspect Data
# Use `xarray` to open the NetCDF file and attach CRS.

# %%
da = xr.open_dataset(target).tp.rio.write_crs("epsg:4326")
date = pd.to_datetime(da.forecast_reference_time.values[0]).strftime("%Y-%m-%d")

# %%
da

# %% [markdown]
# ## 9. Visualize Forecast
# Display the forecast for the AOI at different resolutions.

# %%
da.isel(forecast_reference_time=0, forecast_period=0, number=0).rio.write_crs(
    "EPSG:4326"
).rio.clip(shp.geometry).hip.viz.map(title=f"{date} - 24h leadtime - 1 degree")

# %% [markdown]
# The following map shows the forecast at a 0.25 degree resolution. This forecast is obtained by reprojecting the native one using the bilinear interpolation method.

# %%
xr_reproject(da, area.geobox, resampling="bilinear").isel(
    forecast_reference_time=0, forecast_period=0, number=0
).dropna("latitude", how="all").dropna("longitude", how="all").rio.clip(
    shp.geometry
).hip.viz.map(
    title=f"{date} - 24h leadtime - 0.25 degree"
)

# %% [markdown]
# ## 10. Compute Zonal Statistics
# Aggregate forecast values by administrative boundaries (admin-2 level).

# %%
area = AnalysisArea.from_admin_boundaries(iso3=country, admin_level=2, resolution=0.25)
admin_fc = (
    area.zonal_stats(
        da.isel(forecast_reference_time=0, forecast_period=0, number=0), stats=["mean"]
    )
    .reset_index()
    .assign(time=date)
    .set_index(["zone", "time"])
)
zonal_gdf = area.join_zonal_stats(admin_fc["mean"])
zonal_gdf.hip.viz.map(
    title=f"{date} - 24h leadtime - admin 2", column=date, annotate="Name", legend=True
)

# %% [markdown]
# ## Next Steps
# - Fetch all lead times and years for full analysis
# - Adjust the format: convert cumulative rainfall to daily amounts
# - Derive SPI and drought probabilities from complete dataset
