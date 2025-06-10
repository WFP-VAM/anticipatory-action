# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Run this notebook using the HDC environment.

# ### Imports

# +
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.simplefilter("ignore")
# -

# ### Read cyclones data

cyclones = pd.read_csv("MOZ_overland_ibtracs.since1980.list.v04r00.txt")

cyclones

# ### Read blended data

blended = xr.open_zarr(
    "/s3/scratch/public-share/long_term_blending/MOZ/chirp_daily_blended_corrected_dek.zarr"
)

# Mask no data
blended = blended.where(blended != blended.daily_blended.nodata, np.nan)

blended

# ### Add blended data into cyclones dataframe

# +
for i, row in cyclones.iterrows():
    t = pd.to_datetime(row.ISO_TIME[:10])
    lon = row.LON
    lat = row.LAT

    try:
        cyclones.loc[i, "BLENDED_CHIRP"] = (
            blended.sel(time=t)
            .sel(longitude=lon, latitude=lat, method="nearest")
            .daily_blended.values
        )
    except IndexError:
        cyclones.loc[i, "BLENDED_CHIRP"] = np.nan
# -

cyclones

# ### Save updated dataframe

cyclones.to_csv("MOZ_overland_ibtracs.since1980.list.v04r00.blended.txt")
