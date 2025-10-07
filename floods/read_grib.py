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
# ### This script shows how to read grib or netcdf data, visualize, and extract to csv

# %%
import xarray as xr
import rioxarray as rx
from pathlib import Path
import matplotlib.pyplot as plt

# %%
directory = Path(r"C:\Users\15133\Documents\WFP\flood_hazard\flood_aa\MOZ_training")  # define main working directory

# read in forecast grib or netcdf
# for netcdf just change engine to "netcdf4" instead of "cfgrib"
fcst = xr.open_dataset(directory / "glofas_20251006.grib", engine="cfgrib")
fcst

# %% [markdown]
# #### visualize forecast average over the whole area

# %%
fcst["dis24"].mean(dim=["number","step"]).plot()

# %% [markdown]
# #### select one point along river and plot the ensemble mean

# %%
fcst["dis24"].sel(latitude=-24.525,longitude=33.025,method="nearest").mean(dim="number").plot()

# %% [markdown]
# #### plot ensemble distribution and ensemble mean

# %%
# select point of interest
loc_da = fcst["dis24"].sel(latitude=-24.525,longitude=33.025,method="nearest") 

# create figure 
fig, ax = plt.subplots()
# plot all ensembles
loc_da.plot.line(x="step", add_legend=False, color="black", alpha=0.25, ax=ax)
# plot ensemble mean
loc_da.mean(dim="number").plot.line(color="red", ax=ax)

# %% [markdown]
# #### save location of interest as a csv

# %%
df = loc_da.to_dataframe()
df

# %%
df.to_csv(directory / "current_forecast.csv")

# %%
