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
# ### Split forecasts into individual netcdf files
# This will prevent memory errors in the following scripts

# %%
import xarray as xr
import os
from tqdm import tqdm
from pathlib import Path

# %%
country = 'mozambique'  # define country of interest

# %%
directory = Path(r"C:\Users\15133\Documents\WFP\flood_hazard\flood_aa\MOZ_training")  # define main working directory
input_dir = directory / "data/forecasts/glofas_reforecasts/stations"
output_dir = directory / "data/forecasts/glofas_reforecasts"
os.makedirs(output_dir, exist_ok=True)

# %%
for filename in tqdm(os.listdir(input_dir)):
    if not filename.endswith(".nc"):
        continue
    
    station_name = filename.replace(".nc", "")
    ds = xr.open_dataset(os.path.join(input_dir, filename))

    times = ds['time'].values

    for i, t in enumerate(times):
        # Select data for a single forecast issue date
        ds_sel = ds.isel(time=i).expand_dims('time')

        # Format date
        date_str = str(t)[:10].replace('-', '_')  # YYYY_MM_DD

        # Build output filename
        out_filename = f"{station_name}_{date_str}.nc"
        out_path = os.path.join(output_dir, out_filename)

        # Save to file
        ds_sel.to_netcdf(out_path)


# %%
