# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: hdc
#     language: python
#     name: conda-env-hdc-py
# ---

# %%
import xarray as xr
import os
from tqdm import tqdm

input_dir = "/s3/scratch/jamie.towner/flood_aa/zimbabwe/data/forecasts/glofas_reforecasts/stations"
output_dir = "/s3/scratch/jamie.towner/flood_aa/zimbabwe/data/forecasts/glofas_reforecasts"
os.makedirs(output_dir, exist_ok=True)

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
