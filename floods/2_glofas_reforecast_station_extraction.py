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

# %% [markdown]
# Script to extract GloFAS reforecast data at station locations stored in an s3 bucket. Metadata file is used to identify which station points to extract (use Lisflood x and y coordinates if available).

# %%
import s3fs
import dask
import xarray as xr
import pandas as pd
import os
from tqdm import tqdm
from hip.analysis.compute.utils import persist_with_progress_bar

# %%
country = 'mozambique'  # define country of interest
directory = '/s3/scratch/jamie.towner/flood_aa'  # define main working directory

# %%
# Set up the S3 path for the Zarr files
store = f"s3://wfp-seasmon/input/cds/glofas-reforecast/saf/*/*.zarr"

# Set up connection to s3 store
s3 = s3fs.S3FileSystem.current()

# Define mapper object for multiple files
remote_files = s3.glob(store)
store = [s3fs.S3Map(root=f"s3://{file}", s3=s3, check=False) for file in remote_files]

# Configure dask to avoid creating large chunks and open the dataset
with dask.config.set(**{"array.slicing.split_large_chunks": True}):
    ds = xr.open_mfdataset(store, decode_coords="all", engine="zarr")

# %%
# Load the CSV file containing station information (i.e., station name, lat, lon)
# define paths to data
metadata_directory = os.path.join(directory, country, "data/metadata")
station_info_file = "metadata_observations.csv"
station_info_path = os.path.join(metadata_directory, station_info_file)
station_info = pd.read_csv(station_info_path)

# Define the output directory
out_dir = os.path.join(directory, country, "data/forecasts/glofas_reforecasts/stations")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# %%
# Initialize tqdm progress bar
pbar = tqdm(total=len(station_info), desc="Extracting Data")

# Loop over each station in the metadata file
for index, row in station_info.iterrows():

    point_name = row['station name']
    latitude = row['lisflood_y']
    longitude = row['lisflood_x']
    
    if np.isnan(latitude) or np.isnan(longitude):
        latitude = row['latitude']
        longitude = row['longitude']
    
    # Sanitize the station name for use in a file path (e.g., remove spaces, special characters)
    station_name = "".join(c for c in point_name if c.isalnum() or c in (' ', '_')).replace(' ', '_')
    
    # Define expected output file path
    nc_file_name = os.path.join(out_dir, f"{station_name}.nc")

    # Skip if file already exists
    if os.path.exists(nc_file_name):
        print(f"Skipping {station_name} (already exists)")
        pbar.update(1)
        continue

    # Find the nearest latitude and longitude in the dataset
    lat_index = ds['latitude'].sel(latitude=latitude, method='nearest')
    lon_index = ds['longitude'].sel(longitude=longitude, method='nearest')

    # Extract river discharge data for the nearest point and for all ensemble members
    data_at_point = persist_with_progress_bar(ds['dis24'].sel(latitude=lat_index, longitude=lon_index))

    # Print the actual dimensions for diagnostics
    print("Data at point dimensions:", data_at_point.dims)
    print("Data at point shape:", data_at_point.shape)

    # Extract dates, ensemble member, and step
    dates = ds.time.values
    ensemble_members = ds['dis24'].coords['number'].values
    steps = ds.step.values

    # Create a new xarray dataset for storing data in NetCDF format
    station_ds = xr.Dataset(
        {
            "dis24": (["number", "time", "step"], data_at_point.values)
        },
        coords={
            "number": ensemble_members,
            "time": dates,
            "step": steps
        },
        attrs={
            "description": f"River discharge forecasts for {point_name}",
            "units": "m^3/s"
        }
    )

    # Save the dataset as a NetCDF file
    nc_file_name = os.path.join(out_dir, f"{station_name}.nc")
    station_ds.to_netcdf(nc_file_name)

    # Update the progress bar
    pbar.update(1)

# Close the progress bar
pbar.close()

# %%
