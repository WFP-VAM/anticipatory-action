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
# Script to extract GloFAS reanalysis data at station locations stored in an s3 bucket. Metadata file is used to identify which station points to extract (use Lisflood x and y coordinates if available).

# %%
import s3fs
import dask
import xarray as xr
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# %%
country = 'mozambique'  # define country of interest
directory = '/s3/scratch/jamie.towner/flood_aa'  # define main working directory

# %%
# Set up the S3 path for the Zarr files
store = f"s3://wfp-seasmon/input/cds/glofas-historical/saf/01/*.zarr"

# Set up connection to s3 store
s3 = s3fs.S3FileSystem.current()

# Fetch list of .zarr stores (files)
remote_files = s3.glob(store)
store = [
    s3fs.S3Map(root=f"s3://{file}", s3=s3, check=False) for file in remote_files
]

# %%
# Load the CSV file containing station information (i.e., station name, lat, lon)
# define paths to data
metadata_directory = os.path.join(directory, country, "data/metadata")
station_info_file = "metadata_observations.csv"
station_info_path = os.path.join(metadata_directory, station_info_file)
station_info = pd.read_csv(station_info_path)

# Create the output directory if it doesn't exist
out_dir = os.path.join(directory, country, "data/forecasts/glofas_reanalysis")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Initialize a dictionary to store data for each station
station_data = {}

# Initialize tqdm with the total number of iterations to track progress
total_iterations = len(remote_files) * len(station_info)
pbar = tqdm(total=total_iterations, desc="Extracting Data")

# Open multiple .zarr files with dask and xarray, setting chunk configuration
with dask.config.set(**{"array.slicing.split_large_chunks": True}):
    ds = xr.open_mfdataset(
        store,
        decode_coords="all",
        engine="zarr",
        parallel=True,  # Enable parallel processing for speed-up
        combine="by_coords"
    )

    # Loop over each station in the station_info CSV
    for index, row in station_info.iterrows():
        point_name = row['station name']
        latitude = row['lisflood_y']
        longitude = row['lisflood_x']
        if np.isnan(latitude) or np.isnan(longitude):
            latitude = row['latitude']
            longitude = row['longitude']

        # Replace 'lat' and 'lon' with 'latitude' and 'longitude'
        lat_index = ds['latitude'].sel(latitude=latitude, method='nearest').values
        lon_index = ds['longitude'].sel(longitude=longitude, method='nearest').values

        # Extract river discharge data for the nearest point
        data_at_point = ds['dis24'].sel(latitude=lat_index, longitude=lon_index).values
        dates = ds.time.values

        # Convert dates to DD/MM/YYYY format
        formatted_dates = pd.to_datetime(dates).strftime('%d/%m/%Y')

        # Create a DataFrame for the extracted data
        extracted_df = pd.DataFrame({'date': formatted_dates, 'river discharge': data_at_point})

        # Append the data to the station's DataFrame within the station_data dictionary
        if point_name not in station_data:
            station_data[point_name] = extracted_df
        else:
            # Merge with the existing data for the same station
            station_data[point_name] = pd.concat([station_data[point_name], extracted_df])
        
        pbar.update(len(remote_files))  # Update tqdm progress by number of files processed

# Close the tqdm progress bar
pbar.close()

# Save extracted data for each station to CSV files
for station, data in station_data.items():
    csv_file_name = os.path.join(out_dir, f"{station}.csv")
    data.to_csv(csv_file_name, index=False)

# %%
all_dfs = []
for station, data in station_data.items():
    name = "".join(c for c in station if c.isalnum() or c in (' ', '_')).replace(' ', '_')
    data = data.rename(columns={'river discharge':name})
    data = data.set_index('date')
    all_dfs.append(data)
pd.concat(all_dfs,axis=1)

# %%
csv_file_name = os.path.join(out_dir, f"all_stations/glofas_reanalysis_complete_series.csv")
pd.concat(all_dfs,axis=1).to_csv(csv_file_name)

# %%
csv_file_name = os.path.join(out_dir, f"all_stations/glofas_reanalysis.csv")

df_all = pd.concat(all_dfs,axis=1)
df_all.index = pd.to_datetime(df_all.index,format='%d/%m/%Y')
df_all[df_all.index>='01/01/2003'].to_csv(csv_file_name)

# %% [markdown]
# ### get correlation of observed data with glofas

# %%
df_obs = pd.read_csv('/s3/scratch/jamie.towner/flood_aa/mozambique/data/observations/gauging_stations/all_stations/observations_newstations.csv')
df_obs = df_obs.rename(columns={'Unnamed: 0':'date'})
df_obs["date"] = pd.to_datetime(df_obs["date"], format='mixed')
df_obs = df_obs.set_index('date')
df_obs

# %%
df_all[df_all.index>='01/01/2003'].corrwith(df_obs)

# %%
