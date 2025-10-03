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
# Script to calculate the equivalent GloFAS thresholds compared to the observed thresholds. Uses GloFAS reanalysis data from 1979. Uses quantile mapping approach to map values between the observed and reanalysis dataset. 

# %%
# import relevant packages
import numpy as np
import pandas as pd
from scipy import stats
import os
from pathlib import Path

# %%
# define country and directory
country = 'mozambique'
#directory = Path(f'/s3/scratch/jamie.towner/flood_aa/{country}')
directory = Path(r"C:\Users\15133\Documents\WFP\flood_hazard\flood_aa\MOZ_training")  # define main working directory


output_directory = directory / "outputs/thresholds"
Path(output_directory).mkdir(parents=True, exist_ok=True)  # create directory if it does not already exist 

# %%
# define paths to data
metadata_directory = directory / "data/metadata"
observed_data_directory = directory / "data/observations/gauging_stations/all_stations"
reanalysis_data_directory = directory / "data/forecasts/glofas_reanalysis/all_stations"

observed_data_file = "observations_complete_series.csv"
reanalysis_data_file = "glofas_reanalysis_complete_series.csv"
station_info_file = "metadata_observations.csv"

# load data
observed_data_path = observed_data_directory / observed_data_file
reanalysis_data_path = reanalysis_data_directory / reanalysis_data_file
station_info_path = metadata_directory / station_info_file

observed_data = pd.read_csv(observed_data_path)
reanalysis_data = pd.read_csv(reanalysis_data_path)
station_info = pd.read_csv(station_info_path)

# %%
# select only chokwe station for training
station_info = station_info[station_info['station name'] == 'Limpopo_em_Chokwe']

# %%
# convert date columns to datetime
observed_data["date"] = pd.to_datetime(observed_data["date"], format='mixed')
reanalysis_data["date"] = pd.to_datetime(reanalysis_data["date"], format='mixed')
station_info['obs_bankfull'] = pd.to_numeric(station_info['obs_bankfull'], errors='coerce')
station_info['obs_moderate'] = pd.to_numeric(station_info['obs_moderate'], errors='coerce')
station_info['obs_severe'] = pd.to_numeric(station_info['obs_severe'], errors='coerce')

# %%
# Remove leading/trailing whitespace from metadata station names
station_info['station name'] = ["".join(c for c in name if c.isalnum() or c in (' ', '_')).replace(' ', '_') for name in station_info['station name']]

# Remove whitespace from observed and reanalysis data columns
observed_data.columns = observed_data.columns.str.strip()
reanalysis_data.columns = reanalysis_data.columns.str.strip()

# %%
# initialize list to store results
results = []

# loop over each station and threshold in metadata
for index, row in station_info.iterrows():
    station = row['station name']
    
    # skip station if any threshold is missing (NaN)
    if pd.isna(row['obs_bankfull']) or pd.isna(row['obs_moderate']) or pd.isna(row['obs_severe']):
        continue
    
    # get observed and reanalysis data for the station
    data_observed = observed_data[station].dropna().values
    data_reanalysis = reanalysis_data[station].dropna().values

    # standardize both datasets (z-score normalization)
    obs_mean, obs_std = np.mean(data_observed), np.std(data_observed)
    reanalysis_mean, reanalysis_std = np.mean(data_reanalysis), np.std(data_reanalysis)

    z_observed = (data_observed - obs_mean) / obs_std
    z_reanalysis = (data_reanalysis - reanalysis_mean) / reanalysis_std

    # define thresholds to loop over
    thresholds = {
        'obs_bankfull': row['obs_bankfull'],
        'obs_moderate': row['obs_moderate'],
        'obs_severe': row['obs_severe']
    }

    # loop over each threshold
    for threshold_name, threshold_value in thresholds.items():
        # convert threshold to z-score in observed data space
        z_threshold = (threshold_value - obs_mean) / obs_std

        # get percentile rank of threshold in observed data
        percentile_rank_observed = stats.percentileofscore(z_observed, z_threshold)

        # ensure percentiles are within valid range
        percentile_rank_observed = max(0, min(percentile_rank_observed, 100))

        # interpolate the corresponding value in reanalysis data
        percentiles = np.linspace(0, 100, len(z_reanalysis))
        z_mapped = np.interp(percentile_rank_observed, percentiles, np.sort(z_reanalysis))

        # convert back to the original reanalysis scale
        value_reanalysis = (z_mapped * reanalysis_std) + reanalysis_mean

        # store results
        results.append({
            'station': station,
            'threshold_name': threshold_name,
            'threshold_value': threshold_value,
            'percentile_rank_observed': percentile_rank_observed,
            'value_reanalysis': value_reanalysis
        })

# convert results to a dataframe and print
results_df = pd.DataFrame(results)
results_df

# %%
# save output as a csv 
results_df.to_csv(os.path.join(output_directory, "glofas_return_periods_complete_series.csv"), index=True)

# %%
results_df.pivot_table(index='station',columns='threshold_name',values='value_reanalysis')

# %%
