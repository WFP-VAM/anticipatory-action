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
# Script to calculate return periods of the observed water level data. Can either use the complete time series or the 2003-2023 period (i.e., GloFAS analysis period). 

# %%
# import relevant packages
import pandas as pd
import numpy as np
from scipy.stats import genextreme
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
observed_data_directory = directory / "data/observations/gauging_stations/all_stations"
observed_data_file = "observations_complete_series.csv" # use observations.csv for 2003-2023 period

# load data
observed_data_path = directory / observed_data_directory /observed_data_file
observed_data = pd.read_csv(observed_data_path)

# %%
# convert date columns to datetime
observed_data["date"] = pd.to_datetime(observed_data["date"], format='mixed')

# %%
# check data
observed_data.head()


# %%
# function to calculate return periods for a given station's data
def calculate_return_periods(station_data, years=[2, 5, 10, 20]):
    # drop NA values
    station_data = station_data.dropna().copy()

    # extract the year from the date column 
    station_data.loc[:, 'year'] = station_data['date'].dt.year
    
    # group by year and get the maximum value for each year 
    annual_max = station_data.groupby('year')[station_data.columns[1]].max()
    
    # fit the data to a GEV distribution (Generalized Extreme Value distribution)
    #params = genextreme.fit(annual_max)

    # calculate the return period for each year 
    return_periods = {}
    for return_year in years:
        # the formula for return period is: 1 / (1 - F(x))
        # F(x) is the CDF of the fitted distribution at the threshold (max value)
        threshold = np.percentile(annual_max, 100 * (1 - 1/return_year))
        #threshold = genextreme.ppf(1 - 1/return_year, *params)
        return_periods[return_year] = threshold

    return return_periods

# initialize a dictionary to store return periods for each station
return_periods_dict = {}

# iterate over each station in the observed_data 
for station in observed_data.columns:
    if station == 'date':
        continue  # Skip 'date' column

    # get the data for this station
    station_data = observed_data[['date', station]]

    # skip if all values are NaN
    if station_data[station].dropna().empty:
        continue

    # calculate return periods for the station
    return_periods = calculate_return_periods(station_data)
    
    # store the return periods in the dictionary
    return_periods_dict[station] = return_periods

# convert the dictionary to a dataframe
return_periods_df = pd.DataFrame.from_dict(return_periods_dict, orient='index')

# %%
# check the output
return_periods_df

# %%
# save output as a csv 
return_periods_df.to_csv(os.path.join(output_directory, "observed_return_periods_complete_series.csv"), index=True)

# %%
