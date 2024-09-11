# process observed and forecast data and run verification analysis (i.e., contingency and skill score metrics)
# for all lead times out to 46 days. 
# calculate triggers for each gauging station. 

from tqdm import tqdm
import xarray as xr
import pandas as pd
import os
import glob
import numpy as np
import time

#######################################################
# section 1: define variables, paths and read in data # 
#######################################################

# timer start
start_time = time.time()

country = 'mozambique'  # define country of interest
directory = '/Volumes/TOSHIBA EXT/'  # define directory

# define paths to data
forecast_data_directory = os.path.join(directory, "flood aa/forecast data/example")                                      
observed_data_directory = os.path.join(directory, "flood aa", country, "observed data/gauging stations")
metadata_directory = os.path.join(directory, "flood aa", country, "observed data/metadata")
output_directory = os.path.join(directory, country, "outputs")

# create directories if they don't exist
os.makedirs(forecast_data_directory, exist_ok=True)
os.makedirs(observed_data_directory, exist_ok=True)
os.makedirs(metadata_directory, exist_ok=True)
os.makedirs(output_directory, exist_ok=True)

# define filenames
observed_data_file = "all_stations_observed.csv"
station_info_file = "metadata_moz.csv"

# load the observed data and gauging stations metadata
observed_data_path = os.path.join(observed_data_directory, observed_data_file)
station_info_path = os.path.join(metadata_directory, station_info_file)
observed_data = pd.read_csv(observed_data_path)
station_info = pd.read_csv(station_info_path)

# convert the date column in observed_data to pandas timestamps 
observed_data["date"] = pd.to_datetime(observed_data["date"])

# load all glofas forecast files
forecast_files = glob.glob(os.path.join(forecast_data_directory, '*.nc'))

# print paths to ensure they are set correctly
print(f"""
forecast directory: {forecast_data_directory}
observed data directory: {observed_data_directory}
metadata directory: {metadata_directory}
output directory: {output_directory}
""")

##############################################################################
# section 2: process observations and forecasts and define events/non events #
##############################################################################

# create an empty list to store events/non-events 
events = []

# loop over each forecast file 
for forecast_file in tqdm(forecast_files, desc="processing forecast files"):
    # load the NetCDF file
    ds = xr.open_dataset(forecast_file)
    
    # identify ensemble member variables by variable name (e.g., dis24_0 to dis24_10)
    ensemble_members = [var for var in ds.variables if var.startswith("dis24_")]

    # loop over lead times 
    lead_times = list(range(0,46)) # remember 0 is day 1; out to 46 days (i.e., 0-46)
    for lead_time in lead_times:
            # calculate the forecast end date based on lead time
            first_time_step = ds['time'].values[0]
            forecast_start_date = pd.to_datetime(first_time_step)
            forecast_end_date = forecast_start_date + pd.DateOffset(days=lead_time)
            
            # loop over station metadata to extract information for each station
            for index, station_row in station_info.iterrows():
                station_name = station_row["station name"]
                station_lat = station_row["lisflood_y"]
                station_lon = station_row["lisflood_x"]
                
                # extract individual station thresholds from metadata file
                thresholds = {
                "bankfull": (station_row["obs_bankfull"], station_row["glofas_bankfull"]),
                "moderate": (station_row["obs_moderate"], station_row["glofas_moderate"]),
                "severe": (station_row["obs_severe"], station_row["glofas_severe"]),
                }
                
                # filter observed data for the matching period
                observed_period = observed_data[observed_data["date"] == forecast_end_date].iloc[-1]
                # filter observed data for the matching period and ±2 days window
                observed_window = observed_data[(observed_data["date"] >= forecast_end_date - pd.DateOffset(days=2)) & 
                                                (observed_data["date"] <= forecast_end_date + pd.DateOffset(days=2))]
                
                # extract observed values
                observed_values_window = observed_window[station_name]
                observed_values = observed_period[station_name]
                
                # check if there's no observation data (NaN value) for the specific station
                if pd.isnull(observed_values):
                    continue  # skip if there are NaN values in the observed data
                   
                # check if there's no observation data (NaN value) for the specific station
                if pd.isnull(observed_values_window).all():
                    continue  # skip if there are NaN values in the observed data
                        
                # extract forecast values for each ensemble member
                for variable_name in ensemble_members:
                    forecast_data = ds[variable_name].sel(lat=station_lat, lon=station_lon, time=forecast_end_date, method='nearest')

                    # loop over the thresholds
                    for severity, (obs_threshold, sim_threshold) in thresholds.items():
                        # define events and non-events
                        observed_event = observed_values > obs_threshold
                        forecast_event = forecast_data.values > sim_threshold
                        
                        # define tolerant events and non-events (within ±2 days window)
                        tolerant_observed_event = (observed_values_window > obs_threshold).any()
                        
                        # create a dictionary to store results
                        events_dict = {
                            "forecast file": os.path.basename(forecast_file),
                            "lead time": lead_time,
                            "station name": station_name,
                            "ensemble member": variable_name,
                            "forecasted date": forecast_end_date.date(),
                            "threshold": severity,
                            "observed event": observed_event,
                            "forecast event": forecast_event,
                            "tolerant observed event": tolerant_observed_event
                        }
    
                        # append the events dictionary to the list
                        events.append(events_dict)

# create a data frame from the list of event dictionaries
events_df = pd.DataFrame(events)

#################################################################
# section 3: construct contigency table and skill score metrics #
#################################################################

# pivot the events_df to list ensemble members as columns
pivot_events_df = events_df.pivot_table(index=["forecast file", "lead time", "station name", "forecasted date","threshold", "observed event", "tolerant observed event"],
                                        columns="ensemble member",)

# reset index to convert the pivoted DataFrame to a flat table structure
pivot_events_df.reset_index(inplace=True)

# define the columns corresponding to the forecast ensemble members (5 to 16)
ensemble_member_columns = pivot_events_df.columns[7:18]

# calculate the percentage of ensemble members with events (i.e., 1's) for each row
pivot_events_df["probability of detection"] = pivot_events_df[ensemble_member_columns].sum(axis=1) / len(ensemble_members)

# Generate trigger thresholds from 0.01 to 0.99 with a step size of 0.01
thresholds = np.arange(0.01, 1, 0.01)

# initialize an empty dataframe to store the event occurrence for each threshold
metrics_df = pd.DataFrame(index=pivot_events_df.index)

# loop over each threshold
for threshold in thresholds:
    # determine if the forecast event occurs based on the threshold
    event_occurrence = (pivot_events_df['probability of detection'] >= threshold).astype(int)
    
    # add the event occurrence as a column in the event_df DataFrame
    metrics_df[f'threshold_{threshold:.2f}'] = event_occurrence

# concatenate the forecast file and observed event columns from result_df to event_df
metrics_df = pd.concat([pivot_events_df[['forecast file','station name', 'lead time','forecasted date','threshold','observed event','tolerant observed event','probability of detection']], metrics_df], axis=1)

# flatten the multi-index columns
metrics_df.columns = [''.join(col).strip() for col in metrics_df.columns.values]

# define the lead time ranges (optional, change as desired)
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 46, float('inf')]
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-46', '0-46']

# create a new column called lead time category' that categorizes lead times into these ranges
metrics_df['lead time category'] = pd.cut(metrics_df['lead time'], bins=bins, labels=labels, right=False)

# group by station name, lead time category, and threshold
grouped = metrics_df.groupby(['station name', 'lead time category', 'threshold'], observed=False)

# create a dictionary to store each group's dataframe
grouped_dfs = {name: group for name, group in grouped}

# function to calculate verification metrics metrics for each grouped dataframe 
def calculate_metrics(df):
    hits = {}
    false_alarms = {}
    misses = {}
    correct_rejections = {}
    tolerant_hits = {}
    tolerant_false_alarms = {}
    tolerant_misses = {}
    tolerant_correct_rejections = {}
    hit_rate = {}
    false_alarm_rate = {}
    tolerant_hit_rate = {}
    tolerant_false_alarm_rate = {}
    csi = {}
    pss = {}
    f1_scores = {}

    # calculate contigency table metrics for each trigger threshold column
    for column in df.columns[8:107]:  # check columns from index 8 onwards are thresholds
        hits[column] = ((df['observed event'] == 1) & (df[column] == 1)).sum()
        false_alarms[column] = ((df['observed event'] == 0) & (df[column] == 1)).sum()
        misses[column] = ((df['observed event'] == 1) & (df[column] == 0)).sum()
        correct_rejections[column] = ((df['observed event'] == 0) & (df[column] == 0)).sum()
       
        tolerant_hits[column] = ((df['tolerant observed event'] == 1) & (df[column] == 1)).sum()
        tolerant_false_alarms[column] = ((df['tolerant observed event'] == 0) & (df[column] == 1)).sum()
        tolerant_misses[column] = ((df['tolerant observed event'] == 1) & (df[column] == 1)).sum()
        tolerant_correct_rejections[column] = ((df['tolerant observed event'] == 0) & (df[column] == 0)).sum()

        # calculate verification metrics
        total_observed_events = hits[column] + misses[column]
        total_forecasted_events = hits[column] + false_alarms[column]
        tolerant_total_forecasted_events = tolerant_hits[column] + tolerant_false_alarms[column]
        hit_rate[column] = hits[column] / total_observed_events if total_observed_events > 0 else 0
        false_alarm_rate[column] = false_alarms[column] / total_forecasted_events if total_forecasted_events > 0 else 0
        tolerant_hit_rate[column] = tolerant_hits[column] / total_observed_events if total_observed_events > 0 else 0
        tolerant_false_alarm_rate[column] = tolerant_false_alarms[column] / tolerant_total_forecasted_events if total_forecasted_events > 0 else 0
        csi[column] = hits[column] / (hits[column] + false_alarms[column] + misses[column]) if (hits[column] + false_alarms[column] + misses[column]) > 0 else 0
        pss[column] = (hits[column] * correct_rejections[column] - false_alarms[column] * misses[column]) / \
                      ((hits[column] + misses[column]) * (false_alarms[column] + correct_rejections[column])) if (hits[column] + misses[column]) * (false_alarms[column] + correct_rejections[column]) > 0 else 0
        precision = hits[column] / total_forecasted_events if total_forecasted_events > 0 else 0
        recall = hit_rate[column]
        f1_scores[column] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # convert the metrics dictionaries to dataframes
    hits_df = pd.DataFrame(hits, index=['hits'])
    false_alarms_df = pd.DataFrame(false_alarms, index=['false alarms'])
    misses_df = pd.DataFrame(misses, index=['misses'])
    correct_rejections_df = pd.DataFrame(correct_rejections, index=['correct rejections'])
    hit_rate_df = pd.DataFrame(hit_rate, index=['hit rate'])
    false_alarm_rate_df = pd.DataFrame(false_alarm_rate, index=['false alarm rate'])
    
    tolerant_hits_df = pd.DataFrame(tolerant_hits, index=['tolerant hits'])
    tolerant_false_alarms_df = pd.DataFrame(tolerant_false_alarms, index=['tolerant false alarms'])
    tolerant_misses_df = pd.DataFrame(tolerant_misses, index=['tolerant misses'])
    tolerant_correct_rejections_df = pd.DataFrame(tolerant_correct_rejections, index=['tolerant correct rejections'])
    tolerant_hit_rate_df = pd.DataFrame(tolerant_hit_rate, index=['tolerant hit rate'])
    tolerant_false_alarm_rate_df = pd.DataFrame(tolerant_false_alarm_rate, index=['tolerant false alarm rate'])
    
    csi_df = pd.DataFrame(csi, index=['csi'])
    pss_df = pd.DataFrame(pss, index=['pss'])
    f1_scores_df = pd.DataFrame(f1_scores, index=['f1 score'])
 
    # concatenate the metrics dataframes
    metrics_df = pd.concat([
        hits_df, false_alarms_df,
        misses_df, correct_rejections_df, hit_rate_df, false_alarm_rate_df, tolerant_hits_df, tolerant_false_alarms_df,
        tolerant_misses_df, tolerant_correct_rejections_df, tolerant_hit_rate_df, tolerant_false_alarm_rate_df, csi_df, pss_df, f1_scores_df,
    ])

    return metrics_df

# iterate through each dataframe in grouped_dfs, apply calculate_metrics, and store the results back into grouped_dfs
for key, df in grouped_dfs.items():
    grouped_dfs[key] = calculate_metrics(df)
    
    # timer end 
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time for calculating metrics: {elapsed_time:.2f} seconds")