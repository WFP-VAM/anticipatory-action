# preseason flood anticipatory action script to process observed/reanalysis and 
# forecast data, calculate contigency metrics and choose the 'best' trigger for 
# operational use based on quality criteria (e.g., f1 score, hit rate and false alarm rates).

# import required libraries 
from tqdm import tqdm
import xarray as xr
import pandas as pd
import os
import glob
import numpy as np

#######################################################
# section 1: define variables, paths and read in data # 
#######################################################

country = 'mozambique'  # define country of interest
directory = '/s3/scratch/jamie.towner/flood_aa'  # define main working directory
benchmark = 'glofas_reanalysis'  # choose 'observations' or 'reanalysis' as the benchmark

# define paths to data
forecast_data_directory = os.path.join(directory, country, "data/forecasts/glofas_reforecasts")
metadata_directory = os.path.join(directory, country, "data/metadata")
output_directory = os.path.join(directory, country, "outputs")

# create output directory if it does not exist
os.makedirs(output_directory, exist_ok=True)

# set observed data and metadata directory and filenames based on benchmark choice
if benchmark == 'observations':
    observed_data_directory = os.path.join(directory, country, "data/observations/gauging_stations/all_stations")
    observed_data_file = "observations.csv"
    station_info_file = "metadata_observations.csv"
elif benchmark == 'glofas_reanalysis':
    observed_data_directory = os.path.join(directory, country, "data/forecasts/glofas_reanalysis/all_stations")
    observed_data_file = "glofas_reanalysis_moz.csv"
    station_info_file = 'metadata_glofas_reanalysis.csv'
else:
    raise ValueError("invalid benchmark choice. choose 'observations' or 'glofas_reanalysis'.")

# load the observed or reanalysis data and gauging stations metadata
observed_data_path = os.path.join(observed_data_directory, observed_data_file)
station_info_path = os.path.join(metadata_directory, station_info_file)

observed_data = pd.read_csv(observed_data_path)
station_info = pd.read_csv(station_info_path)

# convert the date column in observed_data to pandas timestamps 
observed_data["date"] = pd.to_datetime(observed_data["date"], format='mixed')

# load all GloFAS forecast files
forecast_files = glob.glob(os.path.join(forecast_data_directory, '*.nc'))

# print paths to ensure they are set correctly
print(f"""
forecast directory: {forecast_data_directory}
observed data directory: {observed_data_directory}
observed data file: {observed_data_file}
metadata directory: {metadata_directory}
output directory: {output_directory}
""")

##############################################################################
# section 2: process observations and forecasts and define events/non-events #
##############################################################################

# create an empty list to store events/non-events 
events = []

# filter station_info (i.e., the metadata) to include only stations present in the forecast_files
# extract station names from forecast filenames
station_names_in_files = [os.path.basename(file).split("_")[0] for file in forecast_files]
# get unique station names
unique_station_names = list(set(station_names_in_files))
# convert station names to lowercase if required
filtered_station_info = station_info[station_info["station name"].str.lower().isin([name.lower() for name in unique_station_names])]

# loop over each forecast file 
for forecast_file in tqdm(forecast_files, desc="processing forecast files"):
    # load the netcdf file
    ds = xr.open_dataset(forecast_file, decode_timedelta=True)
    
    # extract the station name from the filename
    station_name = os.path.basename(forecast_file).split("_")[0].lower()

    # process only the station that matches the current forecast file
    station_row = filtered_station_info[filtered_station_info["station name"].str.lower() == station_name]

    if station_row.empty:
        print(f"Skipping {station_name}: no matching station info found.")
        continue  # skip if station is not found in the metadata

    station_row = station_row.iloc[0]  # convert to series since we expect only one row
    
    # extract all ensemble members 
    ensemble_data = ds['dis24']  # shape: (number, step)
    ensemble_members = ensemble_data['number'].values  # extract ensemble member IDs

    # extract the forecast issue date from the file and convert to pandas datetime
    forecast_issue_ns = ds['time'].values.item()  
    forecast_issue_date = pd.to_datetime(forecast_issue_ns, unit='ns')

    # define the lead times up to 46 days ahead (lead time = 0 is actually lead time =1 in reality)
    lead_times = list(range(0, 15))  # adjust to match desired lead times

    # extract individual station thresholds from metadata file
    thresholds = {
        "bankfull": (station_row["obs_bankfull"], station_row["glofas_bankfull"]),
        "moderate": (station_row["obs_moderate"], station_row["glofas_moderate"]),
        "severe": (station_row["obs_severe"], station_row["glofas_severe"]),
    }
        
    # process each lead time
    for lead_time in lead_times:
            # calculate the forecast end date based on lead time (adds one day due to python indexing from zero)
            forecast_end_date = forecast_issue_date + pd.DateOffset(days=lead_time + 1)
            
            # filter observed data for the matching period
            observed_period = observed_data[observed_data["date"] == forecast_end_date]
            
            # skip if no observation data is available for this period
            if observed_period.empty:
                continue

            observed_values = observed_period[station_name].values[0]

            # skip if there's no observation data (NaN value) for the specific station
            if pd.isnull(observed_values):
                continue

            # extract forecast values for all ensemble members at the current lead time
            forecast_data = ensemble_data.isel(step=lead_time).values.squeeze()  # remove extra dimensions
            
            # **debug check**: ensure the shape of forecast_data is correct
            if forecast_data.ndim != 1 or forecast_data.shape[0] != len(ensemble_members):
                raise ValueError(f"unexpected shape for forecast_data: {forecast_data.shape}. expected ({len(ensemble_members)},).")

            # loop over the thresholds
            for severity, (obs_threshold, forecast_threshold) in thresholds.items():
                # define events and non-events for each ensemble member
                observed_event = observed_values > obs_threshold
                forecast_event = forecast_data > forecast_threshold     
                
                # create events dictionaries for all ensemble members at once
                for member_idx, ensemble_member in enumerate(ensemble_members):
                    events_dict = {
                        "forecast file": os.path.basename(forecast_file),
                        "lead time": lead_time,
                        "station name": station_name,
                        "ensemble member": ensemble_member,
                        "forecasted date": forecast_end_date.date(),
                        "threshold": severity,
                        "observed event": observed_event,
                        "forecast event": bool(forecast_event[member_idx]),  
                    }
                    # append the events dictionary to the list
                    events.append(events_dict)

# create a data frame from the list of event dictionaries
events_df = pd.DataFrame(events)
print("processing complete.")

####################################################################################
# section 3: create function to construct contigency table and skill score metrics #
####################################################################################

# function to calculate verification metrics 
def calculate_metrics(df):
    hits, false_alarms, misses, correct_rejections = {}, {}, {}, {}
    hit_rate, false_alarm_rate, csi, f1_score = {}, {}, {}, {}

    # loop through all "trigger" columns
    for column in [col for col in df.columns if 'trigger' in col]:
        obs_1, obs_0 = df['observed event'] == 1, df['observed event'] == 0
        fcst_1, fcst_0 = df[column] == 1, df[column] == 0

        # calculate contingency table elements
        hits[column] = (obs_1 & fcst_1).sum()
        false_alarms[column] = (obs_0 & fcst_1).sum()
        misses[column] = (obs_1 & fcst_0).sum()
        correct_rejections[column] = (obs_0 & fcst_0).sum()

        # compute verification metrics
        total_observed_events = hits[column] + misses[column]
        total_forecasted_events = hits[column] + false_alarms[column]

        hit_rate[column] = hits[column] / total_observed_events if total_observed_events > 0 else 0
        false_alarm_rate[column] = false_alarms[column] / total_forecasted_events if total_forecasted_events > 0 else 0
        csi[column] = hits[column] / (hits[column] + false_alarms[column] + misses[column]) if (hits[column] + false_alarms[column] + misses[column]) > 0 else 0
        
        precision = hits[column] / total_forecasted_events if total_forecasted_events > 0 else 0
        recall = hit_rate[column]
        f1_score[column] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # convert metrics dictionaries into dataframes
    metrics_df = pd.concat([
        pd.DataFrame(hits, index=['hits']),
        pd.DataFrame(false_alarms, index=['false_alarms']),
        pd.DataFrame(misses, index=['misses']),
        pd.DataFrame(correct_rejections, index=['correct_rejections']),
        pd.DataFrame(hit_rate, index=['hit_rate']),
        pd.DataFrame(false_alarm_rate, index=['false_alarm_rate']),
        pd.DataFrame(csi, index=['csi']),
        pd.DataFrame(f1_score, index=['f1_score']),
    ])

    return metrics_df

############################################################
# section 4: grouping by lead-time and calculating metrics #
############################################################

# pivot the events_df to list ensemble members as columns
events_df = events_df.pivot_table(index=["forecast file", "lead time", "station name", "forecasted date","threshold", "observed event"],
                                        columns="ensemble member",)

# reset index to convert the pivoted dataframe to a flat table structure
events_df.reset_index(inplace=True)

# define the columns corresponding to the forecast ensemble members
ensemble_member_columns = [col for col in events_df.columns if col[0] == "forecast event"]

# calculate the percentage of ensemble members with events (i.e., 1's) for each row
events_df["probability"] = events_df[ensemble_member_columns].sum(axis=1) / len(ensemble_members)

# check if the columns of events_df have a multi index
if isinstance(events_df.columns, pd.MultiIndex):
    # flatten the multi index into strings
    events_df.columns = [' '.join(map(str, col)).strip() for col in events_df.columns]

# remove columns with "forecast event" in their names
events_df = events_df.loc[:, ~events_df.columns.str.contains('forecast event')]

# identify forecasts where lead_time = 0 and observed event = true (i.e., where flooding is already occuring)
forecasts_to_remove = events_df[(events_df['lead time'] == 0) & (events_df['observed event'])][['forecast file', 'station name', 'threshold']].drop_duplicates()

# filter out all rows with the same forecast file, station name, and threshold for any lead time
filtered_events_df = events_df[~events_df[['forecast file', 'station name', 'threshold']].apply(tuple, axis=1).isin(forecasts_to_remove.apply(tuple, axis=1))]

# get the number of forecasts removed including lead time
removed_forecasts_count = len(events_df) - len(filtered_events_df)
print(f"number of forecasts removed: {removed_forecasts_count}")

# get the number of forecasts removed excluding lead time
removed_forecasts_unique_count = forecasts_to_remove.drop_duplicates(subset=['forecast file', 'station name', 'threshold']).shape[0]

print(f"number of unique forecasts removed: {removed_forecasts_unique_count}")

# assign back to the original variable
events_df = filtered_events_df

# filter the dataframe to include specific lead times
events_df = events_df[(events_df['lead time'] >=1) & (events_df['lead time'] <=4)]

# group by station name, lead time category, and threshold
grouped = events_df.groupby(['forecast file','station name','threshold'], observed=False)

# create a dictionary to store each group's dataframe
grouped_dfs = {name: group for name, group in grouped}

# dictionary to store processed data
new_grouped_dfs = {}

# calculate events and non-events in the lead time period (i.e., flood event if any observed value in period is a 1, take the mean probability for the forecast data)
for name, df in grouped_dfs.items():
    first_row = df.iloc[0]

    new_grouped_dfs[name] = pd.DataFrame({
        'forecast file': [first_row['forecast file']],
        'station name': [first_row['station name']],
        'threshold': [first_row['threshold']],
        'observed event': [(df['observed event'] == 1).any()],
        'probability': [df['probability'].mean()]
    })

# combine all resulting dataframe into one 
final_df = pd.concat(new_grouped_dfs.values(), ignore_index=True)

# add trigger thresholds ranging from 1-100% 
trigger_columns = {}

for trigger in np.arange(0.01, 1.01, 0.01): 
    event_occurrence = (final_df['probability'] >= trigger).astype(int)
    trigger_columns[f'trigger{trigger:.2f}'] = event_occurrence

# concatanate the new trigger columns to the dataframe
final_df = pd.concat([final_df, pd.DataFrame(trigger_columns, index=final_df.index)], axis=1)

# group by station name, lead time category, and threshold
grouped = final_df.groupby(['station name','threshold'], observed=False)

# create a dictionary to store each group's dataframe
grouped_dfs = {name: group for name, group in grouped}

# iterate through each dataframe in grouped_dfs, apply calculate_metrics, and store the results back into grouped_dfs
for key, df in grouped_dfs.items():
    grouped_dfs[key] = calculate_metrics(df)

################################
# section 5: trigger selection #
################################

# create an empty dictionary to store the best triggers
best_triggers = {}

# iterate through each dataframe in grouped_dfs
for key, df in grouped_dfs.items():
    # find the column names corresponding to trigger thresholds (i.e., the percentages)
    threshold_columns = [col for col in df.columns if col.startswith('trigger')]

    # filter triggers based on the f1 score
    filtered_columns = [
        col for col in threshold_columns
        if df.loc['f1_score', col] >= 0.45
    ]
    
    # if there are any columns left after filtering, identify the maximum f1 score for each threshold (i.e., bankfull, moderate, severe)
    if filtered_columns:
        max_f1 = df.loc['f1_score', filtered_columns].max()
        
        # find all thresholds with the maximum f1 score 
        best_f1_thresholds = df.loc['f1_score', filtered_columns][df.loc['f1_score', filtered_columns] == max_f1].index.tolist()
        
        # if there are multiple thresholds with the same f1 score, proceed to resolve ties
        if len(best_f1_thresholds) > 1:
            # resolve ties by choosing the highest hit rate
            hit_rates = df.loc['hit_rate', best_f1_thresholds]
            max_hit_rate = hit_rates.max()
            best_f1_thresholds = hit_rates[hit_rates == max_hit_rate].index.tolist()

            # if there are still ties, resolve by choosing the lowest false alarm rate
            if len(best_f1_thresholds) > 1:
                false_alarm_rates = df.loc['false_alarm_rate', best_f1_thresholds]
                min_false_alarm_rate = false_alarm_rates.min()
                best_f1_thresholds = false_alarm_rates[false_alarm_rates == min_false_alarm_rate].index.tolist()

                # if there are still ties, choose the lowest trigger (threshold)
                if len(best_f1_thresholds) > 1:
                    best_threshold = min(best_f1_thresholds, key=lambda x: float(x.split('trigger')[1]))  # Sorting by numeric threshold
                else:
                    best_threshold = best_f1_thresholds[0]
            else:
                best_threshold = best_f1_thresholds[0]
        else:
            best_threshold = best_f1_thresholds[0]
        
        # store the best threshold information
        best_triggers[key] = {
            'best_threshold': best_threshold,
            'f1_score': df.loc['f1_score', best_threshold],
            'hit_rate': df.loc['hit_rate', best_threshold],
            'false_alarm_rate': df.loc['false_alarm_rate', best_threshold]
        }

# convert the best_triggers dictionary to a dataframe
best_triggers_df = pd.DataFrame(best_triggers).T

# print the best triggers
print(best_triggers_df)

# define the full output file path
output_file = os.path.join(output_directory, "best_triggers.csv")

# save the results as a CSV file
best_triggers_df.to_csv(output_file, index=True)
