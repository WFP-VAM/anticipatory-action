# %% [raw]
# Preseason flood anticipatory action script to process observed/reanalysis and 
# forecast data, calculate contigency metrics and choose the 'best' trigger for 
# operational use based on quality criteria (e.g., f1 score, hit rate and false alarm rates).
# This script is the alternative, which uses the Google Flood reforecasts instead of GloFAS. The Google reforecasts are only available from 2016-2023, and they are not provided in ensembles. The script has been adjusted to account for this.

# %%
# import relevant packages 
from tqdm import tqdm
import xarray as xr
import pandas as pd
import os
import glob
import numpy as np

# %% [markdown]
# Section 1: Define variables, paths and read in data 
#
# In this section we begin by defining our country of interest, our main working directory and whether we are using observed or reanalysis data. We then define the paths to load the data and perform visual checks to ensure the data is what we are expecting. 

# %%
country = 'zimbabwe'  # define country of interest
directory = '/s3/scratch/jamie.towner/flood_aa'  # define main working directory
benchmark = 'observations'  # choose 'observations' or 'glofas_reanalysis' as the benchmark

# define paths to data
forecast_data_directory = os.path.join(directory, country, "data/forecasts/glofas_reforecasts")
metadata_directory = os.path.join(directory, country, "data/metadata")
output_directory = os.path.join(directory, country, "outputs/triggers")

# create output directory if it does not exist
os.makedirs(output_directory, exist_ok=True)

# set observed data and metadata directory and filenames based on benchmark choice
if benchmark == 'observations':
    observed_data_directory = os.path.join(directory, country, "data/observations/gauging_stations/all_stations")
    observed_data_file = "observations.csv"
    station_info_file = "metadata_observations.csv"
elif benchmark == 'glofas_reanalysis':
    observed_data_directory = os.path.join(directory, country, "data/forecasts/glofas_reanalysis/all_stations")
    observed_data_file = "glofas_reanalysis.csv"
    station_info_file = 'metadata_glofas_reanalysis.csv'
else:
    raise ValueError("invalid benchmark choice. choose 'observations' or 'glofas_reanalysis'.")

# load the observed or reanalysis data and gauging stations metadata
observed_data_path = os.path.join(observed_data_directory, observed_data_file)
station_info_path = os.path.join(metadata_directory, station_info_file)

observed_data = pd.read_csv(observed_data_path)
station_info = pd.read_csv(station_info_path)
# format station name
#station_info['station name'] = ["".join(c for c in name if c.isalnum() or c in (' ', '_')).replace(' ', '_') for name in station_info['station name']]

# %%
# check the observed data 
observed_data.head()

# %%
observed_data = observed_data[(observed_data['date']>='2016-01-01')&(observed_data['date']<='2023-07-01')]

# %%
# check the metadata
station_info.head()

# %%
# convert the date column in observed_data to pandas timestamps 
observed_data["date"] = pd.to_datetime(observed_data["date"], format='mixed')

# %%
# read in Google reforecasts and return periods to use as thresholds
forecasts = pd.read_csv(os.path.join(forecast_data_directory,'stations/ZWE_google_reforecast.csv'))
rps = pd.read_csv(os.path.join(directory, country,'data/forecasts/glofas_reanalysis/all_stations/ZWE_google_return_period.csv'))

# %%
# match google IDs to station names
goog_ids = {
    'hybas_1121548490': 'beitbridge',
    'hybas_1121530890': 'manyuchi',
    'hybas_1122213940': 'mazowe',
    'hybas_1121502180': 'condo',
    'hybas_1121515070': 'mutirikwi',
    'hybas_1121532060': 'tokwe',
    'hybas_1121532310': 'runde',
    'hybas_1121512000': 'chisurgwe',
    'hybas_1122227230': 'pungwe',
    'hybas_1122240700': 'ypres',
    'hybas_1122226070': 'katiyo'
}

forecasts['station name'] = forecasts['gauge_id'].map(goog_ids)
rps['station name'] = rps['gauge_id'].map(goog_ids)
rps.head()

# %%
# format forecast df
forecasts = forecasts.rename(columns={'lead_time':'lead time'})
forecasts['lead time'] = forecasts['lead time'].str.split(' ').str[0]

# %%
# print data paths to ensure they are set correctly
print(f"""
forecast directory: {forecast_data_directory}
observed data directory: {observed_data_directory}
observed data file: {observed_data_file}
metadata directory: {metadata_directory}
output directory: {output_directory}
""")

# %% [markdown]
# Section 2: Process observations and forecasts and define events/non-events 
#
# In this section we begin by matching the station names in the forecast files from those in the metadata and then process one forecast file at a time. We should have 1052 forecast files per station analysed. To process a station we need the name to be in both the station_info file (i.e., the metadata) and the naming convention of the forecast files. A message will highlight if this is not the case and the code will skip that particular gauging station. We then extract the variable of the forecast data which is river discharge (m3/s) and each ensemble member (we have 11 for GloFAS reforecasts and there will be 51 operationally). After we define the forecast issue date from the forecast netcdf and calculate the forecast end date based on lead time. Remember that due to the indexing of Python from 0 we add 1 to each end date. 
#
# Looking at the metadata you can see that we have three thresholds for both observations and forecasts. These are bankfull, moderate and severe. The bankfull represents the point at which a gauging station begins to flood and is the readiness phase of the system (i.e., no anticipatory action will take place). The moderate and severe thresholds are based on the 5 and 10 year return periods of the observed data. We then use quantile mapping to map these observed thresholds to the GloFAS reanalysis product over the same time period (2003-2023). This in effect performs a bias correction on the forecasts. 
#
# Finally, for each threshold we simply identify if each ensemble members of a forecast for each specific lead time exceeds the forecast thresholds and we do the same for the observations. If there is an exceedance we assign a 1 or TRUE and if not we assign a 0 or FALSE. We end this section by creating a dataframe which displays these results called events_df. 

# %%
# filter station_info (i.e., the metadata) to include only stations present in the forecast_files
# extract station names from forecast filenames
station_names_in_files = forecasts['station name'].unique()
# get unique station names
unique_station_names = list(set(station_names_in_files))
# convert station names to lowercase if required
filtered_station_info = station_info[station_info["station name"].str.lower().isin([name.lower() for name in unique_station_names])]


# create an empty list to store events/non-events 
events = []

# loop over each forecast file 
for forecast_file in forecasts['issue_time'].unique():
    for station_name in unique_station_names:

        # filter forecast for the issue date and station
        ds = forecasts[(forecasts['issue_time']==forecast_file)&(forecasts['station name']==station_name)]

        # process only the station that matches the current forecast file
        station_row = station_info[station_info["station name"] == station_name]
        rp_row = rps[rps["station name"] == station_name]

        if station_row.empty:
            print(f"Skipping {station_name}: no matching station info found.")
            continue  # skip if station is not found in the metadata

        station_row = station_row.iloc[0]  # convert to series since we expect only one row
        rp_row = rp_row.iloc[0]

        # extract the forecast issue date from the file and convert to pandas datetime
        forecast_issue_ns = forecast_file
        forecast_issue_date = pd.to_datetime(forecast_issue_ns)

        # define the lead times up to 46 days ahead (lead time = 0 is actually lead time =1 in reality)
        lead_times = list(range(0, 7))  # adjust to match desired lead times

        # extract individual station thresholds from metadata file
        thresholds = {
            "bankfull": (station_row["obs_bankfull"], rp_row["return_period_2"]),
            "moderate": (station_row["obs_moderate"], rp_row["return_period_5"]),
            "severe": (station_row["obs_severe"], rp_row["return_period_10"]),
        }

        # process each lead time
        for lead_time in lead_times:
                # calculate the forecast end date based on lead time (adds one day due to python indexing from zero)
                forecast_end_date = forecast_issue_date + pd.DateOffset(days=lead_time + 1)

                # filter observed data for the matching period
                observed_period = observed_data[observed_data["date"] == str(forecast_end_date)[:10]]

                # skip if no observation data is available for this period
                if observed_period.empty:
                    continue

                observed_values = observed_period[station_name].values[0]

                # skip if there's no observation data (NaN value) for the specific station
                if pd.isnull(observed_values):
                    continue

                # extract forecast values for all ensemble members at the current lead time
                forecast_data = ds[ds['lead time'].astype(int)==lead_time]['streamflow']  # remove extra dimensions

                # **debug check**: ensure the shape of forecast_data is correct
                if forecast_data.ndim != 1:
                    raise ValueError(f"unexpected shape for forecast_data: {forecast_data.shape}. expected ({len(ensemble_members)},).")

                # loop over the thresholds
                for severity, (obs_threshold, forecast_threshold) in thresholds.items():
                    # define events and non-events for each ensemble member
                    observed_event = observed_values > obs_threshold
                    forecast_event = forecast_data > forecast_threshold     

                    # create events dictionary
                    events_dict = {
                        "forecast file": os.path.basename(forecast_file),
                        "lead time": lead_time,
                        "station name": station_name,
                        "forecasted date": forecast_end_date.date(),
                        "threshold": severity,
                        "observed event": observed_event,
                        "forecast event": forecast_event.item(),  
                    }
                    # append the events dictionary to the list
                    events.append(events_dict)
                        
# create a data frame from the list of event dictionaries
events_df = pd.DataFrame(events)
print("processing complete.")

# %%
events_df.to_csv('events_df_google.csv')


# %% [markdown]
# Section 3: Create a function to construct contigency table and skill score metrics
#
# In this section we create a function called calculate_metrics which counts the number of hits, misses, false alarms and correct rejections before calculating metrics such as the hit and false alarm rate, critical success index (CSI) and f1 score. These metrics will help us determine how good the forecast is at detecting floods events when compared to the observed or reanalysis datasets. 

# %%
# function to calculate verification metrics 
def calculate_metrics(df):
    hits, false_alarms, misses, correct_rejections = {}, {}, {}, {}
    hit_rate, false_alarm_rate, csi, f1_score = {}, {}, {}, {}

    # loop through all "trigger" columns
    obs_1, obs_0 = df['observed event'] == 1, df['observed event'] == 0
    fcst_1, fcst_0 = df['forecast event'] == 1, df['forecast event'] == 0

    # calculate contingency table elements
    hits = (obs_1 & fcst_1).sum()
    false_alarms = (obs_0 & fcst_1).sum()
    misses = (obs_1 & fcst_0).sum()
    correct_rejections = (obs_0 & fcst_0).sum()

    # compute verification metrics
    total_observed_events = hits + misses
    total_forecasted_events = hits + false_alarms

    hit_rate = hits / total_observed_events if total_observed_events > 0 else 0
    false_alarm_rate = false_alarms / total_forecasted_events if total_forecasted_events > 0 else 0
    csi = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) > 0 else 0

    precision = hits / total_forecasted_events if total_forecasted_events > 0 else 0
    recall = hit_rate
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # convert metrics dictionaries into dataframes
    metrics_df = pd.concat([
        pd.DataFrame([hits], index=['hits']),
        pd.DataFrame([false_alarms], index=['false_alarms']),
        pd.DataFrame([misses], index=['misses']),
        pd.DataFrame([correct_rejections], index=['correct_rejections']),
        pd.DataFrame([hit_rate], index=['hit_rate']),
        pd.DataFrame([false_alarm_rate], index=['false_alarm_rate']),
        pd.DataFrame([csi], index=['csi']),
        pd.DataFrame([f1_score], index=['f1_score']),
    ])

    return metrics_df

# %% [markdown]
# Section 4: Grouping by lead-time and calculating metrics
#
# In this section we begin by pivoting the events_df created in Section 2 so that the ensemble members are displayed as columns before calculating the probability of each forecast by summing the number of 1's and divding by the total number of ensemble members (11 in our case). Once we have the probability of each forecast we remove any forecasts where there is already a flood observed (i.e., observed event = 1 or TRUE) on the first lead time (i.e., lead time = 0). We do this as we do not want to include forecasts in the analysis where there is already a flood occuring or where a flood is imminent. It's important to note here that if a forecast is removed, we remove all lead times associated with that forecast. We only remove the forecast for the threshold where there is an observed event (i.e., if the bankfull threshold is exceeded on lead_time = 0, but not for the moderate threshold then we keep the forecasts for the moderate and severe thresholds only. We do this in order to see if the flood event gets progressively worse. 
#
# We then filter our events_df based on the lead-times to a particular grouping which balances the need for 2-3 days lead time for anticipatory action along with the limited period in which we have forecast skill (e.g., 2-5 days, 3-5 days, 3-6 days etc.). We then move onto grouping the large events_df by station, threshold and per forecast file into more managable small dataframes. Here, we have for one station, threshold and forecast file a list of probabilities for each of the lead-times and a binary TRUE or FALSE classification in the observed_event column for each lead-time. We now classify events and non-events based on if there is a 1 or TRUE in any of the lead times filtered to for the observations. While for the forecasted events we take the mean of each probability. When this is complete we finish by adding triggers ranging from 0.01 to 1.0 and run the calcualte_metrics function over each dataframe now groupedby station_name and threshold. We groupby these variables so we account for all forecast files for a given station and threshold. This provides us the contigency scores and verification metrics for each station which we evaluate in the final section.

# %%
# read in saved events csv 
events_df_1 = pd.read_csv('events_df_google.csv')

events_df = events_df_1.set_index(
    ["forecast file", "lead time", "station name", "forecasted date","threshold", "observed event"]
)
events_df = events_df.drop(columns=['Unnamed: 0'])
events_df.reset_index(inplace=True)

# %%
# identify forecasts where lead_time = 0 and observed event = true (i.e., where flooding is already occuring)
forecasts_to_remove = events_df[(events_df['lead time'] == 0) & (events_df['observed event'])][['forecast file', 'station name', 'threshold']].drop_duplicates()

# filter out all rows with the same forecast file, station name, and threshold for any lead time
filtered_events_df = events_df[~events_df[['forecast file', 'station name', 'threshold']].apply(tuple, axis=1).isin(forecasts_to_remove.apply(tuple, axis=1))]

# get the number of forecasts removed including lead time
removed_forecasts_count = len(events_df) - len(filtered_events_df)
print(f"number of forecasts removed: {removed_forecasts_count}")

removed_forecasts_unique_count = forecasts_to_remove.drop_duplicates(subset=['forecast file', 'station name', 'threshold']).shape[0]

print(f"number of unique forecasts removed: {removed_forecasts_unique_count}")

# %%


# assign back to the original variable
events_df = filtered_events_df

all_dfs = []
# test multiple lead times
for lead_start, lead_stop in [[3,5],[3,6],[3,7],[2,5],[2,6],[2,7]]:
    # filter the dataframe to include specific lead times
    events_df_lead = events_df[(events_df['lead time'] >=lead_start) & (events_df['lead time'] <=lead_stop)]
    events_df_lead['lead'] = f'{lead_start}-{lead_stop} days'
    
    # group by station name, lead time category, and threshold
    grouped = events_df_lead.groupby(['forecast file','station name','threshold','lead'], observed=False)

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
            'lead time': [first_row['lead']],
            'observed event': [(df['observed event'] == 1).any()],
            'forecast event': [(df['forecast event'] == 1).any()],
        })

    # combine all resulting dataframe into one 
    final_df = pd.concat(new_grouped_dfs.values(), ignore_index=True)

    # group by station name, lead time category, and threshold
    grouped = final_df.groupby(['station name','threshold','lead time'], observed=False)

    # create a dictionary to store each group's dataframe
    grouped_dfs = {name: group for name, group in grouped}

    # iterate through each dataframe in grouped_dfs, apply calculate_metrics, and store the results back into grouped_dfs
    for key, df in grouped_dfs.items():
        grouped_dfs[key] = calculate_metrics(df)
        
    all_dfs.append(grouped_dfs)

# %%
all_dfs_combine = [pd.concat(dfs, ignore_index=False) for dfs in all_dfs]

# %%
# Concatenate all DataFrames in the list
combined_df = pd.concat(all_dfs_combine, ignore_index=False)
combined_df

# %%
combined_df.reset_index().rename(
    columns={'level_0':'station','level_1':'threshold','level_2':'lead time','level_3':'scores'}
             ).pivot_table(
    index=['station','threshold','lead time'],columns='scores',values=0
).to_csv(f'{country}_skill_scores_google.csv', index=True)

# %%
combined_df.to_csv(f'{country}_combined_output_google.csv', index=True)

# %%
# display an example of one of the grouped_dfs
example = grouped_dfs['beitbridge','bankfull']
example

# %% [markdown]
# Section 5: Trigger selection 
#
# In the final section we evaluate each of the small dataframes in grouped_dfs and evaluate the performance of GloFAS at each gauging station for each of the three thresholds by looking at each of the percentage triggers. Here for each percentage we look at the contigency metrics and skill scores and identify based on previous forecasts how well or not GloFAS can capture flooding. We then pick the best trigger percentage based primarily on the f1 score which balances the hit and false alarm rates. Ideally we want f1 scores above 0.5 and closer to 1.0. The code is set up to filter triggers where the f1 score is greater than 0.45. After we choose the trigger with the highest f1 value. In the event we have more than one trigger left we decide by choosing the one with the highest hit rate, then the lowest false alarm rate and finally the lowest trigger percentage. 
#
# To finish we save the list of the best performing triggers to a csv which can be found in our output folder. These will be our triggers for the operational flood AA system.

# %%
# create an empty dictionary to store the best triggers
best_triggers = {}

# iterate through each dataframe in grouped_dfs
for grouped_dfs in all_dfs:
    for key, df in grouped_dfs.items():
        # find the column names corresponding to trigger thresholds (i.e., the percentages)
        threshold_columns = [0]

        # filter triggers based on the f1 score
        filtered_columns = [
            col for col in threshold_columns
            #if df.loc['f1_score', col] >= 0.45
            # if no triggers fit the criteria, you can get the max f score by uncommenting the next line
            if df.loc['f1_score', col] >= df.loc['f1_score'].max()
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
                        best_threshold = min(best_f1_thresholds, key=lambda x: float(x.split('trigger')[1].split('_')[0]))  # Sorting by numeric threshold
                    else:
                        best_threshold = best_f1_thresholds[0]
                else:
                    best_threshold = best_f1_thresholds[0]
            else:
                best_threshold = best_f1_thresholds[0]

            # split key into station and threshold type (if it's a tuple)
            station, threshold, lead = key if isinstance(key, tuple) else (key, 'unknown')
            # store the best threshold information
            best_triggers[(station, threshold, lead)] = {
                'station': station,
                'threshold': threshold,
                'lead time': lead,
                'best_trigger': best_threshold,
                'f1_score': df.loc['f1_score', best_threshold],
                'hit_rate': df.loc['hit_rate', best_threshold],
                'false_alarm_rate': df.loc['false_alarm_rate', best_threshold]
            }

best_triggers_df = pd.DataFrame(best_triggers.values())

# print the best triggers
print(best_triggers_df)

# Save output as a CSV using country name
filename = f"{country.lower()}_triggers_2025_2026.csv"
#best_triggers_df.to_csv(os.path.join(output_directory, filename), index=False)

# %%
best_triggers_df.loc[best_triggers_df.groupby(['station','threshold'])['f1_score'].idxmax()]

# %%
event_count

# %%
events_df

# %%
best_triggers_df_lead= best_triggers_df.loc[best_triggers_df.groupby(['station','threshold'])['f1_score'].idxmax()]
event_count = events_df.groupby(['station name','threshold']).sum()[['observed event','forecast event']]



# %%
best_triggers_df_lead = event_count.reset_index().rename(
    columns={'observed event':'total observed events','forecast event':'total forecast events','station name':'station'}
).merge(best_triggers_df,how='left',on=['station','threshold'])

best_triggers_df_lead.to_csv(f'{country}_best_triggers_google.csv')

# %% [markdown]
# ## calculate # of exceedances of each threshold

# %%
exc = []

if benchmark == 'observations':
    prefix = 'obs_'
else:
    prefix = 'glofas_'

for c in observed_data.columns:
    if c=='date':
        continue
    info = station_info[station_info['station name']==c]
    if len(info)>0:
        bank = info[f'{prefix}bankfull'].item()
        mod = info[f'{prefix}moderate'].item()
        sev = info[f'{prefix}severe'].item()
        count_b = (observed_data[c]>bank).sum()
        count_m = (observed_data[c]>mod).sum()
        count_s = (observed_data[c]>sev).sum()
        exc.append({
                'station': c,
                'bankfull_exceedance': count_b,
                'moderate_exceedance': count_m,
                'severe_exceedance': count_s,
            })
exc_df = pd.DataFrame(exc)
exc_df

# %%
prefix

# %%
observed_data

# %%
