import numpy as np
import pandas as pd
from scipy import stats

# read observed data
observed = pd.read_csv('all_stations_observed.csv')

# read glofas data
glofas = pd.read_csv('all_stations_glofas.csv')

# read metadata for thresholds
metadata = pd.read_csv('metadata_moz.csv')

# initialize a list to store results
results = []

# loop over each station and threshold in the metadata
for index, row in metadata.iterrows():
    station = row['station name']
    
    # get the observed data for the station
    data_observed = observed[station].dropna().values
    
    # get the glofas data for the station
    data_glofas = glofas[station].dropna().values
    
    # define the thresholds to loop over
    thresholds = {
        'obs_bankfull': row['obs_bankfull'],
        'obs_moderate': row['obs_moderate'],
        'obs_severe': row['obs_severe']
    }
    
    # loop over each threshold
    for threshold_name, threshold_value in thresholds.items():
        # calculate percentile rank of the threshold value in the observed dataset
        percentile_rank_observed = stats.percentileofscore(data_observed, threshold_value)
        
        # cap the percentile rank at 100 if it exceeds this value
        percentile_rank_observed = min(percentile_rank_observed, 100)
        
        # calculate the value in glofas corresponding to the observed dataset's percentile rank
        value_glofas = np.percentile(data_glofas, percentile_rank_observed)
        
        # store the results in the list
        results.append({
            'station': station,
            'threshold_name': threshold_name,
            'threshold_value': threshold_value,
            'percentile_rank_observed': percentile_rank_observed,
            'value_glofas': value_glofas
        })

# convert the results list to a dataframe
results_df = pd.DataFrame(results)

# save the results to a csv file
results_df.to_csv('glofas_thresholds_moz.csv', index=False)

