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
# Create a function to sort and process a list of gauging station csv files to create a 
# single csv of all river discharge data for one country. Note set output path to a different directory
# to prevent script failing if rerunning. 

# %%
import pandas as pd
from os import listdir
from os.path import isfile, join

# %%
directory = '/s3/scratch/jamie.towner/flood_aa/zimbabwe/data/observations/gauging_stations' # define directory 


# %%
def process_station_files(input_folder, output_file):
    # Get a list of all CSV files in the input folder
    station_files = [f for f in listdir(input_folder) if isfile(join(input_folder, f)) and f.lower().endswith('.csv')]

    # Initialise an empty data frame to store the combined data
    combined_df = pd.DataFrame()
    
    # Initialise variables to store the minimum and maximum dates across all stations
    min_date = pd.Timestamp.max
    max_date = pd.Timestamp.min

    # Process each station file
    for station_file in station_files:
        # Read CSV file into a data frame
        df = pd.read_csv(join(input_folder, station_file))

        # Convert date column to datetime format
        df['date'] = pd.to_datetime(df['date'], format='mixed')

        # Update min_date and max_date based on the current station's dataset
        #min_date = min(min_date, df['date'].min())
        #max_date = max(max_date, df['date'].max())
        # or choose specific dates
        min_date ='2003-01-01'
        max_date = '2023-12-31'

        # Remove rows with negative values 
        df = df[df['discharge'] >= 0]

        # Calculate the average river discharge for multiple entries within a single day
        df = df.groupby('date')['discharge'].mean().reset_index()
        
        # Rename column to include the station name
        station_name = station_file.split('.')[0]  # Extract station name from filename
        df.rename(columns={'discharge': f'{station_name}'}, inplace=True)

        # Merge the current station's data with the combined data frame
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on='date', how='outer')
            
    # Reindex the data frame with a complete date range for all stations
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    combined_df = combined_df.set_index('date').reindex(date_range).reset_index()
    
    combined_df = combined_df.rename(columns={combined_df.columns[0]: "date"})

    # Replace NaN with 'NA'
    combined_df = combined_df.fillna('NA')

    # Save the data frame to a new CSV file
    combined_df.to_csv(output_file, index=False)
    
# Run the function:
input_folder_path = '/s3/scratch/jamie.towner/flood_aa/zimbabwe/data/observations/gauging_stations'
output_file_path = '/s3/scratch/jamie.towner/flood_aa/zimbabwe/data/observations/gauging_stations/all_stations/observations.csv' 
process_station_files(input_folder_path, output_file_path)

# %%
