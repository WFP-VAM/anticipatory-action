# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: geo-ds
#     language: python
#     name: python3
# ---

# ## Presentation of the core Tropical Storms monitoring methodology

# ### Import libraries

# +
import json
import numpy as np
import pandas as pd
import geopandas as gpd

from math import pi, radians
from shapely.geometry import Point, Polygon, LineString
from hip.analysis.aoi.analysis_area import AnalysisArea

import warnings
warnings.filterwarnings("ignore")
# -

# ### Define constants

# Before we define some constants, here is some information about the Ready / Set system for Tropical storms. 
#
#
# - **Readiness** trigger: Readiness activities will be implemented when there is a 20% likelihood of tropical storm-force winds impacting Mozambique within the next 120 hours (5 days) with lead time higher than 72 hours. 
#
# - **Activation** trigger: Activities will be implemented on the pilot districts when there is a 50% likelihood of tropical storm-force winds or cyclone-force winds impacting the pilot districts within the next 72 hours (3 days) with lead time less or equal to 72 hours
#
#
# Also, we define two different intensity speeds:
#
# - a **Severe Tropical Storm** will be defined by a wind speed exceeding 89 km/h and lower than 119 km/h
#
# - a **Tropical Cyclone** will be defined by a wind speed exceeding 119 km/h

# +
SEVERE_TS_SPEED = 89 # km/h
CYCLONE_SPEED = 119

NAUTICAL_MILE_TO_KM = 1.852

# Convert knots / miles to km
def nautical_mile_to_km(data):
    return NAUTICAL_MILE_TO_KM * data


# -

# ### Read data

# The first step here is to load the shapefile in order to check when and where the system will cross the coastline and hist Mozambican districts. 

# +
# Define aoi to read datasets using hip-analysis
area = AnalysisArea.from_admin_boundaries(
    iso3="MOZ",
    admin_level=2,
    resolution=0.25,
    datetime_range=f"1981-01-01/2023-06-30",
)

# Read the shapefile || ADMIN 4 NOT AVAILABLE
shp = area.get_dataset([area.BASE_AREA_DATASET])
# -

# For the monitoring we will need the following files:
#
# **Shapefile**
#
# **ISO-PROBA 20 and 50% windspeed probabilities**
#
# These files each contain one polygon corresponding to the area impacted by a XXkt windspeed with a 50 or 20% probability over a 5-day or 3-day period. 
#
# 5-day polygons are used for Readiness so we are ok with the 20% confidence.
# 3-day polygons are used for Activation so we prefer the 50% confidence. 
#
# - 48ktPROBA20_20222023_7_system_2023-03-10_00Z.json (5 days)
# - 48ktPROBA50_20222023_7_system_2023-03-10_00Z.json (3 days)
#
# - 64ktPROBA20_20222023_7_system_2023-03-10_00Z.json (5 days)
# - 64ktPROBA50_20222023_7_system_2023-03-10_00Z.json (3 days)
#
# **Track file**
#
# This file contains the time series with the observed coordinates of the system, the forecasted locations of the system with a 6-hr time step, and the associated wind buffers. We also find the uncertainty cone of the track in this file but we don't need it.
#
# - CMRSTRACK_20222023_7_system_2023-03-10_00Z.json
#
# For the example below, we'll use a file named differently because this one has a format that is different to the one we expect. So we'll use probas from FREDDY and the track from FILIPO as the track file for FILIPO respects the format that METEO-FRANCE should adopt for the next forecasts. 
#
# You will be able to find all the necessary data on SharePoint for testing: https://wfp.sharepoint.com/:f:/s/HQGeospatial/Egb1kh7sdkRKmPA14mfnuEYBDRrpOAkx6rjV26VL_PpHhg?e=7HqCo4

# +
folder_name = "FREDDY"

# NO PROBS DATA FOUND YET IN FTP SERVER SO FREDDY DATA USED FOR METHODOLOGY DEFINITION

# 48kt
proba_48kt_20_5d_file = f"{folder_name}/48ktPROBA20_20222023_7_FREDDY_2023-03-10_00Z.json"
proba_48kt_50_3d_file = f"{folder_name}/48ktPROBA50_20222023_7_FREDDY_2023-03-10_00Z.json" # actually 5d here as 3d missing for now

# 64kt
proba_64kt_20_5d_file = f"{folder_name}/64ktPROBA20_20222023_7_FREDDY_2023-03-10_00Z.json"
proba_64kt_50_3d_file = f"{folder_name}/64ktPROBA50_20222023_7_FREDDY_2023-03-10_00Z.json" # actually 5d here as 3d missing for now

# Track
# We'll use a track file similar to the one from the server as the format is different compared to the example we received
# It corresponds to a different system indeed but the objective here is to stick with the most expectable format to define an output csv
folder_name = "FILIPO"
track_file = f"{folder_name}/2024-03-12T06_44_33_FR-METEOFRANCE-REUNION,PREVISIONS,JSON-CYCLONE.jsonFR-METEOFRANCE-REUNION,PREVISIONS,JSON-CYCLONE.json"  # CMRSTRACK_20222023_7_FREDDY_2023-03-10_00Z"

# +
with open(proba_48kt_20_5d_file) as f:
    proba_48kt_20_5d = Polygon(np.array(json.load(f)["features"][0]["geometry"]["coordinates"])[0])

with open(proba_48kt_50_3d_file) as f:
    proba_48kt_50_3d = Polygon(np.array(json.load(f)["features"][0]["geometry"]["coordinates"])[0])

with open(proba_64kt_20_5d_file) as f:
    proba_64kt_20_5d = Polygon(np.array(json.load(f)["features"][0]["geometry"]["coordinates"])[0])

with open(proba_64kt_50_3d_file) as f:
    proba_64kt_50_3d = Polygon(np.array(json.load(f)["features"][0]["geometry"]["coordinates"])[0])


# List of probability polygons (only 50_3d for districts, 20_5d for country check)
proba_polygons_set = [
    ("proba_48kt_50_3d", proba_48kt_50_3d),  # These should be Polygon or MultiPolygon objects
    ("proba_64kt_50_3d", proba_64kt_50_3d)
]

proba_polygons_ready = [
    ("proba_48kt_20_5d", proba_48kt_20_5d),
    ("proba_64kt_20_5d", proba_64kt_20_5d)
]


# -

# ### Derive time series

# The objective of the following code is to parse a JSON file that contains detailed information about a cyclone's track and translate it into a GeoDataFrame.
#
# The JSON track file typically includes various properties related to the cyclone, such as its name, season, reference time, position accuracy, and geometrical data (coordinates). By converting this information into a GeoDataFrame, we can easily perform spatial analysis and visualization of the cyclone's path and its associated wind data.
#
# The process involves several key steps:
#
# 1. **Parsing the JSON**: Extract relevant details from the JSON structure, including cyclone metadata and feature properties.
#
# 2. **Creating Records**: For each feature in the JSON, create records that encapsulate essential cyclone information, such as wind speed, development stage, and geometrical representation (points or polygons).
#
# 3. **Converting Units**: Convert wind speeds from nautical miles to kilometers for consistency and easier analysis.
#
# 4. **Calculating Wind Buffers**: Generate wind buffer polygons based on the specified wind contours, providing a visual representation of the areas affected by varying wind speeds.
#
# 5. **Constructing the GeoDataFrame**: Compile all records into a GeoDataFrame, which facilitates efficient spatial operations and integrates seamlessly with geographic plotting libraries.
#
# ---
#
# **`get_wind_contour_column(wind_speed_kph, wind_contour_columns)`**
#
# This function determines the appropriate wind contour column based on the given wind speed in kilometers per hour (kph). It iterates through a sorted list of wind contour columns and finds the first column where the wind speed is greater than or equal to the specified wind speed. If none of the columns meet this condition, it returns the largest column in the list. This is useful for mapping wind speeds to their corresponding contour data.
#
# ---
#
# **`create_wind_buffer(point, radii)`**
#
# This function generates a wind buffer polygon around a specified point, based on given radii for different directional sectors (Northeast, Southeast, Southwest, Northwest). It defines sectors in terms of angular ranges and uses them to calculate points at the specified distances from the center point. The radii are converted from kilometers to degrees for accurate geographic representation. The function returns a polygon formed by these calculated points, which represents the area affected by the wind.
#

# +
def get_wind_contour_column(wind_speed_kph, wind_contour_columns):
    # Find the first column with a wind speed greater than or equal to the wind_speed_kt
    for col in sorted(wind_contour_columns):
        col_speed = nautical_mile_to_km(int(col.split('_')[2].replace('kt', '')))
        if wind_speed_kph <= col_speed:
            return col

    # If all columns are smaller, return the largest one
    return sorted(wind_contour_columns)[-1]


def create_wind_buffer(point, radii):    
    sectors = {
        'NEQ': (0, pi/2),
        'SEQ': (pi/2, pi),
        'SWQ': (pi, 3*pi/2),
        'NWQ': (3*pi/2, 2*pi)
    }

    points = []

    # Approximation for converting radius from km to degrees
    km_to_deg = 1 / 111.32
    
    for sector, (start_angle, end_angle) in sectors.items():
        if isinstance(radii, dict):
            radius_km = radii.get(sector)
            radius_deg = radius_km * km_to_deg
            
            # Generate points in this sector
            for angle in np.linspace(start_angle, end_angle, num=25):
                x = point.x + radius_deg * np.cos(angle)
                y = point.y + radius_deg * np.sin(angle)
                points.append((x, y))
    
    # Create polygon from the points
    if points == []:
        return None
    else:
        return Polygon(points)
    

def parse_track_json(track_json):
    # Initialize lists to store the data
    records = []

    # Store forecast details in dict
    fc_details = dict()
    fc_details["cyclone_name"] = track_json["cyclone_name"]
    fc_details["season"] = track_json["season"]
    fc_details["reference_time"] = pd.Timestamp(track_json["reference_time"], tz="UTC")
    fc_details["basin"] = track_json["basin"]

    # Iterate over features to extract all information
    for feature in track_json['features']:
        properties = feature['properties']
        geometry_type = feature['geometry']['type']
        geometry_coords = feature['geometry']['coordinates']

        # Prepare a record
        record = {
            'data_type': properties.get('data_type'),
            'time': pd.Timestamp(properties.get('time')),
            'position_accuracy': properties.get('position_accuracy'),
            'development': properties.get('cyclone_data', {}).get('development'),
            'maximum_wind_speed': properties.get('cyclone_data', {}).get('maximum_wind', {}).get('wind_speed_kt'),
            'maximum_wind_gust': properties.get('cyclone_data', {}).get('maximum_wind', {}).get('wind_speed_gust_kt'),
            'geometry': Point(*geometry_coords) if geometry_type == 'Point' else Polygon(*geometry_coords)
        }

        # Convert to km
        if record['maximum_wind_speed']:
            record['maximum_wind_speed'] = nautical_mile_to_km(float(record['maximum_wind_speed']))
            record['maximum_wind_gust'] = nautical_mile_to_km(float(record['maximum_wind_gust']))

        # Add wind contour data if present
        wind_contours = properties.get('cyclone_data', {}).get('wind_contours', [])
        for contour in wind_contours:
            wind_speed_kt = contour.get('wind_speed_kt')
            radii = {}
            for sector_data in contour.get('radius', []):
                sector = sector_data.get('sector')
                value = sector_data.get('value') # nautical mile
                radii[sector] = nautical_mile_to_km(value)
            # Store the radii as a tuple in the record
            record[f'wind_contour_{wind_speed_kt}kt'] = radii

        records.append(record)

    # Convert the list of records into a DataFrame
    gdf = gpd.GeoDataFrame(records)
    
    # Iterate through rows and calculate the wind buffer
    windspeeds = ['28', '34', '48', '64']
    for windspeed in windspeeds:
        gdf[f'wind_buffer_{windspeed}'] = [
            create_wind_buffer(
                row['geometry'], 
                row[f'wind_contour_{windspeed}kt'] # [get_wind_contour_column(row.maximum_wind_speed, wind_contour_columns)],
            )
            if row[f'wind_contour_{windspeed}kt'] else np.nan
            for _, row in gdf.iterrows()
        ]

    # Drop uncertainty cone that is on the last row of the GeoDataFrame
    gdf_track = gdf.drop(len(gdf) - 1)
    
    return gdf_track, fc_details


# +
# Load a PROBS JSON data file
with open(track_file) as f:
    track_data = json.load(f)

gdf, fc_details = parse_track_json(track_data)


# -

# Next, we classify the system in terms of intensity at each time step in order to get the development stage corresponding to our own criteria. This will mainly be useful for visualization on the dashboard and in order to extract the intensity level at the landfall time.

def classify_storm(row): # keep development stage from raw data
    match row['maximum_wind_speed']:
        case speed if speed >= CYCLONE_SPEED:
            return 'cyclone'
        case speed if CYCLONE_SPEED > speed >= SEVERE_TS_SPEED:
            return 'severe tropical storm'
        case _:
            return 'moderate tropical storm'


# Apply the classify_storm function to the DataFrame
gdf['type'] = gdf.apply(classify_storm, axis=1)

gdf.tail()

# +
# Conversion to json

time_series_gdf = gdf.drop(['position_accuracy', 'development', 'wind_contour_28kt', 'wind_contour_34kt', 'wind_contour_48kt', 'wind_contour_64kt', 'wind_buffer_28', 'wind_buffer_34'], axis=1)
time_series_gdf['time'] = time_series_gdf['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
time_series_gdf['wind_buffer_48'] = time_series_gdf['wind_buffer_48'].apply(lambda geom: geom.wkt if geom is not None else geom)
time_series_gdf['wind_buffer_64'] = time_series_gdf['wind_buffer_64'].apply(lambda geom: geom.wkt if geom is not None else geom)
time_series_json = time_series_gdf.to_json()
time_series_json
# -

# ### Get landfall details

# If available, we would like to visualize some information about the landfall on the dashboard. These information are extracted from the GeoDataFrame defined above and then stored in a json dict.  
#
# - The impact time (6hr time interval). This is determined by identifying the timesteps between which the system wind buffer (here I took 48kt: ***TBC***) crosses the coastline. 
# - The impact district: first district impacted, by taking only center of the track.
# - The impact intensity (taken at the time step right before landfall).

# +
# Determine landfall time

# Add a new column to store 'land' or 'ocean'
gdf['buffer_location'] = gdf['wind_buffer_48'].apply(lambda buffer: 'land' if buffer is not None and buffer.intersects(shp.unary_union) else 'ocean')

# Derive the landfall time from the first transition from ocean to land
transition_mask = (gdf['buffer_location'] == 'land') & (gdf['buffer_location'].shift() == 'ocean')

if transition_mask.any():
    # Get the first occurrence of landfall
    landfall_index = transition_mask.idxmax()  # Get the index of the first transition

    # Extract landfall time
    landfall_time = [
        gdf.at[landfall_index, 'time'].strftime('%Y-%m-%d %H:%M:%S'), 
        gdf.at[landfall_index + 1, 'time'].strftime('%Y-%m-%d %H:%M:%S')
    ]
    
    # Get the intensity of the system at landfall
    landfall_intensity = gdf.at[landfall_index, 'type']

    # Get the coordinates of the center of the system at the time of landfall
    landfall_center = gpd.GeoDataFrame(geometry=[gdf.at[landfall_index + 1, 'geometry']], crs=gdf.crs)

    # Spatial join to find the district touched
    entry_district = gpd.sjoin(shp, landfall_center, how="inner", predicate="intersects")

    # Check the district where landfall will occur
    landfall_district = entry_district.index[0] if not entry_district.empty else None

    # Create JSON dictionary for landfall information
    landfall_info = {
        "landfall_time": landfall_time,
        "landfall_impact_district": landfall_district,
        "landfall_impact_intensity": landfall_intensity
    }

    print(f"Landfall detected at time: {landfall_time}") 
    print(f"Landfall intensity: {landfall_intensity}")
    print(f"Landfall in district: {landfall_district}")

else:
    landfall_info = {}
    print("No landfall detected in the data.")
# -

# ### Use polygons to get activated districts

# +
# Initialize the dictionary to store results for each probability polygon
probability_results = {}

# Process probability polygons and find affected districts
for proba_label, proba_polygon in proba_polygons_set:
    # Create a GeoDataFrame for the current probability polygon
    proba_gdf = gpd.GeoDataFrame(geometry=[proba_polygon], crs=shp.crs)
    
    # Perform spatial join to find intersecting districts
    joined = gpd.sjoin(shp, proba_gdf, how="inner", predicate="intersects")
    
    # Store the affected districts and the polygon geometry in the results dictionary
    probability_results[proba_label] = {
        "affected_districts": joined.index.tolist(),  # List of affected districts
        "polygon": proba_polygon.__geo_interface__  # Store the geometry as GeoJSON
    }

# Check intersection for 20_5d probability polygons
intersect_country_results = {}

for proba_label, proba_polygon in proba_polygons_ready:
    # Check intersection with the country boundary
    intersect_country_results[proba_label] = proba_polygon.intersects(shp.unary_union)

# Add intersection results to the results dictionary
for proba_label in intersect_country_results:
    if proba_label in probability_results:
        probability_results[proba_label]["intersects_country"] = intersect_country_results[proba_label]
    else:
        # In case the polygon isn't already in the results, add it
        probability_results[proba_label] = {
            "intersects_country": intersect_country_results[proba_label],
            "polygon": proba_polygon.__geo_interface__
        }

# Display results
probability_results
# -

# ### Visualization

# +
import matplotlib.pyplot as plt

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Plot cone of uncertainty
# gpd.GeoSeries(uncertainty_cone.boundary).plot(ax=ax, color='red', linewidth=2, label='Cone of Uncertainty')

# Plot points
gdf.loc[gdf.data_type == 'analysis'].plot(ax=ax, color='grey', markersize=50, alpha=0.5, label='analysis')

# Plot points
gdf.loc[gdf.data_type == 'forecast'].plot(ax=ax, color='blue', markersize=50, label='forecasts')

# Plot buffers
gdf.set_geometry('wind_buffer_34').boundary.plot(ax=ax, edgecolor='orange', linewidth=0.5, label='Buffer34kt')
gdf.set_geometry('wind_buffer_48').boundary.plot(ax=ax, edgecolor='brown', linewidth=0.5, label='Buffer48kt')
gdf.set_geometry('wind_buffer_64').boundary.plot(ax=ax, edgecolor='blue', linewidth=0.5, label='Buffer64kt')

# Plot shapefile (e.g., coastline, land areas)
shp.plot(ax=ax, color='none', edgecolor='green', linewidth=0.5, label='Shapefile Layer')

# Plot probability polygons
gpd.GeoSeries(proba_48kt_50_3d).plot(ax=ax, color='purple', alpha=0.3, label='Proba 48kt 50_3d')
gpd.GeoSeries(proba_48kt_20_5d).boundary.plot(ax=ax, edgecolor='yellow', linewidth=1, label='Proba 48kt 20_5d')


# Add legend and titles
ax.legend(loc='upper right')
ax.set_title(f"Storm forecasts and Wind Buffers - {fc_details['reference_time']}")
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.show()
# -

# ### Save final output

# +
# Create the main dictionary to store all data
tropical_storm_output = {
    "landfall_info": landfall_info,  # Store landfall information
    "time_series": time_series_gdf.__geo_interface__,  # Convert GeoDataFrame to GeoJSON format
    "probability_results": probability_results  # Store probability results
}

# Save the dictionary to a JSON file
with open('tropical_storm_output_test.json', 'w') as json_file:
    json.dump(tropical_storm_output, json_file, indent=4)  # Write the JSON with indentation for readability

print("Output saved to tropical_storm_output_test.json")  # Confirmation message
# -

# The output file is also stored on SharePoint for sharing: 
# https://wfp.sharepoint.com/:u:/s/HQGeospatial/EeL5hnrW48RBnwoB2fXMVeQBnt1XFsrILxBsr3aY35FcFg?e=IxmMmz

# **Remaining tasks:**
#
# - Drop 64 5-day: only use 48 for Readiness
# - Landfall: add leadtime until landfall + intensity range + patch to check if coastal district else interpolate
# - Remove classify_storm function and take development stage from data
# - Decide whether pilot district filtering should be done here or downstream
