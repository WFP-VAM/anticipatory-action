# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: aa-env
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
from shapely.geometry import Point, Polygon, LineString, shape
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
# - **Activation** trigger: Activities will be implemented on the pilot districts when they will be impacted by tropical storm-force winds or cyclone-force winds with lead time less or equal to 72 hours.
#
#
# Also, two different intensity speeds are defined and extracted from the data:
#
# - a **Severe Tropical Storm**: wind speed exceeding 89 km/h and lower than 119 km/h
#
# - a **Tropical Cyclone**: wind speed exceeding 119 km/h

# +
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
# **ISO-PROBA 20 % windspeed probabilities**
#
# This file contains one polygon corresponding to the area impacted by a 48kt windspeed with a 20% probability over a 5-day period. This polygon will be used to determine the readiness state of the system. 
#
# - 48ktPROBA20_20222023_7_system_2023-03-10_00Z.json (5 days)
#
# **Track file**
#
# This file contains the time series with the observed coordinates of the system, the forecasted locations of the system with a 6-hr time step, and the associated wind buffers. We also find the uncertainty cone of the track in this file that will be used to get the *exposed_area*.
#
# - CMRSTRACK_20222023_7_system_2023-03-10_00Z.json
#
# For the example below, we'll use a file named differently because this one has a format that is different to the one we expect. So we'll use probas from FREDDY and the track from FILIPO as the track file for FILIPO respects the format that METEO-FRANCE should adopt for the next forecasts. 
#
# You will be able to find all the necessary data on SharePoint for testing: https://wfp.sharepoint.com/:f:/s/HQGeospatial/Egb1kh7sdkRKmPA14mfnuEYBDRrpOAkx6rjV26VL_PpHhg?e=7HqCo4

# +
folder_name = "FREDDY"

# NO PROBS DATA FOUND YET IN FTP SERVER SO FREDDY DATA USED FOR METHODOLOGY DEFINITION

# Readiness polygon: 48kt 5 days 20%
proba_48kt_20_5d_file = f"{folder_name}/48ktPROBA20_20222023_7_FREDDY_2023-03-10_00Z.json"

with open(proba_48kt_20_5d_file) as f:
    proba_polygon_48kt_20p_5d = Polygon(np.array(json.load(f)["features"][0]["geometry"]["coordinates"])[0])

# Track
# We'll use a track file similar to the one from the server as the format is different compared to the example we received
# It corresponds to a different system indeed but the objective here is to stick with the most expectable format to define an output csv
folder_name = "FILIPO"
track_file = f"{folder_name}/2024-03-12T06_44_33_FR-METEOFRANCE-REUNION,PREVISIONS,JSON-CYCLONE.jsonFR-METEOFRANCE-REUNION,PREVISIONS,JSON-CYCLONE.json"  # CMRSTRACK_20222023_7_FREDDY_2023-03-10_00Z"


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
    sorted_columns = sorted(wind_contour_columns, key=lambda col: int(col.split('_')[2].replace('kt', '')))
    for col in sorted_columns:
        col_speed = nautical_mile_to_km(int(col.split('_')[2].replace('kt', '')))
        if wind_speed_kph <= col_speed:
            return col
    return sorted_columns[-1]

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
    return Polygon(points) if points else None


# -

# The `parse_track_json` function returns three main outputs:
#
# - **gdf_track**: A GeoDataFrame containing key information about each time step in the cyclone track. This includes details on the cyclone’s position, development stage, maximum wind speed, gusts, and wind buffers for specified wind speed thresholds (e.g., 48kt and 64kt). Each row represents a time step in the forecast, making gdf_track ideal for tracking cyclone progression. This GeoDataFrame is used to derive landfall information and will be transferred to PRISM for visualizing the cyclone's track and wind buffers.
#
# - **fc_details**: A dictionary with basic metadata about the cyclone forecast, including:
#     - cyclone_name: The name of the cyclone,
#     - season: The cyclone season,
#     - reference_time: The initial timestamp of the forecast,
#     - basin: The geographical basin where the cyclone is located.
#
# - **exposed_areas_set**: A dictionary containing polygons for the uncertainty cone and exposed areas buffered to represent maximum wind impact zones. For each wind speed threshold (48kt and 64kt), the buffer applied to the uncertainty cone corresponds to the maximum radius across all forecasted wind contours. These polygons indicate areas potentially affected by the cyclone’s impact at different intensities, supporting response activities.

def parse_track_json(track_json):
    records = []
    fc_details = {
        "cyclone_name": track_json.get("cyclone_name"),
        "season": track_json.get("season"),
        "reference_time": pd.Timestamp(track_json.get("reference_time"), tz="UTC"),
        "basin": track_json.get("basin")
    }

    uncertainty_cone = None
    max_radii = {ws: 0 for ws in [48, 64]}  # Initialize max radii for 48kt and 64kt winds

    for feature in track_json.get('features', []):
        properties = feature.get('properties', {})
        geometry = feature.get('geometry', {})
        geometry_type = geometry.get('type')
        geometry_coords = geometry.get('coordinates', [])

        if properties.get('data_type') == "uncertainty_cone":
            uncertainty_cone = shape(geometry)
            continue  # Skip adding this feature as a record
        
        record = {
            'data_type': properties.get('data_type'),
            'time': pd.Timestamp(properties.get('time')),
            'position_accuracy': properties.get('position_accuracy'),
            'development': properties.get('cyclone_data', {}).get('development'),
            'maximum_wind_speed': nautical_mile_to_km(float(properties.get('cyclone_data', {}).get('maximum_wind', {}).get('wind_speed_kt', 0))),
            'maximum_wind_gust': nautical_mile_to_km(float(properties.get('cyclone_data', {}).get('maximum_wind', {}).get('wind_speed_gust_kt', 0))),
            'geometry': Point(*geometry_coords) if geometry_type == 'Point' else Polygon(*geometry_coords)
        }
        
        wind_contours = properties.get('cyclone_data', {}).get('wind_contours', [])
        for contour in wind_contours:
            wind_speed_kt = contour.get('wind_speed_kt')
            radii = {
                sector_data.get('sector'): nautical_mile_to_km(sector_data.get('value'))
                for sector_data in contour.get('radius', [])
                if sector_data.get('value') is not None
            }
            record[f'wind_contour_{wind_speed_kt}kt'] = radii
            
            # Check for wind contours to update max radii
            if wind_speed_kt in max_radii:
                max_radii[wind_speed_kt] = max(max_radii[wind_speed_kt], max(radii.values()))

        records.append(record)
    
    # Create exposed area polygons based on max radii
    exposed_areas_set = {}
    for ws, max_radius_km in max_radii.items():
        if max_radius_km > 0:  # Only create if there's a valid max radius
            max_radius_deg = max_radius_km / 111.32  # Convert km to degrees
            if uncertainty_cone is not None:
                exposed_area = uncertainty_cone.buffer(max_radius_deg)  # Buffer the uncertainty cone
                exposed_areas_set[f'exposed_area_{ws}kt'] = exposed_area
    exposed_areas_set['uncertainty_cone'] = uncertainty_cone

    gdf = gpd.GeoDataFrame(records)
    windspeeds = ['48', '64']
    
    # Create wind buffers using wind contours
    for windspeed in windspeeds:
        contour_col = f'wind_contour_{windspeed}kt'
        gdf[f'wind_buffer_{windspeed}'] = [
            create_wind_buffer(row['geometry'], row[contour_col]) 
            if row[contour_col] and row['data_type'] == 'forecast' else np.nan
            for _, row in gdf.iterrows()
        ]

    gdf_track = gdf.drop(len(gdf) - 1) if not gdf.empty else gdf
    
    return gdf_track, fc_details, exposed_areas_set


# +
# Load a PROBS JSON data file
with open(track_file) as f:
    track_data = json.load(f)

gdf, fc_details, exposed_areas_set = parse_track_json(track_data)
# -

gdf.loc[gdf.data_type == 'forecast']

# +
# Conversion to json

time_series_gdf = gdf.drop(['position_accuracy', 'wind_contour_28kt', 'wind_contour_34kt', 'wind_contour_48kt', 'wind_contour_64kt'], axis=1)
time_series_gdf['time'] = time_series_gdf['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
time_series_gdf['wind_buffer_48'] = time_series_gdf['wind_buffer_48'].apply(lambda geom: geom.wkt if geom is not np.nan else geom)
time_series_gdf['wind_buffer_64'] = time_series_gdf['wind_buffer_64'].apply(lambda geom: geom.wkt if geom is not np.nan else geom)
time_series_json = time_series_gdf.to_json()
time_series_json
# -

# ### Get landfall details

# If available, we would like to visualize some information about the landfall on the dashboard. These information are extracted from the GeoDataFrame defined above and then stored in a json dict.  
#
# - The impact time (6hr time interval). This is determined by identifying the timesteps between which the system 48kt wind buffer crosses the coastline. 
# - The impact district: first district impacted, by taking only the center of the track.
# - The impact intensity.

# +
# Add a new column to store 'land' or 'ocean'
gdf['system_location'] = gdf.apply(
    lambda row: 'land' if row['wind_buffer_48'] is not np.nan and row['wind_buffer_48'].intersects(shp.unary_union)
    else ('land' if row['wind_buffer_48'] is np.nan and row['geometry'].intersects(shp.unary_union) else 'ocean'),
    axis=1
)

# Derive the landfall time from the first transition from ocean to land
transition_mask = (gdf['system_location'] == 'land') & (gdf['system_location'].shift() == 'ocean')

if transition_mask.any():
    # Get the first occurrence of landfall
    landfall_index = transition_mask.idxmax()  # Get the index of the first transition

    # Extract landfall time
    landfall_time = [
        gdf.at[landfall_index - 1, 'time'].strftime('%Y-%m-%d %H:%M:%S'), 
        gdf.at[landfall_index, 'time'].strftime('%Y-%m-%d %H:%M:%S')
    ]
    
    # Get the intensity of the system at landfall
    landfall_intensity = [gdf.at[landfall_index - 1, 'development'], gdf.at[landfall_index, 'development']]

    # Create a line segment connecting the centers of the two wind buffers
    center_before = gdf.at[landfall_index - 1, 'geometry']  # Center of the previous position
    center_after = gdf.at[landfall_index, 'geometry']  # Center of the current position

    # Create a line segment between the two centers
    line_segment = LineString([center_before, center_after])

    # Intersect with the coastline
    coastline_intersection = line_segment.intersection(shp.geometry.unary_union)
    intersection_gdf = gpd.GeoDataFrame(geometry=[coastline_intersection], crs=shp.crs)
        
    # Spatial join to find the closest district
    closest_districts = gpd.sjoin(shp, intersection_gdf, how="inner", predicate="intersects")

    # Find the closest district (if any)
    landfall_district = closest_districts.index[0] if not closest_districts.empty else None

    # Check if the identified district is indeed a coastal district
    is_coastal = shp.loc[landfall_district].geometry.intersects(shp.geometry.unary_union)

    # Create JSON dictionary for landfall information
    landfall_info = {
        "landfall_time": landfall_time,
        "landfall_impact_district": landfall_district,
        "landfall_impact_intensity": landfall_intensity,
    }

    print(f"Landfall detected at time: {landfall_time}") 
    print(f"Landfall intensity: {landfall_intensity}")
    print(f"Landfall in district: {landfall_district}")
    print(f"Is impact district coastal: {is_coastal}")

else:
    landfall_info = {}
    print("No landfall detected in the data.")
# -

gdf

# ### Use polygons to get activated districts

# +
# Initialize the dictionary to store all results
ready_set_results = {}

# Readiness: Process each probability polygon in proba_polygons_ready (specifically for proba_20_5d)
intersects_country = proba_polygon_48kt_20p_5d.intersects(shp.unary_union)
ready_set_results['proba_48kt_20_5d'] = {
    "type": "readiness",
    "intersects_country": intersects_country,
    "polygon": proba_polygon_48kt_20p_5d.__geo_interface__
}

# Activation: Process each exposed area polygon in exposed_areas_set
for windspeed in ['48', '64']:  # Iterate over the relevant wind speeds
    exposed_area_polygon = exposed_areas_set[f'exposed_area_{windspeed}kt']  # Get the corresponding exposed area
    exposed_area_gdf = gpd.GeoDataFrame(geometry=[exposed_area_polygon], crs=shp.crs)

    # Perform spatial join to find intersecting districts
    joined_activation = gpd.sjoin(shp, exposed_area_gdf, how="inner", predicate="intersects")

    # Store the affected districts and the polygon geometry in the results dictionary
    ready_set_results[f'exposed_area_{windspeed}kt'] = {
        "affected_districts": joined_activation.index.tolist(),  # List of affected districts
        "polygon": exposed_area_polygon.__geo_interface__  # Store the geometry as GeoJSON
    }

# Display results
ready_set_results
# -

# ### Visualization

# +
import matplotlib.pyplot as plt

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Plot cone of uncertainty
gpd.GeoSeries(exposed_areas_set['uncertainty_cone'].boundary).plot(ax=ax, color='red', linewidth=2, label='Cone of Uncertainty')

# Plot shapefile (e.g., coastline, land areas)
shp.plot(ax=ax, color='none', edgecolor='green', linewidth=0.5, label='Shapefile Layer')

# Plot polygons
gpd.GeoSeries(exposed_areas_set['exposed_area_48kt']).plot(ax=ax, color='yellow', alpha=0.3, label='ExposedArea48kt')
gpd.GeoSeries(exposed_areas_set['exposed_area_64kt']).plot(ax=ax, color='orange', alpha=0.3, label='ExposedArea64kt')

gpd.GeoSeries(proba_polygon_48kt_20p_5d).plot(ax=ax, color='yellow', linewidth=1, label='Proba_48kt_20_5d')

# Plot points
gdf.loc[gdf.data_type == 'analysis'].plot(ax=ax, color='grey', markersize=50, alpha=0.5, label='analysis')

# Plot points
gdf.loc[gdf.data_type == 'forecast'].plot(ax=ax, color='blue', markersize=50, label='forecasts')

# Plot buffers
gdf.set_geometry('wind_buffer_48').boundary.plot(ax=ax, edgecolor='brown', linewidth=0.5, label='Buffer48kt')
gdf.set_geometry('wind_buffer_64').boundary.plot(ax=ax, edgecolor='blue', linewidth=0.5, label='Buffer64kt')


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
    "ready_set_results": ready_set_results  # Store Ready/Set results
}

# Save the dictionary to a JSON file
with open('tropical_storm_output_test.json', 'w') as json_file:
    json.dump(tropical_storm_output, json_file, indent=4)  # Write the JSON with indentation for readability

print("Output saved to tropical_storm_output_test.json")  # Confirmation message
# -

# The output file is also stored on SharePoint for sharing: 
# https://wfp.sharepoint.com/:u:/s/HQGeospatial/EeL5hnrW48RBnwoB2fXMVeQBnt1XFsrILxBsr3aY35FcFg?e=IxmMmz
