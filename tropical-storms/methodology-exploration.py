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

# +
import glob
import json
import numpy as np
import pandas as pd
import geopandas as gpd

from pykml import parser
from math import pi, radians
from shapely.geometry import Point, Polygon, LineString
from hip.analysis.aoi.analysis_area import AnalysisArea

# +
READINESS_LEAD_TIME = pd.Timedelta(hours=120)
ACTIVATION_LEAD_TIME = pd.Timedelta(hours=60)
LANDFALL_LEAD_TIME = pd.Timedelta(hours=6)

SEVERE_TS_SPEED = 89 # km/h
CYCLONE_SPEED = 119

NAUTICAL_MILE_TO_KM = 1.852

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
shp
# -

json_file = '2024-03-12T06_44_33_FR-METEOFRANCE-REUNION,PREVISIONS,JSON-CYCLONE.jsonFR-METEOFRANCE-REUNION,PREVISIONS,JSON-CYCLONE.json'

fc_date = pd.Timestamp(json_file[:13], tz="UTC")
fc_date

# Load the JSON data
with open(json_file) as f:
    data = json.load(f)


def parse_json(json):
    # Initialize lists to store the data
    records = []

    # Define the consistent order of sectors
    sectors_order = ['NEQ', 'SEQ', 'SWQ', 'NWQ']    

    # Iterate over features to extract all information
    for feature in json['features']:
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

    # Keep uncertainty cone separately
    uncertainty_cone = gdf.iloc[-1].geometry
    gdf = gdf.drop(len(gdf) - 1)
    
    return gdf, uncertainty_cone



# +
gdf, uncertainty_cone = parse_json(data)

gdf_forecast = gdf.loc[gdf.data_type == 'forecast']
gdf_analysis = gdf.loc[gdf.data_type == 'analysis']


# +
# Convert knots / miles to km
def nautical_mile_to_km(data):
    return NAUTICAL_MILE_TO_KM * data
    
gdf_forecast['maximum_wind_speed'] = nautical_mile_to_km(gdf_forecast.maximum_wind_speed)
gdf_forecast['maximum_wind_gust'] = nautical_mile_to_km(gdf_forecast.maximum_wind_gust)
# -

uncertainty_cone


# +
def determine_status(leadtime):
    match leadtime:
        case lt if ACTIVATION_LEAD_TIME < lt <= READINESS_LEAD_TIME:
            return 'ready'
        case lt if LANDFALL_LEAD_TIME < lt <= ACTIVATION_LEAD_TIME:
            return 'set'
        case _:
            return 'not activated'

def classify_storm(row):
    match row['status']:
        case 'set':
            match row['maximum_wind_speed']:
                case speed if speed >= CYCLONE_SPEED:
                    return 'cyclone'
                case speed if CYCLONE_SPEED > speed >= SEVERE_TS_SPEED:
                    return 'severe tropical storm'
                case _:
                    return 'moderate tropical storm'
        case _:
            return pd.NA


# +
# Determine leadtime from issue date and forecast date
gdf_forecast['leadtime'] = gdf_forecast.time - fc_date

# Apply the status function to the leadtime column
gdf_forecast['status'] = gdf_forecast['leadtime'].apply(determine_status)
# -

# Apply the classify_storm function to the DataFrame
gdf_forecast['type'] = gdf_forecast.apply(classify_storm, axis=1)

gdf_forecast.head()


# +
def get_wind_contour_column(wind_speed_kph, wind_contour_columns):
    # Find the first column with a wind speed greater than or equal to the wind_speed_kt
    for col in sorted(wind_contour_columns):
        col_speed = nautical_mile_to_km(int(col.split('_')[2].replace('kt', '')))
        if wind_speed_kph <= col_speed:
            return col

    # If all columns are smaller, return the largest one
    return sorted(wind_contour_columns)[-1]


def create_wind_buffer(point, wind_speed_kph, radii):    
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
        radius_km = radii.get(sector)
        if radius_km is not None:
            radius_deg = radius_km * km_to_deg
            
            # Generate points in this sector
            for angle in np.linspace(start_angle, end_angle, num=25):
                x = point.x + radius_deg * np.cos(angle)
                y = point.y + radius_deg * np.sin(angle)
                points.append((x, y))
    
    # Create polygon from the points
    wind_buffer = Polygon(points)
    
    return wind_buffer


# -

# Iterate through rows and calculate the wind buffer
wind_contour_columns = gdf.columns[gdf.columns.str.startswith("wind_contour")]
gdf_forecast['wind_buffer_28'] = [
    create_wind_buffer(
        row.geometry, 
        row.maximum_wind_speed, 
        row['wind_contour_28kt'] # [get_wind_contour_column(row.maximum_wind_speed, wind_contour_columns)],
    )
    for _, row in gdf_forecast.iterrows()
]
gdf_forecast['wind_buffer_34'] = [
    create_wind_buffer(
        row.geometry, 
        row.maximum_wind_speed, 
        row['wind_contour_34kt'] # [get_wind_contour_column(row.maximum_wind_speed, wind_contour_columns)],
    )
    for _, row in gdf_forecast.iterrows()
]
gdf_forecast['wind_buffer_48'] = [
    create_wind_buffer(
        row.geometry, 
        row.maximum_wind_speed, 
        row['wind_contour_48kt'] # [get_wind_contour_column(row.maximum_wind_speed, wind_contour_columns)],
    )
    for _, row in gdf_forecast.iterrows()
]

gdf_forecast

# +
import matplotlib.pyplot as plt

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Plot cone of uncertainty
gpd.GeoSeries(uncertainty_cone.boundary).plot(ax=ax, color='red', linewidth=2, label='Cone of Uncertainty')

# Plot points
gdf_analysis.plot(ax=ax, color='grey', markersize=50, alpha=0.5, label='analysis')

# Plot points
gdf_forecast.plot(ax=ax, color='blue', markersize=50, label='forecasts')

# Plot buffers
gdf_forecast.set_geometry('wind_buffer_28').boundary.plot(ax=ax, edgecolor='blue', linewidth=0.5, label='Buffer28kt')
gdf_forecast.set_geometry('wind_buffer_34').boundary.plot(ax=ax, edgecolor='orange', linewidth=0.5, label='Buffer34kt')
gdf_forecast.set_geometry('wind_buffer_48').boundary.plot(ax=ax, edgecolor='brown', linewidth=0.5, label='Buffer48kt')

# Plot shapefile (e.g., coastline, land areas)
shp.plot(ax=ax, color='none', edgecolor='green', linewidth=0.5, label='Shapefile Layer')

# Add legend and titles
ax.legend(loc='upper right')
ax.set_title(f'Storm forecasts and Wind Buffers - {fc_date}')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.show()

# +
# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Plot filtered shapefile (e.g., country with districts)
shp.plot(ax=ax, color='none', edgecolor='green', linewidth=0.5, label='Country Districts')

# Plot points
gdf_analysis.plot(ax=ax, color='grey', markersize=50, alpha=0.5, label='Analysis Points')

# Plot forecast points
gdf_forecast.plot(ax=ax, color='blue', markersize=50, label='Forecast Points')

# Plot buffers
gdf_forecast.set_geometry('wind_buffer_28').boundary.plot(ax=ax, edgecolor='blue', linewidth=0.5, label='Buffer28kt')
gdf_forecast.set_geometry('wind_buffer_34').boundary.plot(ax=ax, edgecolor='orange', linewidth=0.5, label='Buffer34kt')
gdf_forecast.set_geometry('wind_buffer_48').boundary.plot(ax=ax, edgecolor='brown', linewidth=0.5, label='Buffer48kt')

xmin, xmax = 30, 38
ymin, ymax = -27, -18

# Add district names (labeling)
for x, y, label in zip(shp.geometry.centroid.x, shp.geometry.centroid.y, shp['Name']):
    if ymin < y < ymax and xmin < x < xmax:
        ax.text(x, y, label, fontsize=8, ha='center', va='center', color='darkgreen')

# Set plot limits to zoom in on the country
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

# Add legend and titles
ax.legend(loc='upper right')
ax.set_title('Zoomed-In Visualization of Wind Buffers and Districts - {fc_date}')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.show()

# +
# Set the geometry of gdf_points to 'buffer' for spatial join
gdf_buffer = gdf_forecast.set_geometry('wind_buffer_28')

# Perform spatial join to find intersecting districts
intersecting_districts = gpd.sjoin(shp, gdf_buffer, how='right', predicate='intersects')
# -

gdf_districts = intersecting_districts.drop(['index_left', 'Code', 'adm1_Code', 'adm0_Code'], axis=1)

gdf_districts.to_csv("test_districts_df_storms.csv", index=False)

gdf_districts





kml_file = "2024-03-12T06_44_04_FR-METEOFRANCE.kml"

fc_date = pd.Timestamp(kml_file[:13])
fc_date


def parse_kml(file_path):
    with open(file_path, 'r') as file:
        root = parser.parse(file).getroot()

    placemarks = root.findall('.//{http://www.opengis.net/kml/2.2}Placemark')

    data = []
    
    for placemark in placemarks:
        name = placemark.find('{http://www.opengis.net/kml/2.2}name').text
        coordinates = placemark.find('.//{http://www.opengis.net/kml/2.2}coordinates')
        
        if coordinates is not None:
            coords = coordinates.text.strip().split()
            if placemark.find('.//{http://www.opengis.net/kml/2.2}Point') is not None:
                # For Point geometries
                timestamp = pd.Timestamp(name.split(" : ")[-1])
                lon, lat, alt = map(float, coords[0].split(','))
                geom = Point(lon, lat, alt)
                balloon_text = placemark.find('.//{http://www.opengis.net/kml/2.2}BalloonStyle/{http://www.opengis.net/kml/2.2}text')
                wind_speed = None
                
                if balloon_text is not None:
                    text = balloon_text.text
                    if "Wind Speed" in text:
                        wind_speed = float(text.split("Wind Speed : ")[1].split(" knots")[0])
                
                data.append({
                    'timestamp': timestamp,
                    'latitude': lat,
                    'longitude': lon,
                    'wind_speed': 1.852 * wind_speed, # km/h
                    'geometry': geom,
                })

            elif placemark.find('.//{http://www.opengis.net/kml/2.2}LineString') is not None:
                # For LineString geometries (Track Line)
                line_coords = [tuple(map(float, c.split(','))) for c in coords]
                track_line = LineString(line_coords)

            elif placemark.find('.//{http://www.opengis.net/kml/2.2}Polygon') is not None:
                # For Polygon geometries (Uncertainty Cone)
                poly_coords = [tuple(map(float, c.split(','))) for c in coords]
                uncertainty_cone = Polygon(poly_coords)
    
    df = pd.DataFrame(data)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    return gdf, track_line, uncertainty_cone


gdf, track_line, uncertainty_cone = parse_kml(kml_file)

track_line

uncertainty_cone

gdf

gdf['leadtime'] = gdf.timestamp - fc_date

# Apply the status function to the leadtime column
gdf['status'] = gdf['leadtime'].apply(determine_status)
gdf

# Apply the classify_storm function to the DataFrame
gdf['type'] = gdf.apply(classify_storm, axis=1)
gdf

# +
# Calculate the shortest distance from each point to the cone
gdf['buffer'] = gdf.geometry.apply(lambda point: point.distance(uncertainty_cone.boundary))

# Create buffers around each point using the shortest distance
gdf['buffer'] = gdf.apply(lambda row: row['geometry'].buffer(row['buffer']), axis=1)
# -

gdf

# +
import matplotlib.pyplot as plt

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Plot cone of uncertainty
gpd.GeoSeries(uncertainty_cone).plot(ax=ax, color='red', linewidth=2, label='Cone of Uncertainty')

# Plot points
gdf.plot(ax=ax, color='blue', markersize=50, label='Points')

# Plot buffers
gdf.loc[gdf.leadtime >= pd.Timedelta(0)].set_geometry('buffer').plot(ax=ax, color='lightblue', alpha=0.5, edgecolor='blue', linewidth=0.5, label='Buffers')
# -

# Keep only rows corresponding to some activation
gdf = gdf.loc[gdf.status != 'not activated']

# +
# Set the geometry of gdf_points to 'buffer' for spatial join
gdf_buffer = gdf.set_geometry('buffer')

# Perform spatial join to find intersecting districts
intersecting_districts = gpd.sjoin(shp, gdf_buffer, how='right', predicate='intersects')
# -

final_df = intersecting_districts[["Name", "timestamp", "latitude", "longitude", "wind_speed", "geometry", "leadtime", "status", "storm", "buffer"]]
final_df.columns = ["district", "timestamp", "latitude", "longitude", "windspeed", "geometry", "leadtime", "status", "type", "buffer"]

final_df # admin 4


