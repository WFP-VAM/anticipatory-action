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
# # Observed streamflow data exploration
# This script is made to:
# - explore the data completeness for the observed streamflow at given stations
# - visualize station location on the GloFAS grid to determine if lat/lon values need adjusted 
#
# station data may be given in a range of formats, such as csv/excel, aggregated hourly/daily, with different column headings or in different languages. This script will need to be adjusted to account for these differences

# %%
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import s3fs
import dask
from hip.analysis.compute.utils import persist_with_progress_bar

import cartopy.crs as ccrs
import hvplot.pandas
import hvplot.xarray
import panel as pn
pn.extension()

import holoviews as hv
from holoviews import streams

hv.extension("bokeh", width=90)

import geoviews.feature as gf

# %% [markdown]
# ## step 1: read and map station locations

# %%
country = 'mozambique'  # define country of interest
# directory = Path(f'/s3/scratch/jamie.towner/flood_aa/{country}/data/observations')  # define main working directory
# station_info = pd.read_excel(directory / "raw_data/Limpopo_e_Zambeze/Limpopo/LIMPOPO_Cadastro.xlsx")
directory = Path(r"C:\Users\15133\Documents\WFP\flood_hazard\flood_aa\MOZ_training\data")  # define main working directory
station_info = pd.read_excel(directory / r"observations\raw_data\Limpopo_e_Zambeze\Limpopo\LIMPOPO_Cadastro.xlsx")
station_info.head()

# %% [markdown]
# #### convert locations to decimal degrees
# Latitude and longitude here were given as degrees minutes seconds and must be converted to decimal degrees. The code below will need edited depending on the hemisphere the points are in

# %%
# convert lat lon into decimal degrees
pattern = r'(?P<d>[\d\.]+).*?(?P<m>[\d\.]+).*?(?P<s>[\d\.]+)'

dms = station_info['Latitude'].str.extract(pattern).astype(float)*-1 # *-1 since it is southern hemisphere
station_info['lat'] = dms['d'] + dms['m'].div(60) + dms['s'].div(3600)
dms_lon = station_info['Longitude'].str.extract(pattern).astype(float)
station_info['lon'] = dms_lon['d'] + dms_lon['m'].div(60) + dms_lon['s'].div(3600)

station_info.head()

# %%
# Remove leading/trailing whitespace from metadata station names
station_info['Localidade'] = [
    "".join(c for c in name if c.isalnum() or c in (' ', '_')).replace(' ', '_') for name in station_info['Localidade']
]
station_info.head()

# %% [markdown]
# #### plot station locations

# %%
gdf = gpd.GeoDataFrame(
    station_info, geometry=gpd.points_from_xy(station_info.lon, station_info.lat), crs="EPSG:4326"
)

# adm = gpd.read_file('/s3/scratch/public-share/CDI/boundaries/global_adm2_27082024.shp')
adm = gpd.read_file(directory / r'metadata\MOZ_adm2.shp')
adm = adm[adm['iso3']=='MOZ']

# %%
fig, ax = plt.subplots()
adm.boundary.plot(ax=ax,edgecolor='black',linewidth=0.5)

gdf.plot(ax=ax,label='Limpopo Gauging Stations',legend=True,markersize=15)
plt.legend()
plt.show()

# %% [markdown]
# ## step 2: read in streamflow data and calculate completeness

# %%
# define flood monitoring months of interest
# for the southern hemisphere this is usually October-April
start_month = 10
end_month = 4

# define date, station ID, and streamflow value columns from given observed data
date_column = 'Data'
station_id_column = 'NrEstacao'
value_column = 'Valor'


# %%
def format_streamflow(df,start_month,end_month, date_column, station_id_column, value_column):
    
    # convert date to datetime
    df['date'] = pd.to_datetime(df[date_column])
    
    # format table so stations are columns
    df_pivot = pd.pivot_table(df,values=value_column,index='date',columns=station_id_column)
    # mean by day
    df_pivot = df_pivot.groupby(pd.Grouper(freq='1D')).mean()
    
    # ensure no dates are skipped
    date_range = pd.date_range(start=df_pivot.index.min(), end='2023-12-31', freq='D')
    df_pivot = df_pivot.reindex(date_range)
    df_pivot.index.name = 'date'
    
    # get data completeness only for months of interest
    if start_month>end_month:
        df_pivot_sel = df_pivot[(df_pivot.index.month>=start_month)|(df_pivot.index.month<=end_month)]
    else:
        df_pivot_sel = df_pivot[(df_pivot.index.month>=start_month)&(df_pivot.index.month<=end_month)]
    
    # get completeness from 2003-2023
    df_pivot_sel = df_pivot_sel[(df_pivot_sel.index>='2003-01-01')&(df_pivot_sel.index<='2023-12-31')]
    # get percent of missing values by station
    completeness = (1 - df_pivot_sel.isna().sum()/len(df_pivot_sel))*100
    
    return df_pivot, completeness


# %%
# xl = pd.read_excel(directory / "raw_data/Limpopo_e_Zambeze/Limpopo/LIMPOPO_NIVEL_HIDROMETRICO.xlsx", sheet_name=None)
xl = pd.read_excel(directory / "observations/raw_data/Limpopo_e_Zambeze/Limpopo/LIMPOPO_NIVEL_HIDROMETRICO.xlsx", sheet_name=None)
df = pd.concat(xl)
df_l_1, completeness_l_1 = format_streamflow(df,start_month,end_month, date_column, station_id_column, value_column)

# %%
df_l_1.head()

# %%
# replace station id with names
df_l_1.columns = [station_info[station_info['NrEstacao']==c]['Localidade'].item() for c in df_l_1.columns]

# %%
# save observed data to CSV
# path_csv_full = directory / 'gauging_stations/all_stations/observations_complete_series.csv'
# path_csv_2003_2023 = directory / 'gauging_stations/all_stations/observations.csv'
path_csv_full = directory / 'observations/gauging_stations/all_stations/observations_complete_series.csv'
path_csv_2003_2023 = directory / 'observations/gauging_stations/all_stations/observations.csv'
if Path.exists(path_csv_full):
    print('Station observations file exists. Change file path to save or append to existing csv instead.')
else:
    df_l_1.to_csv(path_csv_full)
    
if Path.exists(path_csv_2003_2023):
    print('Station observations file exists. Change file path to save or append to existing csv instead.')
else:
    df_l_1[(df_l_1.index>='2003-01-01')&(df_l_1.index<='2023-12-31')].to_csv(path_csv_2003_2023)

# %%
completeness_l_1

# %% [markdown]
# ## visualize station locations compared to GloFAS grid

# %%
# # Set up the S3 path for the Zarr files
# store = f"s3://wfp-seasmon/input/cds/glofas-historical/saf/01/*.zarr"

# # Set up connection to s3 store
# s3 = s3fs.S3FileSystem.current()

# # Fetch list of .zarr stores (files)
# remote_files = s3.glob(store)
# store = [
#     s3fs.S3Map(root=f"s3://{file}", s3=s3, check=False) for file in remote_files
# ]
# with dask.config.set(**{"array.slicing.split_large_chunks": True}):
#     ds = xr.open_mfdataset(
#         store,
#         decode_coords="all",
#         engine="zarr",
#         parallel=True,  # Enable parallel processing for speed-up
#         combine="by_coords"
#     )

# # get average glofas streamflow from the past year
# ds_sel = persist_with_progress_bar(ds.isel(time=slice(-365,-1)).mean(dim='time')['dis24'])

# %%
# read in GloFAS reanalysis data from 2023
ds = xr.open_dataset(directory / 'forecasts/glofas_reanalysis/glofas_reanalysis_2023.nc')
# get average glofas streamflow over the year
ds_sel = persist_with_progress_bar(ds.isel(time=slice(-365,-1)).mean(dim='time')['dis24'])

# %% [markdown]
# ### Explore map below to ensure points fall on the gridded river network
# If points do not fall on the river, manually adjust the lisflood_x and lisflood_y in the metadata file so the coordinates fall on the river

# %%
river_map = ds_sel.hvplot.quadmesh("longitude", "latitude", cmap="RdBu", geo=True).opts(clim=(0,100))
station_map = gdf.drop(columns=['Latitude','Longitude']).hvplot.points(geo=True)
river_map * station_map * gf.borders

# %%
