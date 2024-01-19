# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# This notebook is not used operationally or for any validation, its purpose is to have a clear understanding of the core functions of the AA workflow. The outputs and dimensions of each main step can thus be identified here. It can also be used to run the operational workflow for a very specific index or issue month. However, in order to better compare the outputs with the reference ones, some very simple analysis plots/tables will be added. 

# +
import datetime
import pandas as pd
import geopandas as gpd

from config.params import Params 

from helper_fns import (
    read_forecasts_locally,
    read_observations_locally,
    aggregate_by_district,
    merge_un_biased_probs,
    merge_probabilities_triggers_dashboard,
)

# from analysis.aoi import AnalysisArea, AnalysisAreaData, Country
from hip.analysis.analyses.drought import (
    get_accumulation_periods,
    run_accumulation_index,
    run_gamma_standardization,
    run_bias_correction,
    compute_probabilities,
)

# %cd ../
# -

params = Params(iso='MOZ', issue=10, index='SPI')

# To replace with HDC dataset
forecasts = read_forecasts_locally(
    f"data/{params.iso}/forecast/{params.iso}_SAB_tp_ecmwf_{str(params.issue).zfill(2)}/*.nc"
)
forecasts

# To replace with CHIRPS (rfh_daily for DRYSPELL or r1h_dekad if SPI)
observations = read_observations_locally(f"data/{params.iso}/chirps")
observations

# Read triggers file
triggers_df = pd.read_csv(
    f"data/{params.iso}/outputs/Plots/triggers.aa.python.spi.dryspell.2022.csv"
)

gdf = gpd.read_file(
    f"data/{params.iso}/shapefiles/moz_admbnda_2019_SHP/moz_admbnda_adm2_2019.shp"
)

# Get accumulation periods (DJ, JF, FM, DJF, JFM...)
accumulation_periods = get_accumulation_periods(
    forecasts,
    params.start_season,
    params.end_season,
    params.min_index_period,
    params.max_index_period,
)

# Get single use case
period_name, period_months = list(accumulation_periods.items())[6]
period_name, period_months

# Remove 1980 season to harmonize observations between different indexes
if int(params.issue) >= params.end_season:
    observations = observations.where(
        observations.time.dt.date >= datetime.date(1981, 10, 1), drop=True
    )

# Accumulation
accumulation_fc = run_accumulation_index(
    forecasts, params.aggregate, period_months, forecasts=True
)
accumulation_obs = run_accumulation_index(
    observations, params.aggregate, period_months
)
display(accumulation_fc)

# Remove inconsistent observations
accumulation_obs = accumulation_obs.sel(
    time=slice(datetime.date(1979, 1, 1), datetime.date(params.year - 1, 12, 31))
)

# Anomaly
anomaly_fc = run_gamma_standardization(
    accumulation_fc, params.calibration_start, params.calibration_stop, members=True
)
anomaly_obs = run_gamma_standardization(
    accumulation_obs,
    params.calibration_start,
    params.calibration_stop,
)
display(anomaly_fc)

# Probabilities without Bias Correction
probabilities = compute_probabilities(
    anomaly_fc.where(anomaly_fc.time.dt.year == params.year, drop=True),
    levels=params.intensity_thresholds,
).round(2)
display(probabilities)

# Bias correction
index_bc = run_bias_correction(
    anomaly_fc, 
    anomaly_obs, 
    params.end_season,
    params.year,
    int(params.issue),
    enso=True,
)
display(index_bc)

# Probabilities after Bias Correction
probabilities_bc = compute_probabilities(
    index_bc, levels=params.intensity_thresholds
).round(2)
display(probabilities_bc)

# +
# Aggregate by district
probs_district = aggregate_by_district(probabilities, gdf, params)
probs_bc_district = aggregate_by_district(probabilities_bc, gdf, params)

# Build single xarray with merged unbiased/biased probabilities
probs_by_district = merge_un_biased_probs(
    probs_district, probs_bc_district, params, period_name
)
display(probs_by_district)
# -

# Merge probabilities with triggers
probs_df, merged_df = merge_probabilities_triggers_dashboard(
    probs_by_district, triggers_df, params, period_name
)

probs_df

merged_df
