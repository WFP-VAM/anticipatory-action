# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: hip-analysis-dev-env
#     language: python
#     name: python3
# ---

# This notebook is not used operationally or for any validation, its only purpose is to have a clear understanding of the core functions of the AA workflow. The outputs and dimensions of each main step can thus be identified here.

# **Import required libraries and functions**

# %cd ..

# +
import os
import datetime
import pandas as pd

from config.params import Params

from AA.helper_fns import (
    read_forecasts,
    read_observations,
    aggregate_by_district,
    merge_un_biased_probs,
    merge_probabilities_triggers_dashboard,
)

from hip.analysis.analyses.drought import (
    get_accumulation_periods,
    run_accumulation_index,
    run_gamma_standardization,
    run_bias_correction,
    compute_probabilities,
)

from hip.analysis.aoi.analysis_area import AnalysisArea
# -

# **Define parameters**

# The `config/{country}_config.yaml` file gathers all the parameters used in the operational script and that can be customized. For example, the *monitoring_year*, the list of districts or the intensity levels can be defined in that file.

params = Params(iso='ZWE', issue=10, index='SPI')
params.monitoring_year = 2023

# **Read shapefile**

# +
# Define aoi to read datasets using hip-analysis
area = AnalysisArea.from_admin_boundaries(
    iso3=params.iso.upper(),
    admin_level=2,
    resolution=0.25,
    datetime_range=f"1981-01-01/{params.monitoring_year + 1}-06-30",
)

# Read the shapefile
gdf = area.get_dataset([area.BASE_AREA_DATASET])
gdf
# -

# **Read forecasts**

# When update is set to False, the downscaled dataset is read from a local folder or a s3 bucket. Otherwise, it is directly read from HDC.
forecasts = read_forecasts(
    area,
    params.issue,
    f"{params.data_path}/data/{params.iso}/zarr/2022/{str(params.issue).zfill(2)}/forecasts.zarr",
    update=False,  # True,
)
forecasts

# **Read observations**

# Observations data reading (already stored as the dataset used is the same as the one used in the pre-season/analytical script)
observations = read_observations(
    area,
    f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/obs/observations.zarr",
)
observations

# **Read pre-computed triggers**

# Now that we got all the data we need, let's read the triggers file so we can merge the probabilities with it once we have them.

# Read triggers file
if os.path.exists(f"{params.data_path}/data/{params.iso}/probs/aa_probabilities_triggers_pilots.csv"):
    triggers_df = pd.read_csv(
        f"{params.data_path}/data/{params.iso}/probs/aa_probabilities_triggers_pilots.csv",
    )
else:
    triggers_df = pd.read_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.pilots.csv",
    )
triggers_df

# **Get accumulation periods covered by the forecasts of the defined issue month**

# Get accumulation periods (DJ, JF, FM, DJF, JFM...)
accumulation_periods = get_accumulation_periods(
    forecasts,
    params.start_season,
    params.end_season,
    params.min_index_period,
    params.max_index_period,
)
accumulation_periods

# Here we focus on the pipeline for one indicator (one period) so we select a single element from the above dictionary (November-December using October forecasts).

# Get single use case
period_name, period_months = list(accumulation_periods.items())[4]
period_name, period_months

# **Run accumulation (sum for SPI)**

# Remove 1980 season to harmonize observations between different indexes 
if int(params.issue) >= params.start_monitoring:
    observations = observations.where(
        observations.time.dt.date >= datetime.date(1981, 10, 1), drop=True
    )

# Accumulation
accumulation_fc = run_accumulation_index(
    forecasts.chunk(dict(time=-1)), params.aggregate, period_months, forecasts=True
)
accumulation_obs = run_accumulation_index(
    observations.chunk(dict(time=-1)), params.aggregate, period_months
)

accumulation_fc

accumulation_obs

# **Run standardization (SPI)**

# Remove inconsistent observations
accumulation_obs = accumulation_obs.sel(
    time=slice(datetime.date(1979, 1, 1), datetime.date(params.monitoring_year - 1, 12, 31))
)

# Anomaly
anomaly_fc = run_gamma_standardization(
    accumulation_fc.load(),
    params.hist_anomaly_start,
    params.hist_anomaly_stop,
    members=True,
)
anomaly_obs = run_gamma_standardization(
    accumulation_obs.load(),
    params.hist_anomaly_start,
    params.hist_anomaly_stop,
)

anomaly_fc

anomaly_obs

# **Run bias correction**

# Bias correction
index_bc = run_bias_correction(
    anomaly_fc,
    anomaly_obs,
    params.end_season,
    params.monitoring_year,
    int(params.issue),
    nearest_neighbours=8,
    enso=True,
)
display(index_bc)

# **Run probabilities**

# Change dryspell sign as we compare values to a negative threshold to get probabilities
if params.index == "dryspell":
    anomaly_fc *= -1
    index_bc *= -1
    anomaly_obs *= -1

# Probabilities without Bias Correction
probabilities = compute_probabilities(
    anomaly_fc.where(anomaly_fc.time.dt.year == params.monitoring_year, drop=True),
    levels=params.intensity_thresholds,
).round(2)
display(probabilities)

# Probabilities after Bias Correction
probabilities_bc = compute_probabilities(
    index_bc, levels=params.intensity_thresholds
).round(2)
display(probabilities_bc)

# **Admin-2 level aggregation**

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

# **Dataframe formatting**

# Merge probabilities with triggers
probs_df, merged_df = merge_probabilities_triggers_dashboard(
    probs_by_district, triggers_df, params, period_name
)

probs_df

merged_df
