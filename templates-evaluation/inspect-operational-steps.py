# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: hdc
#     language: python
#     name: conda-env-hdc-py
# ---

# This notebook is not used operationally or for any validation, its only purpose is to have a clear understanding of the core functions of the AA workflow. The outputs and dimensions of each main step can thus be identified here.

# **Import required libraries and functions**

# %cd ..

import datetime

# +
import os

import pandas as pd
from hip.analysis.analyses.drought import (
    compute_probabilities,
    get_accumulation_periods,
    run_accumulation_index,
    run_bias_correction,
    run_gamma_standardization,
)
from hip.analysis.aoi.analysis_area import AnalysisArea

from AA.src.utils import (
    compute_district_average,
    merge_probabilities_triggers_dashboard,
    merge_un_biased_probs,
    read_forecasts,
    read_observations,
    read_triggers,
)
from AA.src.params import Params

# -

# **Define parameters**

# The `config/{country}_config.yaml` file gathers all the parameters used in the operational script and that can be customized. For example, the *monitoring_year*, the list of districts or the intensity levels can be defined in that file.

params = Params(
    iso="TZA", 
    issue=6, 
    index="SPI",
    data_path = "s3://wfp-ops-userdata/amine.barkaoui/aa",
    output_path = "s3://wfp-ops-userdata/amine.barkaoui/aa"
)

# **Read shapefile**

# +
# Define aoi to read datasets using hip-analysis
area = AnalysisArea.from_admin_boundaries(
    iso3=params.iso.upper(),
    admin_level=2,
    resolution=0.25,
    datetime_range=f"1981-01-01/{params.calibration_year}-06-30",
)

# Read the shapefile
gdf = area.get_dataset([area.BASE_AREA_DATASET])
gdf
# -

# **Read forecasts**

# +
# When update is set to False, the downscaled dataset is read from a local folder or a s3 bucket. Otherwise, it is directly read from HDC.
forecasts_folder_path = f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}"

forecasts = read_forecasts(
    area,
    params.issue,
    f"{forecasts_folder_path}/{str(params.issue).zfill(2)}/forecasts.zarr",
)
forecasts
# -

# **Read observations**

# Observations data reading (already stored as the dataset used is the same as the one used in the pre-season/analytical script)
observations = read_observations(
    area,
    f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/obs/observations.zarr",
)
observations

# **Read pre-computed triggers**

# Now that we got all the data we need, let's read the triggers file so we can merge the probabilities with it once we have them.

# + jupyter={"outputs_hidden": true}
# Read triggers file
triggers_df = read_triggers(params)
# -

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
period_name, period_months = list(accumulation_periods.items())[0]  # [4]
period_name, period_months

# **Run accumulation (sum for SPI)**

# Remove 1980 season to harmonize datasets between different indexes
forecasts = forecasts.where(
    forecasts.time.dt.date >= datetime.date(1981, params.start_season, 1), drop=True
)
observations = observations.where(
    observations.time.dt.date >= datetime.date(1981, params.start_season, 1), drop=True
)

# Accumulation
accumulation_fc = run_accumulation_index(
    forecasts.chunk(dict(time=-1)),
    params.aggregate,
    period_months,
    (params.start_season, params.end_season),
    forecasts=True,
)
accumulation_obs = run_accumulation_index(
    observations.chunk(dict(time=-1)),
    params.aggregate,
    period_months,
    (params.start_season, params.end_season),
)

# **Run standardization (SPI)**

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

# **Run bias correction**

# Bias correction
index_bc = run_bias_correction(
    anomaly_fc,
    anomaly_obs,
    start_monitoring=params.start_monitoring,
    year=params.monitoring_year,
    issue=int(params.issue),
    nearest_neighbours=8,
    enso=True,
)

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
print(probabilities)

# Probabilities after Bias Correction
probabilities_bc = compute_probabilities(
    index_bc, levels=params.intensity_thresholds
).round(2)
print(probabilities_bc)

# **Admin-2 level aggregation**

probs_district = compute_district_average(probabilities, area)

probs_bc_district = compute_district_average(probabilities_bc, area)

# Build single xarray with merged unbiased/biased probabilities
probs_by_district = merge_un_biased_probs(
    probs_district.squeeze("time"),
    probs_bc_district.squeeze("time"),
    params,
    period_name,
)

# **Dataframe formatting**

# Merge probabilities with triggers
probs_df, merged_df = merge_probabilities_triggers_dashboard(
    probs_by_district.drop_vars("time"), triggers_df, params, period_name
)

probs_df

merged_df
