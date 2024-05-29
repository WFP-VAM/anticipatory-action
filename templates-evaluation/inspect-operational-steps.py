# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

#
# This notebook is not used operationally or for any validation, its purpose is to have a clear understanding of the core functions of the AA workflow. The outputs and dimensions of each main step can thus be identified here. It can also be used to run the operational workflow for a very specific index or issue month. However, in order to better compare the outputs with the reference ones, some very simple analysis plots/tables will be added. 

import sys
sys.path.append('..')

# +
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

from hip.analysis.aoi.analysis_area import AnalysisArea
from hip.analysis.analyses.drought import (
    get_accumulation_periods,
    run_accumulation_index,
    run_gamma_standardization,
    run_bias_correction,
    compute_probabilities,
)
# -

params = Params(iso='MOZ', issue=12, index='SPI')

# +
area = AnalysisArea.from_admin_boundaries(
    iso3=params.iso,
    admin_level=2,
    resolution=0.25,
)

gdf = area.get_dataset([area.BASE_AREA_DATASET])
# -

forecasts = read_forecasts(
    area,
    params.issue,
    f"/s3/scratch/amine.barkaoui/aa/data/{params.iso.lower()}/zarr/{params.year}/{params.issue}/forecasts.zarr",
    #update=True,
)
forecasts

forecasts

observations = read_observations(
    area,
    f"/s3/scratch/amine.barkaoui/aa/data/{params.iso.lower()}/zarr/{params.year}/obs/observations.zarr",
)
observations

forecasts.time

observations.time

# +
# Read triggers file
#triggers_df = pd.read_csv(
#    f"data/{params.iso}/outputs/Plots/triggers.aa.python.spi.dryspell.2022.csv"
#)
# -

# Get accumulation periods (DJ, JF, FM, DJF, JFM...)
accumulation_periods = get_accumulation_periods(
    forecasts,
    params.start_season,
    params.end_season,
    params.min_index_period,
    params.max_index_period,
)

# Get single use case
period_name, period_months = list(accumulation_periods.items())[3]
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

anomaly_fc.where(anomaly_fc.time.dt.day == 1, drop=True).isel(latitude=0, longitude=0).plot.line()
anomaly_obs.isel(latitude=0, longitude=0).plot.line()

# Anomaly
anomaly_fc = run_gamma_standardization(
    accumulation_fc.load(), params.calibration_start, params.calibration_stop, members=True,
)
anomaly_obs = run_gamma_standardization(
    accumulation_obs.load(),
    params.calibration_start,
    params.calibration_stop,
    members=False,
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
    nearest_neighbours=8,
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
probs_district = aggregate_by_district(probabilities, params.gdf, params)
probs_bc_district = aggregate_by_district(probabilities_bc, params.gdf, params)

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
