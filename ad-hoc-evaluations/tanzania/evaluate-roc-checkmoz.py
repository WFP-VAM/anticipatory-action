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

# This notebook aims to evaluate the forecast skill (ROC score) for the JFM period using forecasts issued in January for Mozambique. It uses HDC datasets so we can compare the ROC scores with the ones obtained using local forecasts and observations datasets.

# +
# %cd ../../

import datetime
import logging

import numpy as np
import pandas as pd
import xskillscore as xss
from hip.analysis.analyses.drought import (
    run_accumulation_index,
    run_gamma_standardization,
)
from hip.analysis.aoi.analysis_area import AnalysisArea

from AA.analytical import calculate_forecast_probabilities
from AA.helper_fns import compute_district_average
from config.params import Params

# +
issue = "11"

period_months = (1, 2, 3)  # JFM

params = Params(iso="MOZ", index="SPI")
# -

DATES = "1981-01-01/2023-12-31"
area = AnalysisArea.from_admin_boundaries(
    iso3="MOZ", admin_level=2, resolution=0.25, datetime_range=DATES
)
area.geobox

gdf = area.get_dataset([area.BASE_AREA_DATASET])
gdf.columns = ["geometry", "code", "ADM2_PT"]

forecasts = area.get_dataset(
    ["ECMWF", f"RFH_FORECASTS_SEAS5_ISSUE{int(issue)}_DAILY"],
    load_config={
        "gridded_load_kwargs": {
            "resampling": "bilinear",
        }
    },
)
forecasts = forecasts.where(
    forecasts.time < np.datetime64(f"{params.year}-07-01T12:00:00.000000000"),
    drop=True,
).to_dataset()
forecasts.attrs["nodata"] = np.nan
forecasts.tp.attrs["nodata"] = np.nan

forecasts = forecasts.isel(time=slice(2547, None)).load()

observations = (
    area.get_dataset(
        ["CHIRPS", "RFH_DAILY"],
        load_config={
            "gridded_load_kwargs": {
                "resampling": "bilinear",
            }
        },
    )
    .rename("precip")
    .to_dataset()
)
observations.attrs["nodata"] = observations.precip.nodata

observations.load()

forecasts.isel(longitude=0, latitude=0, ensemble=0).groupby(
    "time.year"
).mean().tp.plot.line()  # , time=slice(3000,3100)

# +

# Remove 1980 season to harmonize observations between different indexes
if int(issue) >= params.end_season:
    observations = observations.where(
        observations.time.dt.date >= datetime.date(1981, 10, 1), drop=True
    )

# Accumulation
accumulation_fc = run_accumulation_index(
    forecasts, params.aggregate, period_months, forecasts=True
)
accumulation_obs = run_accumulation_index(observations, params.aggregate, period_months)
logging.info("Completed accumulation")
# -

accumulation_fc.isel(longitude=0, latitude=0, ensemble=0).tp.plot.line()

# +

# Remove potential inconsistent observations
accumulation_obs = accumulation_obs.sel(
    time=slice(datetime.date(1979, 1, 1), datetime.date(params.year - 1, 12, 31))
)

# TODO handle attributes in accumulation
accumulation_fc.tp.attrs = forecasts.tp.attrs
accumulation_obs.precip.attrs = observations.precip.attrs

# Anomaly
anomaly_fc = run_gamma_standardization(
    accumulation_fc,
    params.calibration_start,
    params.calibration_stop,
    members=True,
)
anomaly_obs = run_gamma_standardization(
    accumulation_obs,
    params.calibration_start,
    params.calibration_stop,
)
logging.info("Completed anomaly")
# -

anomaly_fc.where(anomaly_fc.time.dt.day == 1, drop=True).isel(
    longitude=0, latitude=0
).tp.plot.line()

(
    anomaly_fc.where(anomaly_fc.time.dt.day == 1, drop=True).isel(
        longitude=0, latitude=0
    )
    < -1000
).tp.plot.line()

probs, probs_bc, obs_values, obs_bool = calculate_forecast_probabilities(
    forecasts,
    observations,
    params,
    period_months,
    issue,
)

probs.isel(longitude=0, latitude=0, category=0).tp.plot.line()


# +
def evaluate_roc_forecasts(observations, *forecasts, dim="year"):
    """
    Calculate the area under the ROC curve scores of forecasts dataset(s)

    Args:
        observations: xarray.DataArray, categorical observations at the pixel level for specific index
        *forecasts: xarray.DataArray object(s) containing forecasts as probabilities to evaluate for specific index and issue month
        dim: str or list, dimension(s) over which to compute the contingency table
    Returns:
        auc: tuple of the same length as *forecasts of xarray.DataArray, reduced by dimensions dim, containing area under ROC curve
    """

    aucs = []

    # Compute AUC for each forecasts xarray object
    for fc in forecasts:
        auc = xss.roc(
            observations.sel(year=fc.year), fc, dim=dim, return_results="area"
        )
        # set AUC as NaN where no rain in either chirps or forecasts (replicate R method)
        auc = auc.where((observations.sum(dim) != 0) & (fc.sum(dim) != 0), np.nan)
        aucs.append(auc)

    return tuple(aucs)


# -

auc, auc_bc = evaluate_roc_forecasts(
    obs_bool.precip,
    probs.tp,
    probs_bc.scen,
)

auc.isel(longitude=0, latitude=0, category=0)

auc.to_zarr("ad-hoc-evaluations/tanzania/outputs/moz/moz_auc_jfm_issue11.zarr")

auc_bc.to_zarr("ad-hoc-evaluations/tanzania/outputs/moz/moz_auc_bc_jfm_issue11.zarr")

# Aggregate by district
auc_district = compute_district_average(auc, gdf, params)
auc_bc_district = compute_district_average(auc_bc, gdf, params)

# Read ref roc scores
moz_ref = pd.read_csv(
    "ad-hoc-evaluations/tanzania/outputs/moz/fbf.districts.roc.spi.2022.txt"
)

# Compare
moz_ref = moz_ref.loc[moz_ref.district.isin(auc_district.district.values)]
moz_ref = moz_ref.loc[(moz_ref.Index == "SPI JFM") & (moz_ref.issue == 11)]
auc_hdc = np.array(
    [
        auc_bc_district.sel(district=r.district, category=r.category).values
        if r.BC
        else auc_district.sel(district=r.district, category=r.category).values
        for _, r in moz_ref.iterrows()
    ]
).astype(float)
moz_ref["auc_hdc"] = auc_hdc
moz_ref["diff (hdc - ref)"] = moz_ref.auc_hdc - moz_ref.AUC_best

moz_ref

moz_ref["diff (hdc - ref)"].hist(bins=15)
