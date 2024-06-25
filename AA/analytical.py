import logging

import click

logging.basicConfig(level="INFO")

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import datetime
import os
import traceback

import dask
import numpy as np
import pandas as pd
import xarray as xr
from config.params import Params
from hip.analysis.analyses.drought import (
    compute_probabilities,
    concat_obs_levels,
    get_accumulation_periods,
    run_accumulation_index,
    run_bias_correction,
    run_gamma_standardization,
)
from hip.analysis.aoi.analysis_area import AnalysisArea
from hip.analysis.ops._statistics import evaluate_roc_forecasts

from AA.helper_fns import aggregate_by_district, read_forecasts, read_observations


@click.command()
@click.argument("country", required=True, type=str)
@click.argument("index", default="SPI")
def run(country, index):
    # End to end workflow for a country using ECMWF forecasts and CHIRPS from HDC
    params = Params(iso=country, index=index)

    area = AnalysisArea.from_admin_boundaries(
        iso3=country.upper(),
        admin_level=2,
        resolution=0.25,
        datetime_range=f"1981-01-01/{params.calibration_year}-06-30",
    )

    gdf = area.get_dataset([area.BASE_AREA_DATASET])

    observations = read_observations(
        area,
        f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/obs/observations.zarr",
    )
    logging.info(
        f"Completed reading of observations for the whole {params.iso} country"
    )

    # Create directory for ROC scores df per issue month in case it doesn't exist
    os.makedirs(
        f"{params.data_path}/data/{params.iso}/auc/split_by_issue",
        exist_ok=True,
    )

    # Define empty list for each issue month's ROC score dataframe
    fbf_roc_issues = []

    for issue in params.issue_months:

        forecasts = read_forecasts(
            area,
            issue,
            f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/{issue}/forecasts.zarr",
        )
        logging.info(f"Completed reading of forecasts for the issue month {issue}")

        fbf_roc_issues.append(
            run_issue_verification(
                forecasts,
                observations,
                issue,
                params,
                gdf,
            )
        )
        logging.info(
            f"Completed analytical process for {params.index.upper()} over {country} country"
        )

    fbf_roc = pd.concat(fbf_roc_issues)
    fbf_roc.to_csv(
        f"{params.data_path}/data/{params.iso}/auc/fbf.districts.roc.{params.index}.{params.calibration_year}.csv",
        index=False,
    )

    logging.info(f"FbF dataframe saved for {country}")


def run_issue_verification(forecasts, observations, issue, params, gdf):
    """
    Run analytical / verification pipeline for one issue month

    Args:
        observations: xarray.Dataset, rainfall observations dataset
        issue: str, issue month of forecasts to analyse
        params: Params, parameters class
        gdf: geopandas.GeoDataFrame, shapefile including admin2 levels
    Returns:
        fbf_issue: pandas.DataFrame, dataframe with roc scores for all indexes, districts, categories and a specified issue month
    """

    if os.path.exists(
        f"{params.data_path}/data/{params.iso}/auc/split_by_issue/fbf.districts.roc.{params.index}.{params.calibration_year}.{issue}.csv"
    ):

        logging.info(
            f"FbF ROC verification by district for the issue month {issue} read from disk"
        )

        return pd.read_csv(
            f"{params.data_path}/data/{params.iso}/auc/split_by_issue/fbf.districts.roc.{params.index}.{params.calibration_year}.{issue}.csv"
        )

    else:

        # Get accumulation periods (DJ, JF, FM, DJF, JFM...)
        accumulation_periods = get_accumulation_periods(
            forecasts,
            params.start_season,
            params.end_season,
            params.min_index_period,
            params.max_index_period,
        )

        fbf_indexes = [
            verify_index_across_districts(
                forecasts,
                observations,
                params,
                gdf,
                period_name,
                period_months,
                issue,
            )
            for period_name, period_months in accumulation_periods.items()
        ]

        fbf_issue = pd.concat(fbf_indexes)
        fbf_issue["issue"] = int(issue)

        fbf_issue.to_csv(
            f"{params.data_path}/data/{params.iso}/auc/split_by_issue/fbf.districts.roc.{params.index}.{params.calibration_year}.{issue}.csv",
            index=False,
        )

        logging.info(
            f"FbF ROC verification by district for the issue month {issue} done"
        )

        return fbf_issue


def verify_index_across_districts(
    forecasts,
    observations,
    params,
    gdf,
    period_name,
    period_months,
    issue,
):
    """
    Run analytical / verification pipeline for a single issue month and a single index (period)

    Args:
        forecasts: xarray.Dataset, rainfall forecasts dataset for specific issue month
        observations: xarray.Dataset, rainfall observations dataset
        params: Params, parameters class
        gdf: geopandas.GeoDataFrame, shapefile including admin2 levels
        period_name: str, name of index period (eg "ON")
        period_months: tuple, months of index period (eg (10, 11))
    Returns:
        fbf_issue_df: pandas.DataFrame, dataframe with roc scores for all districts, categories and specified issue month / period
    """

    probs, probs_bc, obs_values, obs_bool = calculate_forecast_probabilities(
        forecasts,
        observations,
        params,
        period_months,
        issue,
    )

    auc, auc_bc = evaluate_roc_forecasts(
        obs_bool,
        probs,
        probs_bc,
    )

    if params.save_zarr:
        save_districts_results(
            obs_values,
            probs,
            probs_bc,
            issue,
            period_name,
            params,
            gdf,
        )

    # Aggregate by district
    auc_district = aggregate_by_district(auc, gdf, params)
    auc_bc_district = aggregate_by_district(auc_bc, gdf, params)

    # Choose W/ or W/OUT BC based on AUROC
    fbf_index_df = get_verification_df(
        auc_district,
        auc_bc_district,
    )
    fbf_index_df["Index"] = f"{params.index.upper()} {period_name}"

    logging.info(
        f"Completed FbF ROC computation by district for the {params.index.upper()} {period_name} index"
    )

    return fbf_index_df


def calculate_forecast_probabilities(
    forecasts,
    observations,
    params,
    period_months,
    issue,
):
    """
    Calculate probabilities with and without bias correction, and extract numerical and categorical observations

    Args:
        forecasts: xarray.Dataset, rainfall forecasts dataset for specific issue month
        observations: xarray.Dataset, rainfall observations dataset
        params: Params, parameters class
        period_months: tuple, months of index period (eg (10, 11))
        issue: str, issue month of forecasts to analyse
    Returns:
        probabilities: xarray.Dataset, raw probabilities at the pixel level for specified period and issue month
        probabilities_bc: xarray.Dataset, bias-corrected probabilities at the pixel level for specified period and issue month
        anomaly_obs: xarray.Dataset, numerical observations at the pixel level for specified period
        levels_obs: xarray.Dataset, categorical observations at the pixel level for specified index
    """

    # Remove 1980 season to harmonize observations between different indexes
    if int(issue) >= params.end_season:
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
    logging.info(f"Completed accumulation")

    # Remove potential inconsistent observations
    accumulation_obs = accumulation_obs.sel(
        time=slice(
            datetime.date(1979, 1, 1),
            datetime.date(params.calibration_year - 1, 12, 31),
        )
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
    logging.info(f"Completed anomaly")

    # Bias correction
    anomaly_bc = xr.concat(
        [
            run_bias_correction(
                anomaly_fc,
                anomaly_obs,
                params.end_season,
                year,
                int(issue),
                nearest_neighbours=8,
                enso=True,
            )
            for year in np.unique(anomaly_fc.time.dt.year)
        ],
        dim="time",
    )
    logging.info(f"Completed bias correction")

    if params.index == "dryspell":
        anomaly_fc *= -1
        anomaly_bc *= -1
        anomaly_obs *= -1

    # Probabilities without Bias Correction
    probabilities = (
        anomaly_fc.groupby(anomaly_fc.time.dt.year)
        .apply(
            compute_probabilities,
            levels=params.intensity_thresholds,
        )
        .round(2)
    )
    probabilities = probabilities.where(
        probabilities.year < params.calibration_year, drop=True
    )
    logging.info(f"Completed probabilities")

    # Convert obs-based SPIs to booleans
    levels_obs = concat_obs_levels(
        anomaly_obs,
        levels=params.intensity_thresholds,
    )
    levels_obs = levels_obs.sel(
        latitude=probabilities.latitude,
        longitude=probabilities.longitude,
    )
    # Harmonize coordinates with forecasts
    levels_obs["time"] = levels_obs.time.dt.year.values
    levels_obs = levels_obs.rename({"time": "year"})
    levels_obs = levels_obs.sel(year=probabilities.year)
    logging.info(f"Completed categorical observations")

    # Probabilities after Bias Correction
    probabilities_bc = (
        anomaly_bc.groupby(anomaly_bc.time.dt.year)
        .apply(
            compute_probabilities,
            levels=params.intensity_thresholds,
        )
        .round(2)
    )
    probabilities_bc = probabilities_bc.where(
        probabilities_bc.year < params.calibration_year, drop=True
    )
    logging.info(f"Completed probabilities with bias correction")

    return probabilities, probabilities_bc, anomaly_obs, levels_obs


def get_verification_df(auc, auc_bc):
    # Convert AUC datasets to dataframes
    auc_df = auc.to_dataframe().drop("spatial_ref", axis=1)
    auc_df.columns = ["AUC"]

    auc_bc_df = auc_bc.to_dataframe().drop("spatial_ref", axis=1)
    auc_bc_df.columns = ["AUC_BC"]

    # Select method based on AUROC score
    auc_merged_df = auc_df.join(auc_bc_df)

    auc_merged_df["AUC_best"] = auc_merged_df.max(axis=1)
    auc_merged_df["BC"] = [
        int(row.AUC_best == row.AUC_BC) for i, row in auc_merged_df.iterrows()
    ]

    auc_merged_df = auc_merged_df.drop(["AUC", "AUC_BC"], axis=1)

    return auc_merged_df.reset_index()


# Save obs and probs for each issue / index in zarr
def save_districts_results(
    observations,
    probabilities,
    probabilities_bc,
    issue,
    period_name,
    params,
    gdf,
):
    obs_district = aggregate_by_district(observations, gdf, params)
    probs_district = aggregate_by_district(probabilities, gdf, params)
    probs_bc_district = aggregate_by_district(probabilities_bc, gdf, params)

    obs_district.to_zarr(
        f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/obs/{params.index} {period_name}/observations.zarr",
        mode="w",
    )

    probs_district.to_zarr(
        f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/{issue}/{params.index} {period_name}/probabilities.zarr",
        mode="w",
    )

    probs_bc_district.to_zarr(
        f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/{issue}/{params.index} {period_name}/probabilities_bc.zarr",
        mode="w",
    )


if __name__ == "__main__":
    # From AA repository:
    # $ python analytical.py MOZ SPI

    run()
