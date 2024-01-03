import click
import logging

logging.basicConfig(level="INFO")

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import dask
import traceback
from dask.distributed import Client

import os
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import xskillscore as xss

from config.params import Params

from helper_fns import (
    read_forecasts_locally,
    read_observations_locally,
    aggregate_by_district,
)

from hip.analysis.analyses.drought import (
    get_accumulation_periods,
    run_accumulation_index,
    run_gamma_standardization,
    run_bias_correction,
    compute_probabilities,
    concat_obs_levels,
)


@click.command()
@click.argument("country", required=True, type=str)
@click.argument("index", default="SPI")
def run(country, index):
    # End to end workflow for a country using pre-stored ECMWF forecasts and CHIRPS

    # if dashboard.link set to default value and running behind hub, make dashboard link go via proxy
    domain = ""
    if (
        dask.config.get("distributed.dashboard.link")
        == "{scheme}://{host}:{port}/status"
    ):
        jup_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")
        if jup_prefix is not None:
            jup_prefix = jup_prefix.rstrip("/")
            dask.config.set(
                {"distributed.dashboard.link": f"{jup_prefix}/proxy/{{port}}/status"}
            )
            domain = "https://jupyter.earthobservation.vam.wfp.org"

    client = Client()
    logging.info("+++++++++++++")
    logging.info(f"Dask dashboard: {domain}{client.dashboard_link}")
    logging.info("+++++++++++++")

    params = Params(iso=country, index=index)

    observations = read_observations_locally(f"AA/data/{params.iso}/chirps")
    logging.info(
        f"Completed reading of observations for the whole {params.iso} country"
    )

    gdf = gpd.read_file(
        f"AA/data/{params.iso}/shapefiles/moz_admbnda_2019_SHP/moz_admbnda_adm2_2019.shp"
    )

    fbf_roc_issues = [
        run_issue_verification(
            observations,
            issue,
            params,
            gdf,
        )
        for issue in params.issue
    ]
    logging.info(
        f"Completed analytical process for {params.index.upper()} over {country} country"
    )

    fbf_roc = pd.concat(fbf_roc_issues)
    fbf_roc.to_csv(
        f"AA/data/{params.iso}/outputs/Districts_FbF/{params.index}/fbf.districts.roc.{params.index}.{params.year}.txt",
        index=False,
    )

    logging.info(f"FbF dataframe saved for {country}")

    client.close()


def run_issue_verification(observations, issue, params, gdf):
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

    forecasts = read_forecasts_locally(
        f"AA/data/{params.iso}/forecast/Moz_SAB_tp_ecmwf_{issue}/*.nc"
    )
    forecasts = forecasts.where(
        forecasts.time < np.datetime64(f"{params.year}-07-01T12:00:00.000000000"),
        drop=True,
    )
    logging.info(f"Completed reading of forecasts for the issue month {issue}")

    # Get accumulation periods (DJ, JF, FM, DJF, JFM...)
    accumulation_periods = get_accumulation_periods(
        forecasts,
        params.start_season,
        params.end_season,
        params.min_index_period,
        params.max_index_period,
    )

    # TODO chunk rfh datasets on lat and lon dim only

    fbf_indexes = [
        # TODO handle distributed computation
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

    logging.info(f"FbF ROC verification by district for the issue month {issue} done")

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

    auc, auc_bc = evaluate_forecast_probabilities(
        probs,
        probs_bc,
        obs_bool,
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
        forecasts, params.aggregate, period_months, forecasts=True
    )
    accumulation_obs = run_accumulation_index(
        observations, params.aggregate, period_months
    )
    logging.info(f"Completed accumulation")

    # Remove potential inconsistent observations
    accumulation_obs = accumulation_obs.sel(
        time=slice(datetime.date(1979, 1, 1), datetime.date(params.year - 1, 12, 31))
    )

    # Anomaly
    anomaly_fc = run_gamma_standardization(
        accumulation_fc, params.calibration_start, params.calibration_stop
    )
    anomaly_obs = run_gamma_standardization(
        accumulation_obs,
        params.calibration_start,
        params.calibration_stop,
        members=False,
    )
    logging.info(f"Completed anomaly")

    # Probabilities without Bias Correction
    probabilities = (
        anomaly_fc.groupby(anomaly_fc.time.dt.year)
        .apply(
            compute_probabilities,
            levels=params.intensity_thresholds,
        )
        .round(2)
    )
    probabilities = probabilities.where(probabilities.year < params.year, drop=True)

    # Convert obs-based SPIs to booleans
    levels_obs = concat_obs_levels(
        anomaly_obs,
        levels=params.intensity_thresholds,
    )
    levels_obs = levels_obs.sel(
        latitude=probabilities.latitude,
        longitude=probabilities.longitude,
    )
    levels_obs = levels_obs.sel(year=probabilities.year)

    # Bias correction
    anomaly_bc = xr.concat(
        [
            run_bias_correction(
                anomaly_fc,
                anomaly_obs,
                params.end_season,
                year,
                int(issue),
                enso=True,
            )
            for year in np.unique(anomaly_fc.time.dt.year)
        ],
        dim="time",
    )

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
        probabilities_bc.year < params.year, drop=True
    )

    return probabilities, probabilities_bc, anomaly_obs, levels_obs


def evaluate_forecast_probabilities(probabilities, probabilities_bc, obs_bool):
    """
    Calculate ROC scores of probabilities computed both with and without bias correction

    Args:
        probabilities: xarray.Dataset, raw probabilities at the pixel level for specified period and issue month
        probabilities_bc: xarray.Dataset, bias-corrected probabilities at the pixel level for specified period and issue month
        levels_obs: xarray.Dataset, categorical observations at the pixel level for specified index
    Returns:
        auc: xarray.Dataset, roc scores related to raw probabilities at the pixel level
        auc_bc: xarray.Dataset, roc scores related to bias-corrected probabilities at the pixel level
    """
    # Compute AUC without BC
    auc = xss.roc(obs_bool.precip, probabilities.tp, dim="year", return_results="area")
    # Compute AUC with BC
    auc_bc = xss.roc(
        obs_bool.precip, probabilities_bc.scen, dim="year", return_results="area"
    )

    # set AUC as NaN where no rain in either chirps or forecasts (replicate R method)
    auc = auc.where(
        (obs_bool.precip.sum("year") != 0) & (probabilities.tp.sum("year") != 0), np.nan
    )
    auc_bc = auc_bc.where(
        (obs_bool.precip.sum("year") != 0) & (probabilities_bc.scen.sum("year") != 0),
        np.nan,
    )

    return auc, auc_bc


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
        f"AA/data/{params.iso}/outputs/zarr/obs/2022/{params.index.upper()} {period_name}/observations.zarr",
        mode="w",
    )

    probs_district.to_zarr(
        f"AA/data/{params.iso}/outputs/zarr/2022/{issue}/{params.index.upper()} {period_name}/probabilities.zarr",
        mode="w",
    )

    probs_bc_district.to_zarr(
        f"AA/data/{params.iso}/outputs/zarr/2022/{issue}/{params.index.upper()} {period_name}/probabilities_bc.zarr",
        mode="w",
    )


if __name__ == "__main__":
    # From AA repository:
    # $ python analytical.py MOZ SPI

    run()
