import datetime
import logging
import os
import warnings

import click
import numpy as np
import pandas as pd
from hip.analysis.analyses.drought import (
    compute_probabilities,
    get_accumulation_periods,
    run_accumulation_index,
    run_bias_correction,
    run_gamma_standardization,
)
from hip.analysis.aoi.analysis_area import AnalysisArea

from AA.src.params import S3_OPS_DATA_PATH, Params
from AA.src.utils import (
    compute_district_average,
    merge_probabilities_triggers_dashboard,
    merge_un_biased_probs,
    read_forecasts,
    read_observations,
    read_triggers,
)

logging.basicConfig(level="INFO", force=True)

warnings.simplefilter(action="ignore", category=FutureWarning)


@click.command()
@click.argument("country", required=True, type=str)
@click.argument("issue", required=True, type=int)
@click.argument("index", default="SPI")
@click.option(
    "--data-path",
    required=True,
    type=str,
    default=S3_OPS_DATA_PATH,
    help="Root directory for data files.",
)
@click.option(
    "--output-path",
    required=False,
    type=str,
    default=S3_OPS_DATA_PATH,
    help="Root directory for output files. Defaults to data-path if not provided.",
)
def run(country, issue, index, data_path, output_path):
    # End to end workflow for a country using pre-stored ECMWF forecasts and CHIRPS

    params = Params(
        iso=country,
        issue=issue,
        index=index,
        data_path=data_path,
        output_path=output_path,
    )

    area = AnalysisArea.from_admin_boundaries(
        iso3=country.upper(),
        admin_level=2,
        resolution=0.25,
        datetime_range=f"1981-01-01/{params.monitoring_year + 1}-06-30",
    )

    forecasts = read_forecasts(
        area,
        issue,
        f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/{str(issue).zfill(2)}/forecasts.zarr",
    )

    # Check if the forecast date is in the time coordinate
    forecast_date = np.datetime64(
        datetime.datetime(params.monitoring_year, params.issue, 1), "ns"
    )
    if forecast_date not in forecasts.time.values:
        raise ValueError(
            "Forecast missing from dataset — it might not have been released yet. Try again later."
        )

    logging.info("Completed reading of forecasts for the whole %s country", params.iso)

    observations = read_observations(
        area,
        f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/obs/observations.zarr",
    )
    logging.info(
        "Completed reading of observations for the whole %s country", params.iso
    )

    os.makedirs(
        f"{params.output_path}/data/{params.iso}/probs",
        exist_ok=True,
    )

    triggers_df = read_triggers(params)

    # Get accumulation periods (DJ, JF, FM, DJF, JFM...)
    accumulation_periods = get_accumulation_periods(
        forecasts,
        params.start_season,
        params.end_season,
        params.min_index_period,
        params.max_index_period,
    )

    # Compute probabilities for each accumulation period
    probs_merged_dataframes = [
        run_full_index_pipeline(
            forecasts,
            observations,
            params,
            triggers_df,
            area,
            period_name,
            period_months,
        )
        for period_name, period_months in accumulation_periods.items()
    ]
    logging.info("Completed analysis for the required indexes over %s country", country)

    probs_df, merged_df = zip(*probs_merged_dataframes)

    probs_dashboard = pd.concat(probs_df).drop_duplicates()
    probs_dashboard.to_csv(
        f"{params.output_path}/data/{params.iso}/probs/aa_probabilities_{params.index}_{params.issue}.csv",
        index=False,
    )

    merged_db = pd.concat(merged_df)
    merged_db = merged_db.sort_values(["prob_ready", "prob_set"])

    # Check for duplicates and raise error if found
    duplicate_cols = list(merged_db.columns.difference(["prob_ready", "prob_set"]))
    duplicates_count = merged_db.duplicated(subset=duplicate_cols).sum()
    if duplicates_count > 0:
        raise ValueError(
            f"Data integrity error: {duplicates_count} duplicate rows found in merged trigger data. "
            f"This indicates a problem with the trigger merging process."
        )

    # Perform left merge to find rows in triggers_df that don't exist in merged_db
    merge_result = triggers_df.merge(
        merged_db[duplicate_cols], on=duplicate_cols, how="left", indicator=True
    )

    # Get only rows that exist in triggers_df but not in merged_db
    new_rows_from_triggers = triggers_df[merge_result["_merge"] == "left_only"]

    # Append the new rows to merged_db
    if not new_rows_from_triggers.empty:
        merged_db = pd.concat([merged_db, new_rows_from_triggers], ignore_index=True)

    # Assert that final merged_db has same number of rows as original triggers_df
    assert len(merged_db) == len(triggers_df), (
        f"Data integrity error: Final merged_db has {len(merged_db)} rows but original "
        f"triggers_df has {len(triggers_df)} rows. Expected them to be equal."
    )

    merged_db.sort_values(["district", "index", "category"]).to_csv(
        f"{params.output_path}/data/{params.iso}/probs/aa_probabilities_triggers_pilots.csv",
        index=False,
    )

    logging.info("Dashboard-formatted dataframe saved for %s", country)


def run_full_index_pipeline(
    forecasts, observations, params, triggers, area, period_name, period_months
):
    """
    Run operational pipeline for single index (period)

    Args:
        forecasts: xarray.Dataset, rainfall forecasts dataset
        observations: xarray.Dataset, rainfall observations dataset
        params: Params, parameters class
        triggers: pd.DataFrame, selected triggers (output of triggers.py)
        area: hip.analysis.AnalysisArea object with aoi information
        period_name: str, name of index period (eg "ON")
        period_months: tuple, months of index period (eg (10, 11))
    Returns:
        probs_df: pandas.DataFrame, probabilities (bc or not depending on analytical output) for all districts
        merged_df: xarray.Dataset, probabilities merged with selected triggers
    """
    probabilities, probabilities_bc = run_aa_probabilities(
        forecasts, observations, params, period_months
    )

    # Aggregate by district
    probs_district = compute_district_average(probabilities, area)
    probs_bc_district = compute_district_average(probabilities_bc, area)

    # Build single xarray with merged unbiased/biased probabilities
    probs_by_district = merge_un_biased_probs(
        probs_district, probs_bc_district, params, period_name
    )

    # Merge probabilities with triggers
    probs_df, merged_df = merge_probabilities_triggers_dashboard(
        probs_by_district, triggers, params, period_name
    )

    logging.info(
        "Completed probabilities computation by district for the %s %s index",
        params.index.upper(),
        period_name,
    )

    return probs_df, merged_df


def run_aa_probabilities(forecasts, observations, params, period_months):
    """
    Compute probabilities based on recent forecasts for operational routine

    Args:
        forecasts: xarray.Dataset, rainfall forecasts dataset
        observations: xarray.Dataset, rainfall observations dataset
        params: Params, parameters class
        period_months: tuple, months of index period (eg (10, 11))
    Returns:
        probabilities: xarray.Dataset, raw probabilities for specified period
        probabilities_bc: xarray.Dataset, bias-corrected probabilities for specified period
    """
    # Remove 1980 season to harmonize datasets between different indexes
    forecasts = forecasts.where(
        forecasts.time.dt.date >= datetime.date(1981, params.start_season, 1), drop=True
    )
    observations = observations.where(
        observations.time.dt.date >= datetime.date(1981, params.start_season, 1),
        drop=True,
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
    logging.info("Completed accumulation")

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
    logging.info("Completed anomaly")

    # Bias correction
    index_bc = run_bias_correction(
        anomaly_fc,
        anomaly_obs,
        params.monitoring_year,
        nearest_neighbours=8,
        enso=True,
    )
    logging.info("Completed bias correction")

    if params.index == "dryspell":
        anomaly_fc *= -1
        index_bc *= -1
        anomaly_obs *= -1

    # Probabilities without Bias Correction
    probabilities = compute_probabilities(
        anomaly_fc.where(anomaly_fc.time.dt.year == params.monitoring_year, drop=True),
        levels=params.intensity_thresholds,
    ).round(2)

    # Probabilities after Bias Correction
    probabilities_bc = compute_probabilities(
        index_bc, levels=params.intensity_thresholds
    ).round(2)
    logging.info("Completed probabilities")

    return probabilities, probabilities_bc


if __name__ == "__main__":
    # From AA repository:
    # $ pixi run python -m AA.operational MOZ 10 SPI --data-path "C:/path/to/data"

    run()
