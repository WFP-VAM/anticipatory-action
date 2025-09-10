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

from AA.helper_fns import (
    compute_district_average,
    merge_probabilities_triggers_dashboard,
    merge_un_biased_probs,
    read_forecasts,
    read_observations,
    read_triggers,
)
from AA.logging_utils import setup_aa_logging, log_array_info, log_processing_step
from config.params import S3_OPS_DATA_PATH, Params

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
    
    # Setup logging with conditional debug level
    logger = setup_aa_logging()
    logger.info("=== Starting AA operational run for %s, issue %d, index %s ===", 
                country, issue, index)
    logger.debug("Debug mode enabled - comprehensive array logging active")

    params = Params(
        iso=country,
        issue=issue,
        index=index,
        data_path=data_path,
        output_path=output_path,
    )
    
    # Log key parameters
    logger.debug("Parameters: monitoring_year=%d, calibration_year=%d", 
                params.monitoring_year, params.calibration_year)
    logger.debug("Season: start_month=%d, end_month=%d", 
                params.start_season, params.end_season)
    logger.debug("Index periods: min=%d, max=%d", 
                params.min_index_period, params.max_index_period)

    area = AnalysisArea.from_admin_boundaries(
        iso3=country.upper(),
        admin_level=2,
        resolution=0.25,
        datetime_range=f"1981-01-01/{params.monitoring_year + 1}-06-30",
    )
    
    # Log area information
    logger.debug("Analysis area: datetime_range=%s, resolution=%.2f, admin_level=2", 
                area.datetime_range, area.resolution)

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

    logger.info("Completed reading of forecasts for the whole %s country", params.iso)
    log_array_info(logger, "Loaded_Forecasts", forecasts)

    observations = read_observations(
        area,
        f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/obs/observations.zarr",
    )
    logger.info("Completed reading of observations for the whole %s country", params.iso)
    log_array_info(logger, "Loaded_Observations", observations)

    os.makedirs(
        f"{params.output_path}/data/{params.iso}/probs",
        exist_ok=True,
    )

    triggers_df = read_triggers(params)
    log_array_info(logger, "Loaded_Triggers", triggers_df)

    # Get accumulation periods (DJ, JF, FM, DJF, JFM...)
    accumulation_periods = get_accumulation_periods(
        forecasts,
        params.start_season,
        params.end_season,
        params.min_index_period,
        params.max_index_period,
    )
    logger.debug("Accumulation periods to process: %s", list(accumulation_periods.keys()))
    for period_name, period_months in accumulation_periods.items():
        logger.debug("Period %s: months %s", period_name, period_months)

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
    logger.info("Completed analysis for the required indexes over %s country", country)

    probs_df, merged_df = zip(*probs_merged_dataframes)

    # Log detailed information about merged_df before processing
    logger.debug("=== Analysis of merged_df collection ===")
    logger.debug("Number of merged_df elements: %d", len(merged_df))
    for i, df in enumerate(merged_df):
        logger.debug("merged_df[%d] shape: %s", i, df.shape)
        log_array_info(logger, f"merged_df[{i}]", df)
        # Log unique periods/indexes for each dataframe
        if hasattr(df, 'index') and 'index' in df.columns:
            unique_indexes = df['index'].unique() if 'index' in df.columns else []
            logger.debug("merged_df[%d] unique indexes: %s", i, list(unique_indexes))

    # Log final aggregation step
    logger.debug("Concatenating %d probability dataframes", len(probs_df))
    probs_dashboard = pd.concat(probs_df).drop_duplicates()
    log_array_info(logger, "Final_Probabilities_Dashboard", probs_dashboard)
    
    probs_dashboard.to_csv(
        f"{params.output_path}/data/{params.iso}/probs/aa_probabilities_{params.index}_{params.issue}.csv",
        index=False,
    )

    # Log detailed merged_df concatenation process
    logger.debug("=== Processing merged trigger dataframes ===")
    logger.debug("Concatenating %d merged trigger dataframes", len(merged_df))
    
    # Log shapes before concatenation
    total_rows_before = sum(df.shape[0] for df in merged_df)
    logger.debug("Total rows before concatenation: %d", total_rows_before)
    
    merged_db = pd.concat(merged_df)
    merged_db = merged_db.sort_values(["prob_ready", "prob_set"])

    # Check for duplicates and raise error if found
    duplicate_cols = list(merged_db.columns.difference(["prob_ready", "prob_set"]))
    duplicates_count = merged_db.duplicated(subset=duplicate_cols).sum()
    if duplicates_count > 0:
        logger.error("CRITICAL: %d duplicate rows found in merged_db!", duplicates_count)
        raise ValueError(f"Data integrity error: {duplicates_count} duplicate rows found in merged trigger data. "
                        f"This indicates a problem with the trigger merging process.")
    
    logger.debug("Duplicate check passed: No duplicate rows found in merged_db")
    
    # Use merge to append rows from triggers_df that don't exist in merged_db
    logger.debug("=== Appending missing triggers from triggers_df using merge ===")
    logger.debug("merged_db shape before append: %s", merged_db.shape)
    logger.debug("triggers_df shape: %s", triggers_df.shape)

    # Perform left merge to find rows in triggers_df that don't exist in merged_db
    merge_result = triggers_df.merge(
        merged_db[duplicate_cols], 
        on=duplicate_cols, 
        how='left', 
        indicator=True
    )

    # Get only rows that exist in triggers_df but not in merged_db
    new_rows_from_triggers = triggers_df[merge_result['_merge'] == 'left_only']

    # Append the new rows to merged_db
    if not new_rows_from_triggers.empty:
        merged_db = pd.concat([merged_db, new_rows_from_triggers], ignore_index=True)
        logger.debug("Added %d new rows from triggers_df", len(new_rows_from_triggers))
    else:
        logger.debug("No new rows to add from triggers_df - all triggers already present")

    logger.debug("merged_db final shape after append: %s", merged_db.shape)
    
    # Assert that final merged_db has same number of rows as original triggers_df
    assert len(merged_db) == len(triggers_df), (
        f"Data integrity error: Final merged_db has {len(merged_db)} rows but original "
        f"triggers_df has {len(triggers_df)} rows. Expected them to be equal."
    )
    logger.debug("Assertion passed: merged_db length (%d) equals triggers_df length (%d)", 
                len(merged_db), len(triggers_df))

    merged_db.sort_values(["district", "index", "category"]).to_csv(
        f"{params.output_path}/data/{params.iso}/probs/aa_probabilities_triggers_pilots.csv",
        index=False,
    )

    logger.info("Dashboard-formatted dataframe saved for %s", country)


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
    logger = logging.getLogger('aa_operational')
    logger.info("=== Starting probability computation for period %s ===", period_months)
    
    # Log input parameters and data
    logger.debug("Period months: %s, Index: %s, Monitoring year: %d", 
                period_months, params.index, params.monitoring_year)
    logger.debug("Historical anomaly period: %s to %s", 
                params.hist_anomaly_start, params.hist_anomaly_stop)
    logger.debug("Season: %d to %d, Intensity thresholds: %s", 
                params.start_season, params.end_season, params.intensity_thresholds)
    
    log_array_info(logger, "Input_Forecasts", forecasts)
    log_array_info(logger, "Input_Observations", observations)

    # Remove 1980 season to harmonize datasets between different indexes
    cutoff_date = datetime.date(1981, params.start_season, 1)
    logger.debug("Filtering data from %s onwards", cutoff_date)
    
    forecasts = forecasts.where(
        forecasts.time.dt.date >= cutoff_date, drop=True
    )
    observations = observations.where(
        observations.time.dt.date >= cutoff_date, drop=True
    )
    
    logger.debug("After 1981 filtering - Forecasts: %d time steps, Observations: %d time steps", 
                forecasts.time.size, observations.time.size)
    log_array_info(logger, "Filtered_Forecasts", forecasts)
    log_array_info(logger, "Filtered_Observations", observations)

    # Accumulation
    logger.info("Starting accumulation step")
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
    logger.info("Completed accumulation")
    log_array_info(logger, "Accumulation_Forecasts", accumulation_fc)
    log_array_info(logger, "Accumulation_Observations", accumulation_obs)

    # Anomaly
    logger.info("Starting gamma standardization (anomaly computation)")
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
    logger.info("Completed anomaly computation")
    log_array_info(logger, "Anomaly_Forecasts", anomaly_fc)
    log_array_info(logger, "Anomaly_Observations", anomaly_obs)

    # Bias correction
    logger.info("Starting bias correction")
    index_bc = run_bias_correction(
        anomaly_fc,
        anomaly_obs,
        params.monitoring_year,
        nearest_neighbours=8,
        enso=True,
    )
    logger.info("Completed bias correction")
    log_array_info(logger, "BiasCorreected_Index", index_bc)

    # Handle dryspell index sign flip
    if params.index == "dryspell":
        logger.debug("Applying dryspell sign correction (multiplying by -1)")
        anomaly_fc *= -1
        index_bc *= -1
        anomaly_obs *= -1
        log_array_info(logger, "SignCorrected_Anomaly_Forecasts", anomaly_fc)
        log_array_info(logger, "SignCorrected_BiasCorreected_Index", index_bc)

    # Probabilities without Bias Correction
    logger.info("Computing raw probabilities (no bias correction)")
    monitoring_year_fc = anomaly_fc.where(anomaly_fc.time.dt.year == params.monitoring_year, drop=True)
    logger.debug("Monitoring year data - time steps: %d", monitoring_year_fc.time.size)
    log_array_info(logger, "MonitoringYear_Data", monitoring_year_fc)
    
    probabilities = compute_probabilities(
        monitoring_year_fc,
        levels=params.intensity_thresholds,
    ).round(2)

    # Probabilities after Bias Correction
    logger.info("Computing bias-corrected probabilities")
    probabilities_bc = compute_probabilities(
        index_bc, levels=params.intensity_thresholds
    ).round(2)
    
    logger.info("Completed probabilities computation")
    log_array_info(logger, "Raw_Probabilities", probabilities)
    log_array_info(logger, "BiasCorreected_Probabilities", probabilities_bc)

    return probabilities, probabilities_bc


if __name__ == "__main__":
    # From AA repository:
    # $ pixi run python -m AA.operational MOZ 10 SPI --data-path "C:/path/to/data"

    run()
