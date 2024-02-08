import click
import logging

logging.basicConfig(level="INFO")

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import dask
import traceback
from dask.distributed import Client

import datetime
import pandas as pd
import geopandas as gpd

from config.params import Params

from AA.helper_fns import (
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


@click.command()
@click.argument("country", required=True, type=str)
@click.argument("issue", required=True, type=int)
@click.argument("index", default="SPI")
def run(country, issue, index):
    # End to end workflow for a country using pre-stored ECMWF forecasts and CHIRPS

    client = Client()

    params = Params(iso=country, issue=issue, index=index)

    # TODO when local reading available: area, forecasts, observations = set_up_analysis_area(params)

    forecasts = read_forecasts_locally(
        f"data/{params.iso}/forecast/{params.iso}_SAB_tp_ecmwf_{str(params.issue).zfill(2)}/*.nc"
    )
    logging.info(f"Completed reading of forecasts for the whole {params.iso} country")

    observations = read_observations_locally(f"data/{params.iso}/chirps")
    logging.info(
        f"Completed reading of observations for the whole {params.iso} country"
    )

    triggers_df = pd.read_csv(
        f"data/{params.iso}/outputs/Plots/triggers.aa.python.spi.dryspell.2022.csv"
    )

    if params.iso == "MOZ":
        gdf = gpd.read_file(
            f"data/{params.iso}/shapefiles/moz_admbnda_2019_SHP/moz_admbnda_adm2_2019.shp"
        )
    else:
        gdf = gpd.read_file(
            f"data/{params.iso}/shapefiles/global_adm2_wfpGE_UN_final/global_adm2_wfpGE_UN_final.shp"
        )
        gdf = gdf.loc[gdf.iso3 == params.iso]

    # Get accumulation periods (DJ, JF, FM, DJF, JFM...)
    accumulation_periods = get_accumulation_periods(
        forecasts,
        params.start_season,
        params.end_season,
        params.min_index_period,
        params.max_index_period,
    )

    # TODO chunk rfh datasets on lat and lon dim only

    probs_merged_dataframes = [
        # TODO handle distributed computation
        run_full_index_pipeline(
            forecasts,
            observations,
            params,
            triggers_df,
            gdf,
            period_name,
            period_months,
        )
        for period_name, period_months in accumulation_periods.items()
    ]
    logging.info(f"Completed analysis for the required indexes over {country} country")

    probs_df, merged_df = zip(*probs_merged_dataframes)

    probs_dashboard = pd.concat(probs_df)
    probs_dashboard.to_csv(
        f"data/{params.iso}/outputs/Fbf_Pilot_MockUp/SPI_probabilities_season_{params.issue}.csv",
        index=False,
    )

    merged_dashboard = pd.concat(merged_df)
    merged_dashboard.to_csv(
        f"data/{params.iso}/outputs/Fbf_Pilot_MockUp/SPI_probabilities_season_triggers_{params.issue}.csv",
        index=False,
    )

    logging.info(f"Dashboard-formatted dataframe saved for {country}")

    client.close()


def run_full_index_pipeline(
    forecasts, observations, params, triggers, gdf, period_name, period_months
):
    """
    Run operational pipeline for single index (period)

    Args:
        forecasts: xarray.Dataset, rainfall forecasts dataset
        observations: xarray.Dataset, rainfall observations dataset
        params: Params, parameters class
        triggers: pd.DataFrame, selected triggers (output of triggers.py)
        gdf: geopandas.GeoDataFrame, shapefile including admin2 levels
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
    probs_district = aggregate_by_district(probabilities, gdf, params)
    probs_bc_district = aggregate_by_district(probabilities_bc, gdf, params)

    # Build single xarray with merged unbiased/biased probabilities
    probs_by_district = merge_un_biased_probs(
        probs_district, probs_bc_district, params, period_name
    )

    # Merge probabilities with triggers
    probs_df, merged_df = merge_probabilities_triggers_dashboard(
        probs_by_district, triggers, params, period_name
    )

    logging.info(
        f"Completed probabilities computation by district for the {params.index.upper()} {period_name} index"
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
    logging.info(f"Completed accumulation")

    # Remove inconsistent observations
    accumulation_obs = accumulation_obs.sel(
        time=slice(datetime.date(1979, 1, 1), datetime.date(params.year - 1, 12, 31))
    )

    # Anomaly
    anomaly_fc = run_gamma_standardization(
        accumulation_fc, params.calibration_start, params.calibration_stop, members=True,
    )
    anomaly_obs = run_gamma_standardization(
        accumulation_obs,
        params.calibration_start,
        params.calibration_stop,
    )
    logging.info(f"Completed anomaly")

    # Probabilities without Bias Correction
    probabilities = compute_probabilities(
        anomaly_fc.where(anomaly_fc.time.dt.year == params.year, drop=True),
        levels=params.intensity_thresholds,
    ).round(2)

    # Bias correction
    index_bc = run_bias_correction(
        anomaly_fc,
        anomaly_obs,
        params.end_season,
        params.year,
        int(params.issue),
        enso=True,
    )
    logging.info(f"Completed bias correction")

    # Probabilities after Bias Correction
    probabilities_bc = compute_probabilities(
        index_bc, levels=params.intensity_thresholds
    ).round(2)
    logging.info(f"Completed probabilities")

    return probabilities, probabilities_bc


if __name__ == "__main__":
    # From AA repository:
    # $ python operational.py MOZ 10 SPI

    run()
