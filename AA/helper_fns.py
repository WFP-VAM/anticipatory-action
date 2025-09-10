import datetime
import glob
import logging
import os

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from hip.analysis.compute.utils import persist_with_progress_bar

from AA.logging_utils import log_array_info, log_data_loading_step

PORTUGUESE_CATEGORIES = dict(
    Normal="Normal", Mild="Leve", Moderate="Moderado", Severe="Severo"
)


def create_flexible_dataarray(start_season, end_season):
    # Create the start and end dates
    start_date = datetime.datetime(1990, start_season, 1)
    end_date = datetime.datetime(1991, end_season + 1, 28)

    # Generate the date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="M")

    # Create the DataArray
    data_array = xr.DataArray(
        np.arange(1, len(date_range) + 1),  # Create a range of values for demonstration
        coords=dict(time=(["time"], date_range)),
        dims="time",
    )

    return data_array


def triggers_da_to_df(triggers_da, score_da):
    """Converts trigger and score DataArrays to a merged DataFrame.

    This function processes two xarray DataArrays containing trigger values and scores,
    converts them to pandas DataFrames, and merges them based on specified indices.

    Args:
        triggers_da (xarray.DataArray): DataArray containing trigger information.
        score_da (xarray.DataArray): DataArray containing score information.

    Returns:
        pandas.DataFrame: A DataFrame combining trigger values and scores, with duplicates removed.
    """
    # Convert triggers to DataFrame, clean and set index
    triggers_df = (
        triggers_da.rename("trigger_value")
        .to_dataframe()
        .drop(columns=["spatial_ref", "return_period", "tolerance"], errors="ignore")
        .dropna()
        .reset_index()
        .set_index(["index", "category", "district", "issue"])
    )

    # Convert score to DataFrame, clean and set index
    score_df = (
        score_da.rename("HR")
        .to_dataframe()
        .reset_index()
        .drop(columns=["lead_time", "return_period", "tolerance"], errors="ignore")
        .set_index(["district", "category", "issue", "index"])
    )

    # Join triggers with score
    triggers_df = triggers_df.join(score_df)

    # Reset index and remove duplicates
    return triggers_df.reset_index().drop_duplicates()


def compute_district_average(da, area):
    """
    Computes zonal statistics on an xarray DataArray for both observations and probabilities.

    Args:
        da : xarray.DataArray, Input DataArray (can be observations or probabilities).
        area : hip.analysis.aoi.analysis_area.AnalysisArea: object characterizing the area
            and admin level of interest.
    Returns: xarray.DataArray, DataArray with computed district averages.
    """
    logger = logging.getLogger('aa_operational')
    logger.debug("=== Computing district averages ===")
    
    # Log input array information
    log_array_info(logger, "Input_GriddedData", da)
    
    # Ensure consistent time dimension
    if "year" in da.dims:
        logger.debug("Renaming 'year' dimension to 'time'")
        da = da.rename({"year": "time"})

    # Determine dimensions to group by (exclude spatial dimensions)
    groupby_dim = set(da.dims) - {"latitude", "longitude", "time"}
    logger.debug("Groupby dimensions: %s", groupby_dim)
    logger.debug("Spatial grid: lat (%d), lon (%d)", da.latitude.size, da.longitude.size)
    
    # Log spatial bounds
    logger.debug("Latitude range: %.4f to %.4f", 
                float(da.latitude.min()), float(da.latitude.max()))
    logger.debug("Longitude range: %.4f to %.4f", 
                float(da.longitude.min()), float(da.longitude.max()))

    # Transpose dims to ensure equality of shapes
    da = da.transpose(..., *groupby_dim, "latitude", "longitude")
    logger.debug("After transpose - dims: %s, shape: %s", da.dims, da.shape)

    # Compute zonal stats: handle different groupby dimensions lengths
    if len(groupby_dim) > 1:
        raise NotImplementedError(
            "Zonal stats with more than one groupby dimension are not supported."
        )
    elif len(groupby_dim) == 1:
        logger.debug("Computing zonal stats with groupby dimension: %s", list(groupby_dim)[0])
        da_grouped = da.groupby(*groupby_dim).map(
            lambda da: area.zonal_stats(
                da.squeeze(groupby_dim), stats=["mean"], zone_ids=None, zones=None
            )
            .query("zone != 'Administrative unit not available'")
            .to_xarray()["mean"]
        )
    else:
        logger.debug("Computing zonal stats without groupby dimensions")
        da_grouped = (
            area.zonal_stats(da, stats=["mean"], zone_ids=None, zones=None)
            .query("zone != 'Administrative unit not available'")
            .to_xarray()["mean"]
        )

    # Log zonal stats results
    logger.debug("Districts found: %d", da_grouped.zone.size if 'zone' in da_grouped.dims else 0)
    if 'zone' in da_grouped.dims:
        districts = list(da_grouped.zone.values)
        logger.debug("District names: %s", districts[:10])  # Log first 10 districts
        if len(districts) > 10:
            logger.debug("... and %d more districts", len(districts) - 10)

    # Rename 'zone' to 'district' for consistency
    da_grouped = da_grouped.rename({"zone": "district"})

    # Ensure district is a string type
    da_grouped["district"] = da_grouped.district.astype(str)
    
    # Log final result
    log_array_info(logger, "District_Averaged_Data", da_grouped)

    return da_grouped


def merge_un_biased_probs(probs_district, probs_bc_district, params, period_name):
    logger = logging.getLogger('aa_operational')
    logger.debug("=== Merging unbiased and bias-corrected probabilities ===")
    
    # Log input arrays
    log_array_info(logger, "Input_Raw_Probabilities_District", probs_district)
    log_array_info(logger, "Input_BiasCorreected_Probabilities_District", probs_bc_district)
    
    # Get fbf_districts data in xarray format
    fbf_bc = params.fbf_districts_df
    logger.debug("FBF districts data shape before filtering: %s", fbf_bc.shape)
    
    index_key = f"{params.index.upper()} {period_name}"
    fbf_bc = fbf_bc.loc[fbf_bc["Index"] == index_key]
    logger.debug("FBF districts data shape after filtering for index '%s': %s", index_key, fbf_bc.shape)
    
    fbf_bc = fbf_bc[["district", "category", "issue", "BC"]]
    log_array_info(logger, "FBF_Districts_Config", fbf_bc)

    # If params.fbf_districts_df has Portuguese category names, ensure these are English
    CATEGORY_TRANSLATIONS = {"Leve": "Mild", "Moderado": "Moderate", "Severo": "Severe"}
    original_categories = fbf_bc["category"].unique()
    fbf_bc["category"] = fbf_bc["category"].apply(
        lambda x: CATEGORY_TRANSLATIONS.get(x, x)
    )
    translated_categories = fbf_bc["category"].unique()
    logger.debug("Category translation: %s -> %s", original_categories, translated_categories)

    fbf_bc_da = fbf_bc.set_index(["district", "category", "issue"]).to_xarray().BC
    fbf_bc_da = fbf_bc_da.expand_dims(dim={"index": [f"{params.index} {period_name}"]})
    
    log_array_info(logger, "FBF_BiasCorrection_Flags", fbf_bc_da)

    # Log bias correction application
    bc_districts = fbf_bc[fbf_bc["BC"] == 1]["district"].unique()
    raw_districts = fbf_bc[fbf_bc["BC"] == 0]["district"].unique()
    logger.debug("Districts using bias correction: %d districts", len(bc_districts))
    logger.debug("Districts using raw probabilities: %d districts", len(raw_districts))

    # Combination of both probabilities datasets
    probs_merged = (1 - fbf_bc_da) * probs_district + fbf_bc_da * probs_bc_district
    logger.info("Applied conditional bias correction mixing")

    probs_merged = probs_merged.to_dataset(name="prob")
    log_array_info(logger, "Final_Merged_Probabilities", probs_merged)

    return probs_merged


def format_triggers_df_for_dashboard(triggers, params):
    triggers["index"] = triggers["index"].str.upper()
    triggers.loc[(triggers.trigger == "trigger2") & (triggers.issue == 12), "issue"] = 0
    triggers.loc[triggers.trigger == "trigger2", "issue"] = (
        triggers.loc[triggers.trigger == "trigger2"].issue.values + 1
    )

    triggers["prob"] = np.nan
    triggers["HR"] = triggers["HR"].abs()

    if "season" not in triggers.columns:
        triggers["season"] = (
            f"{params.monitoring_year}-{str(params.monitoring_year + 1)[-2:]}"
        )
        triggers["date"] = [
            params.monitoring_year if r.issue >= 5 else params.monitoring_year + 1
            for _, r in triggers.iterrows()
        ]
        triggers["date"] = [
            pd.to_datetime(f"{r.issue}-1-{r.date}") for _, r in triggers.iterrows()
        ]

    def substract(issue):
        return 2 if issue == 1 else 1

    triggers["mready"] = [
        r.issue if r.trigger == "trigger1" else (r.issue - substract(int(r.issue))) % 13
        for _, r in triggers.iterrows()
    ]

    triggers_pivot = triggers.pivot_table(
        index=["district", "index", "category", "Window", "mready"],
        columns="trigger",
        values=["trigger_value", "prob", "issue"],
    ).reset_index()
    triggers_pivot.columns = [
        "district",
        "index",
        "category",
        "window",
        "mready",
        "issue_ready",
        "issue_set",
        "trigger_ready",
        "trigger_set",
    ]
    triggers_pivot = triggers_pivot.drop("mready", axis=1)

    return triggers_pivot


def get_coverage(triggers_df, districts: list, columns: list):
    cov = pd.DataFrame(
        columns=columns,
        index=districts,
    )
    for d, _ in cov.iterrows():
        val = []
        for w in triggers_df["window"].unique():
            for c in triggers_df["category"].unique():
                val.append(
                    len(
                        triggers_df[
                            (triggers_df["window"] == w)
                            & (triggers_df["category"] == c)
                            & (triggers_df["district"] == d)
                        ]
                    )
                )
        cov.loc[d] = val

    print(
        f"The coverage is {round(100 * np.sum(cov.values > 0) / np.size(cov.values), 1)} %"
    )
    return cov


def load_trigger_with_reference(params, variant_folder=None):
    """
    Load trigger data and reference trigger data for comparison.

    Args:
        params: An object containing parameters such as data_path, iso, and calibration_year.
        variant_folder (str, optional): If provided, modifies the data path to load from an alternative directory.

    Returns:
        dict: A dictionary containing DataFrames for GT, NRT, and pilots triggers and reference triggers.
    """
    base_path = f"{params.data_path}/data/{variant_folder or params.iso}"

    files = ["GT", "NRT", "pilots"]

    triggers = {}
    for file in files:
        trigger_path = f"{base_path}/triggers/triggers.spi.dryspell.{params.calibration_year}.{file}.csv"
        ref_path = f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.{file}.csv"

        triggers[f"triggers_{file}"] = pd.read_csv(trigger_path)
        triggers[f"reference_{file}"] = pd.read_csv(ref_path)

    return triggers


def merge_probabilities_triggers_dashboard(probs, triggers, params, period):
    logger = logging.getLogger('aa_operational')
    logger.debug("=== Merging probabilities with triggers for dashboard ===")
    
    # Log inputs
    log_array_info(logger, "Input_Probabilities_Dataset", probs)
    log_array_info(logger, "Input_Triggers", triggers)
    logger.debug("Processing period: %s, issue: %d", period, params.issue)
    
    # Format probabilities
    probs_df = probs.to_dataframe().reset_index()
    probs_df["prob"] = [np.round(p, 2) for p in probs_df.prob.values]
    probs_df["index"] = probs_df["index"].str.upper()
    probs_df["aggregation"] = np.repeat(
        f"{params.index.upper()} {len(period)}", len(probs_df)
    )
    
    log_array_info(logger, "Formatted_Probabilities", probs_df)

    triggers_merged = triggers.copy()
    logger.debug("Triggers dataframe copied, shape: %s", triggers_merged.shape)

    # Create prob columns if reading empty triggers df
    if "prob_ready" not in triggers_merged.columns:
        logger.debug("Creating prob_ready and prob_set columns")
        triggers_merged["prob_ready"] = np.nan
        triggers_merged["prob_set"] = np.nan

    # Fill in probabilities columns matching with triggers
    target_index = f"{params.index.upper()} {period}"
    matched_ready = 0
    matched_set = 0
    
    # Drop all rows that do not related to the target_index
    triggers_merged = triggers_merged[triggers_merged['index']==target_index]
    
    for idx, row in triggers_merged.iterrows():
        if (row.issue_ready == params.issue):
            match_filter = (
                (probs_df["index"] == target_index)
                & (probs_df["category"] == row.category)
                & (probs_df["district"] == row.district)
            )
            matching_probs = probs_df.loc[match_filter]
            if len(matching_probs) > 0:
                prob_value = matching_probs.prob.values[0]
                triggers_merged.loc[idx, "prob_ready"] = prob_value
                logger.debug("Updated %s %s %s %s (READY): value: %.3f", 
                           row.issue_ready, row.district, row.category, 
                           target_index, prob_value)
                matched_ready += 1
                
        elif (row.issue_set == params.issue):
            match_filter = (
                (probs_df["index"] == target_index)
                & (probs_df["category"] == row.category)
                & (probs_df["district"] == row.district)
            )
            matching_probs = probs_df.loc[match_filter]
            if len(matching_probs) > 0:
                prob_value = matching_probs.prob.values[0]
                triggers_merged.loc[idx, "prob_set"] = prob_value
                logger.debug("Updated %s %s %s %s (SET): value: %.3f", 
                           row.issue_set, row.district, row.category, 
                           target_index, prob_value)
                matched_set += 1

    logger.debug("Probability matching complete:")
    logger.debug("  Ready triggers matched: %d", matched_ready)
    logger.debug("  Set triggers matched: %d", matched_set)
    logger.debug("  Target index: %s", target_index)
    
    log_array_info(logger, "Final_Triggers_With_Probabilities", triggers_merged)

    return probs_df, triggers_merged


def read_fbf_districts(path_fbf, params):
    fbf_districts = pd.read_csv(path_fbf, sep=",")
    if params.issue:
        fbf_districts = fbf_districts.loc[fbf_districts.issue == params.issue]
    return fbf_districts


def read_forecasts(area, issue, local_path):
    logger = logging.getLogger('aa_operational')
    
    fs = fsspec.open(local_path).fs
    zmetadata_path = os.path.join(local_path, ".zmetadata")
    data_exists = fs.exists(zmetadata_path)
    
    # Determine the forecast period:
    # - `last_date` is the end of the target time range, extracted from area.datetime_range (e.g., "2024-01-01/2024-12-31")
    # - `forecast_date` is the start of the forecast, set to the 1st of the issue month of the year before `last_date`, as last_date.year = monitoring_year + 1
    #   For example, if issue=6 (June) and last_date is 2024-12-31, then forecast_date becomes 2023-06-01
    last_date = datetime.datetime.strptime(
        area.datetime_range.split("/")[1], "%Y-%m-%d"
    )
    forecast_date = datetime.datetime(last_date.year - 1, int(issue), 1)
    
    logger.debug("Forecast loading parameters: issue=%d, last_date=%s, forecast_date=%s", 
                issue, last_date, forecast_date)
    logger.debug("Local zarr path: %s", local_path)
    logger.debug("Zarr metadata exists: %s", data_exists)

    if data_exists:
        logger.info("Reading forecasts from precomputed zarr...")
        ds = xr.open_zarr(local_path).tp
        log_data_loading_step(logger, "ECMWF forecasts", local_path, ds, "zarr cache")
        
        if np.datetime64(forecast_date) in ds.time.values:
            result = persist_with_progress_bar(ds.sel(time=slice(None, last_date)))
            log_array_info(logger, "Final_Cached_Forecasts", result)
            return result
        else:
            logger.info("Forecast date missing in zarr, reading from source...")
    else:
        logger.info("Zarr file not found, reading from source...")

    # Read from source
    logger.debug("Loading from ECMWF SEAS5 source: RFH_FORECASTS_SEAS5_ISSUE%d_DAILY", int(issue))
    forecasts = area.get_dataset(
        ["ECMWF", f"RFH_FORECASTS_SEAS5_ISSUE{int(issue)}_DAILY"],
        load_config={"gridded_load_kwargs": {"resampling": "bilinear"}},
    )
    log_data_loading_step(logger, "ECMWF forecasts", "HDC STAC", forecasts, "source API")
    
    forecasts.attrs["nodata"] = np.nan
    logger.debug("Caching to zarr: %s", local_path)
    forecasts.chunk({"time": -1}).to_zarr(local_path, mode="w", consolidated=True)

    result = persist_with_progress_bar(forecasts)
    log_array_info(logger, "Final_Source_Forecasts", result)
    return result


def read_observations(area, local_path):
    logger = logging.getLogger('aa_operational')
    
    fs = fsspec.open(local_path).fs
    zarr_exists = fs.exists(os.path.join(local_path, ".zmetadata"))
    
    logger.debug("Observations loading parameters: local_path=%s", local_path)
    logger.debug("Zarr metadata exists: %s", zarr_exists)
    
    if zarr_exists:
        logger.info("Reading of observations from precomputed zarr...")
        observations = xr.open_zarr(
            local_path,
            consolidated=True,
        ).band
        log_data_loading_step(logger, "CHIRPS observations", local_path, observations, "zarr cache")
    else:
        logger.info("Reading of observations from HDC STAC...")
        observations = area.get_dataset(
            ["CHIRPS", "RFH_DAILY"],
            load_config={"gridded_load_kwargs": {"resampling": "bilinear"}},
        )
        log_data_loading_step(logger, "CHIRPS observations", "HDC STAC", observations, "source API")
        
        logger.debug("Caching to zarr: %s", local_path)
        observations.to_zarr(
            local_path,
            mode="w",
            consolidated=True,
        )

    result = persist_with_progress_bar(observations)
    log_array_info(logger, "Final_Observations", result)
    return result


def read_triggers(params):
    logger = logging.getLogger('aa_operational')
    
    triggers_path = f"{params.data_path}/data/{params.iso}/probs/aa_probabilities_triggers_pilots.csv"
    fallback_triggers_path = f"{params.data_path}/data/{params.iso}/triggers/triggers.final.{params.monitoring_year}.pilots.csv"
    
    logger.debug("Primary triggers path: %s", triggers_path)
    logger.debug("Fallback triggers path: %s", fallback_triggers_path)
    
    path_exists = fsspec.open(triggers_path).fs.exists(triggers_path)
    logger.debug("Primary triggers file exists: %s", path_exists)

    if path_exists:
        logger.info("Loading triggers from primary path (probabilities-triggers file)")
        triggers_df = pd.read_csv(triggers_path)
        log_data_loading_step(logger, "triggers", triggers_path, triggers_df, "primary CSV")
    else:
        logger.info("Loading triggers from fallback path (final triggers file)")
        triggers_df = pd.read_csv(fallback_triggers_path)
        log_data_loading_step(logger, "triggers", fallback_triggers_path, triggers_df, "fallback CSV")
    
    return triggers_df


def validate_prism_dataframe(df: pd.DataFrame):
    # Expected columns and types
    expected_columns = {
        "district": str,
        "index": str,
        "category": str,
        "window": str,
        "issue_ready": float | int,
        "issue_set": float | int,
        "trigger_ready": float,
        "trigger_set": float,
        "vulnerability": str,
        "prob_ready": float,
        "prob_set": float,
        "season": str,
        "date_ready": str,
        "date_set": str,
    }

    # Check columns
    missing = set(expected_columns) - set(df.columns)
    extra = set(df.columns) - set(expected_columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if extra:
        raise ValueError(f"Unexpected columns: {extra}")

    # Check column types
    for col, expected_type in expected_columns.items():
        if not df[col].map(lambda x: isinstance(x, expected_type)).all():
            raise TypeError(
                f"Column '{col}' has incorrect type. Expected {expected_type.__name__}"
            )

    # Check index column is uppercase
    if not df["index"].map(lambda x: x.isupper()).all():
        raise ValueError("All values in 'index' column must be uppercase")

    # Check category values
    valid_categories = {"Normal", "Mild", "Moderate", "Severe"}
    if not df["category"].isin(valid_categories).all():
        raise ValueError(
            f"'category' column contains invalid values. Allowed: {valid_categories}"
        )

    # Check window values
    valid_windows = {"Window 1", "Window 2"}
    if not df["window"].isin(valid_windows).all():
        raise ValueError(
            f"'window' column contains invalid values. Allowed: {valid_windows}"
        )

    # Check vulnerability values
    valid_vulnerability = {"General Triggers", "Emergency Triggers"}
    if not df["vulnerability"].isin(valid_vulnerability).all():
        raise ValueError(
            f"'vulnerability' column contains invalid values. Allowed: {valid_vulnerability}"
        )

    # Check date formats
    for col in ["date_ready", "date_set"]:
        try:
            parsed_dates = pd.to_datetime(df[col], format="%Y-%m-%d", errors="raise")
        except Exception:
            raise ValueError(f"Column '{col}' must use format YYYY-MM-DD")

        if not all(parsed_dates.dt.day == 1):
            raise ValueError(f"All dates in '{col}' must have day = 01")

    return True


## Get SPI/probabilities of reference produced with R script from Gabriela Nobre for validation ##


def read_spi_references(path_ref, bc: bool = False, obs: bool = False):
    df_ref = pd.DataFrame()
    files_ref_index = glob.glob(f"{path_ref}*.csv")
    list_index_csv = []
    for ind in files_ref_index:
        df_ind = pd.read_csv(ind).melt(id_vars=["V1", "V2"])
        if not (bc) and not (obs):
            df_ind["year"] = [
                np.int16(v.split("_")[-1]) for v in df_ind.variable.values
            ]
            df_ind = df_ind.loc[df_ind.year == 2022]
        if obs:
            df_ind["period"] = np.repeat(ind.split(".")[0].split(" ")[1], len(df_ind))
            offset_year = (sorted(df_ind.variable.values)[0] == "1982") * 1
            df_ind["variable"] = [int(y) - offset_year for y in df_ind.variable.values]
        else:
            df_ind["period"] = np.repeat(ind.split("/")[-1].split(".")[-2], len(df_ind))
            df_ind["year"] = [
                np.int16(e.split("_")[-1]) for e in df_ind.variable.values
            ]
            df_ind["variable"] = [
                np.float64(e.split("_")[-2]) for e in df_ind.variable.values
            ]
            df_ind = df_ind.loc[df_ind.year == 2022]
        df_ind.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_ind = df_ind.dropna()
        list_index_csv.append(df_ind)
    df_ref_spi = pd.concat(list_index_csv)
    df_ref = pd.concat([df_ref, df_ref_spi])
    if bc:
        df_ref.columns = [
            "longitude",
            "latitude",
            "ensemble",
            "spi_ref",
            "period",
            "year",
        ]
    elif obs:
        df_ref.columns = ["longitude", "latitude", "year", "spi_ref", "period"]
    else:
        df_ref.columns = [
            "longitude",
            "latitude",
            "ensemble",
            "spi_ref",
            "year",
            "period",
        ]
    return df_ref


def read_probas_references(path_ref_probas, cats):
    df_ref = pd.DataFrame()
    for cat in cats:
        files_ref_index = glob.glob(f"{path_ref_probas}{cat}/*")
        list_index_csv = []
        for ind in files_ref_index:
            df_ind = pd.read_csv(ind).melt(id_vars=["V1", "V2"])
            df_ind["period"] = np.repeat(
                ind.split("/")[-1].split(".")[-2].split("_")[0], len(df_ind)
            )
            df_ind["category"] = np.repeat(PORTUGUESE_CATEGORIES[cat], len(df_ind))
            df_ind = df_ind.dropna()
            list_index_csv.append(df_ind)
        df_ref_cat = pd.concat(list_index_csv)
        df_ref = pd.concat([df_ref, df_ref_cat])
    df_ref.columns = [
        "longitude",
        "latitude",
        "year",
        "probability_ref",
        "period",
        "category",
    ]
    df_ref = df_ref[df_ref.year == "2022"].drop("year", axis=1)
    return df_ref
