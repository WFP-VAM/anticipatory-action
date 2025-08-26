import datetime
import glob
import logging
import os

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from hip.analysis.compute.utils import persist_with_progress_bar

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
    # Ensure consistent time dimension
    if "year" in da.dims:
        da = da.rename({"year": "time"})

    # Determine dimensions to group by (exclude spatial dimensions)
    groupby_dim = set(da.dims) - {"latitude", "longitude", "time"}

    # Transpose dims to ensure equality of shapes
    da = da.transpose(..., *groupby_dim, "latitude", "longitude")

    # Compute zonal stats: handle different groupby dimensions lengths
    if len(groupby_dim) > 1:
        raise NotImplementedError(
            "Zonal stats with more than one groupby dimension are not supported."
        )
    elif len(groupby_dim) == 1:
        da_grouped = da.groupby(*groupby_dim).map(
            lambda da: area.zonal_stats(
                da.squeeze(groupby_dim), stats=["mean"], zone_ids=None, zones=None
            )
            .query("zone != 'Administrative unit not available'")
            .to_xarray()["mean"]
        )
    else:
        da_grouped = (
            area.zonal_stats(da, stats=["mean"], zone_ids=None, zones=None)
            .query("zone != 'Administrative unit not available'")
            .to_xarray()["mean"]
        )

    # Rename 'zone' to 'district' for consistency
    da_grouped = da_grouped.rename({"zone": "district"})

    # Ensure district is a string type
    da_grouped["district"] = da_grouped.district.astype(str)

    return da_grouped


def merge_un_biased_probs(probs_district, probs_bc_district, params, period_name):
    # Get fbf_districts data in xarray format
    fbf_bc = params.fbf_districts_df
    fbf_bc = fbf_bc.loc[fbf_bc["Index"] == f"{params.index.upper()} {period_name}"]
    fbf_bc = fbf_bc[["district", "category", "issue", "BC"]]

    # If params.fbf_districts_df has Portuguese category names, ensure these are English
    CATEGORY_TRANSLATIONS = {"Leve": "Mild", "Moderado": "Moderate", "Severo": "Severe"}
    fbf_bc["category"] = fbf_bc["category"].apply(
        lambda x: CATEGORY_TRANSLATIONS.get(x, x)
    )

    fbf_bc_da = fbf_bc.set_index(["district", "category", "issue"]).to_xarray().BC
    fbf_bc_da = fbf_bc_da.expand_dims(dim={"index": [f"{params.index} {period_name}"]})

    # Combination of both probabilities datasets
    probs_merged = (1 - fbf_bc_da) * probs_district + fbf_bc_da * probs_bc_district

    probs_merged = probs_merged.to_dataset(name="prob")

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
    # Format probabilities
    probs_df = probs.to_dataframe().reset_index()
    probs_df["prob"] = [np.round(p, 2) for p in probs_df.prob.values]
    probs_df["index"] = probs_df["index"].str.upper()
    probs_df["aggregation"] = np.repeat(
        f"{params.index.upper()} {len(period)}", len(probs_df)
    )

    triggers_merged = triggers.copy()

    # Create prob columns if reading empty triggers df
    if "prob_ready" not in triggers_merged.columns:
        triggers_merged["prob_ready"] = np.nan
        triggers_merged["prob_set"] = np.nan

    # Fill in probabilities columns matching with triggers
    for idx, row in triggers_merged.iterrows():
        if (row.issue_ready == params.issue) and (
            row["index"] == f"{params.index.upper()} {period}"
        ):
            triggers_merged.loc[idx, "prob_ready"] = probs_df.loc[
                (probs_df["index"] == row["index"])
                & (probs_df["category"] == row.category)
                & (probs_df["district"] == row.district)
            ].prob.values[0]
        elif (row.issue_set == params.issue) and (
            row["index"] == f"{params.index.upper()} {period}"
        ):
            triggers_merged.loc[idx, "prob_set"] = probs_df.loc[
                (probs_df["index"] == row["index"])
                & (probs_df["category"] == row.category)
                & (probs_df["district"] == row.district)
            ].prob.values[0]

    return probs_df, triggers_merged


def read_fbf_districts(path_fbf, params):
    fbf_districts = pd.read_csv(path_fbf, sep=",")
    if params.issue:
        fbf_districts = fbf_districts.loc[fbf_districts.issue == params.issue]
    return fbf_districts


def read_forecasts(area, issue, local_path):
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

    if data_exists:
        logging.info("Reading forecasts from precomputed zarr...")
        ds = xr.open_zarr(local_path).tp
        if np.datetime64(forecast_date) in ds.time.values:
            return persist_with_progress_bar(ds.sel(time=slice(None, last_date)))
        else:
            logging.info("Forecast date missing in zarr, reading from source...")
    else:
        logging.info("Zarr file not found, reading from source...")

    # Read from source
    forecasts = area.get_dataset(
        ["ECMWF", f"RFH_FORECASTS_SEAS5_ISSUE{int(issue)}_DAILY"],
        load_config={"gridded_load_kwargs": {"resampling": "bilinear"}},
    )
    forecasts.attrs["nodata"] = np.nan
    forecasts.chunk({"time": -1}).to_zarr(local_path, mode="w", consolidated=True)

    return persist_with_progress_bar(forecasts)


def read_observations(area, local_path):
    fs = fsspec.open(local_path).fs
    if fs.exists(os.path.join(local_path, ".zmetadata")):
        logging.info("Reading of observations from precomputed zarr...")
        observations = xr.open_zarr(
            local_path,
            consolidated=True,
        ).band
    else:
        logging.info("Reading of observations from HDC STAC...")
        observations = area.get_dataset(
            ["CHIRPS", "RFH_DAILY"],
            load_config={"gridded_load_kwargs": {"resampling": "bilinear"}},
        )
        observations.to_zarr(
            local_path,
            mode="w",
            consolidated=True,
        )

    return persist_with_progress_bar(observations)


def read_triggers(params):
    triggers_path = f"{params.data_path}/data/{params.iso}/probs/aa_probabilities_triggers_pilots.csv"
    fallback_triggers_path = f"{params.data_path}/data/{params.iso}/triggers/triggers.final.{params.monitoring_year}.pilots.csv"

    if fsspec.open(triggers_path).fs.exists(triggers_path):
        triggers_df = pd.read_csv(triggers_path)
    else:
        triggers_df = pd.read_csv(fallback_triggers_path)
    return triggers_df


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
