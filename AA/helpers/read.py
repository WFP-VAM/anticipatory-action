import os
import copy
import datetime
import fsspec
import logging
import numpy as np
import pandas as pd
import xarray as xr

from hip.analysis.compute.utils import persist_with_progress_bar


def read_roc_file(roc_path, params):
    roc = pd.read_csv(roc_path, sep=",")
    if params.issue:
        roc = roc.loc[roc.issue == params.issue]
    return roc


def read_forecasts(area, issue, local_path):
    """
    Load ECMWF SEAS5 seasonal forecast data, using a local zarr cache when available.

    Checks the last cached date and fetches only the data from the following day
    onwards, appending it to the existing cache.

    Args:
        area: Area object with a `datetime_range` attribute (e.g. "2023-06-01/2024-12-31")
              and `get_dataset()` method.
        issue (int): Forecast issue month (1–12).
        local_path (str): Path to the local zarr store used as a cache.

    Returns:
        xarray.DataArray: The `tp` variable from the forecast dataset, covering all
                          timesteps up to `last_date`.
    """
    fs = fsspec.open(local_path).fs
    zmetadata_path = os.path.join(local_path, ".zmetadata")
    data_exists = fs.exists(zmetadata_path)

    # Derive the monitoring window:
    # - last_date: end of the target range (e.g. 2024-12-31)
    last_date = datetime.datetime.strptime(
        area.datetime_range.split("/")[1], "%Y-%m-%d"
    )

    if data_exists:
        logging.info("Reading forecasts from precomputed zarr...")
        ds = xr.open_zarr(local_path).tp

        # Find the day after the last cached date and fetch everything from there
        last_cached_date = pd.Timestamp(ds.time.values.max()).date()
        fetch_start = last_cached_date + datetime.timedelta(days=1)

        if fetch_start > last_date.date():
            logging.info("All forecast data present, returning cached data...")
            return persist_with_progress_bar(ds.sel(time=slice(None, last_date)))

        logging.info(
            f"Fetching missing forecasts from {fetch_start} to {last_date.date()}..."
        )
        area_slice = copy.deepcopy(area)
        area_slice.datetime_range = f"{fetch_start}/{last_date.date()}"
        new_data = area_slice.get_dataset(
            ["ECMWF", f"RFH_FORECASTS_SEAS5_ISSUE{int(issue)}_DAILY"],
            load_config={"gridded_load_kwargs": {"resampling": "bilinear"}},
        )
        new_data.attrs["nodata"] = np.nan
        new_data.chunk({"time": -1}).to_zarr(local_path, mode="a", append_dim="time")

        # Re-open the zarr to get a consistent view that includes the appended data
        ds = xr.open_zarr(local_path).tp
        return persist_with_progress_bar(ds.sel(time=slice(None, last_date)))

    # No cache exists yet — fetch the full range and write it
    logging.info("Zarr not found, reading forecasts from source...")
    forecasts = area.get_dataset(
        ["ECMWF", f"RFH_FORECASTS_SEAS5_ISSUE{int(issue)}_DAILY"],
        load_config={"gridded_load_kwargs": {"resampling": "bilinear"}},
    )
    forecasts.attrs["nodata"] = np.nan
    forecasts.chunk({"time": -1}).to_zarr(local_path, mode="w", consolidated=True)
    return persist_with_progress_bar(forecasts)


def read_observations(area, local_path):
    """
    Load CHIRPS daily observation data, using a local zarr cache when available.
    Compares the years present in the cache against those required by `area.datetime_range`
    and only fetches missing years from the source, then appends them to the cache.

    Args:
        area: Area object with a `datetime_range` attribute (e.g. "2023-01-01/2024-12-31")
              and `get_dataset()` / `with_datetime_range()` methods.
        local_path (str): Path to the local zarr store used as a cache.

    Returns:
        xarray.DataArray: The `band` variable from the observations dataset,
                          covering the full requested date range.
    """
    fs = fsspec.open(local_path).fs
    zmetadata_path = os.path.join(local_path, ".zmetadata")
    data_exists = fs.exists(zmetadata_path)

    # Parse the full requested date range from the area object
    last_date = datetime.datetime.strptime(
        area.datetime_range.split("/")[1], "%Y-%m-%d"
    )

    if data_exists:
        logging.info("Reading observations from precomputed zarr...")
        ds = xr.open_zarr(local_path, consolidated=True).band

        # Find the day after the last cached date and fetch everything from there
        last_cached_date = pd.Timestamp(ds.time.values.max()).date()
        fetch_start = last_cached_date + datetime.timedelta(days=1)

        if fetch_start > last_date.date():
            logging.info("All observation data present, returning cached data...")
            return persist_with_progress_bar(ds)

        logging.info(
            f"Fetching missing observations from {fetch_start} to {last_date.date()}..."
        )
        area.datetime_range = f"{fetch_start}/{last_date.date()}"
        new_data = area.get_dataset(
            ["CHIRPS", "RFH_DAILY"],
            load_config={"gridded_load_kwargs": {"resampling": "bilinear"}},
        )

        new_data.to_zarr(local_path, mode="a", append_dim="time")
        ds = xr.open_zarr(local_path, consolidated=True).band
        return persist_with_progress_bar(ds)

    else:
        # No cache exists yet — fetch the full range and write it
        logging.info("Reading observations from HDC STAC...")
        observations = area.get_dataset(
            ["CHIRPS", "RFH_DAILY"],
            load_config={"gridded_load_kwargs": {"resampling": "bilinear"}},
        )
        observations.to_zarr(local_path, mode="w", consolidated=True)
        return persist_with_progress_bar(observations)


def read_triggers(params):
    triggers_path = f"{params.data_path}/data/{params.iso}/probs/aa_probabilities_triggers_pilots.csv"
    fallback_triggers_path = f"{params.data_path}/data/{params.iso}/triggers/triggers.final.{params.monitoring_year}.pilots.csv"

    if fsspec.open(triggers_path).fs.exists(triggers_path):
        triggers_df = pd.read_csv(triggers_path)
    else:
        triggers_df = pd.read_csv(fallback_triggers_path)
    return triggers_df