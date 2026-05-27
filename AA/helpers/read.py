import os
import copy
import datetime
import fsspec
import logging
import numpy as np
import pandas as pd
import xarray as xr

from hip.analysis.compute.utils import persist_with_progress_bar


CHIRPS_BLENDED_STORE = "observations_blended.zarr"
CHIRPS_DEKAD_STORE = "observations_dekad.zarr"
CHIRPS_DAILY_STORE = "observations_daily.zarr"

INDEX_STORE_MAP = {
    "spi": (CHIRPS_DEKAD_STORE, ["CHIRPS", "RFH_DEKAD"]),
    "dryspell": (CHIRPS_DAILY_STORE, ["CHIRPS", "RFH_DAILY_RNL"]),
}


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

        gap_days = (last_date.date() - last_cached_date).days
        if fetch_start > last_date.date() or gap_days < 150:
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
    forecasts.chunk({"time": -1}).to_zarr(
        local_path, mode="w", consolidated=True, zarr_version=2
    )
    return persist_with_progress_bar(forecasts)


def read_observations(area, local_path: str, index: str):
    """
    Load CHIRPS observation data, using a local zarr cache when available.

    Checks for a blended zarr store first (observations_blended.zarr). If absent,
    falls back to the index-specific store and dataset key:
      - "spi"      → observations_dekad.zarr  / CHIRPS | RFH_DEKAD
      - "dryspell" → observations_daily.zarr  / CHIRPS | RFH_DAILY_RNL

    In both cases the CHIRPS v2 → v3 alias resolution is handled transparently
    by the backend (RFH_DEKAD → CHIRPS3|RFH_DEKAD, etc.).

    Compares the last cached date against the end of `area.datetime_range` and
    only fetches the missing tail before appending it to the cache.

    Args:
        area:       Area object with a `datetime_range` attribute
                    (e.g. "2023-01-01/2024-12-31") and `get_dataset()` /
                    `with_datetime_range()` methods.
        local_path: Directory that holds (or will hold) the zarr stores.
        index:      One of "spi" or "dryspell".

    Returns:
        xarray.DataArray: The `band` variable covering the full requested range.
    """
    if index not in INDEX_STORE_MAP:
        raise ValueError(
            f"Unknown index '{index}'. Expected one of {list(INDEX_STORE_MAP)}"
        )

    last_date = datetime.datetime.strptime(
        area.datetime_range.split("/")[1], "%Y-%m-%d"
    ).date()

    # ------------------------------------------------------------------ #
    # 1. Blended store takes priority regardless of index                  #
    # ------------------------------------------------------------------ #
    blended_store = os.path.join(local_path, CHIRPS_BLENDED_STORE)
    fs = fsspec.open(blended_store).fs

    if fs.exists(os.path.join(blended_store, ".zmetadata")):
        logging.info("Blended observations found — using %s", blended_store)
        ds = xr.open_zarr(blended_store, consolidated=True).band
        return persist_with_progress_bar(ds)

    # ------------------------------------------------------------------ #
    # 2. Index-specific store                                              #
    # ------------------------------------------------------------------ #
    store_name, dataset_key = INDEX_STORE_MAP[index]
    store_path = os.path.join(local_path, store_name)
    data_exists = fs.exists(os.path.join(store_path, ".zmetadata")) or fs.exists(
        os.path.join(store_path, "zarr.json")
    )

    if data_exists:
        logging.info("Reading %s observations from cached zarr: %s", index, store_path)
        ds = xr.open_zarr(store_path, consolidated=True).band

        last_cached_date = pd.Timestamp(ds.time.values.max()).date()
        fetch_start = last_cached_date + datetime.timedelta(days=1)

        if fetch_start > last_date:
            logging.info("Cache is up to date, returning cached data...")
            return persist_with_progress_bar(ds)

        logging.info(
            "Fetching missing %s observations from %s to %s...",
            index,
            fetch_start,
            last_date,
        )
        area_missing = copy.deepcopy(area)
        area_missing.datetime_range = f"{fetch_start}/{last_date}"
        new_data = area_missing.get_dataset(
            dataset_key,
            load_config={"gridded_load_kwargs": {"resampling": "bilinear"}},
        )
        new_data.to_zarr(store_path, mode="a", append_dim="time")
        ds = xr.open_zarr(store_path, consolidated=True).band
        return persist_with_progress_bar(ds)

    else:
        logging.info(
            "No cache found — fetching full %s range from HDC STAC (%s)...",
            index,
            dataset_key,
        )
        observations = area.get_dataset(
            dataset_key,
            load_config={"gridded_load_kwargs": {"resampling": "bilinear"}},
        )
        observations.to_zarr(store_path, mode="w", consolidated=True, zarr_version=2)
        return persist_with_progress_bar(observations)


def read_triggers(params):
    triggers_path = f"{params.data_path}/data/{params.iso}/probs/aa_probabilities_triggers_pilots.csv"
    fallback_triggers_path = f"{params.data_path}/data/{params.iso}/triggers/triggers.final.{params.monitoring_year}.pilots.csv"

    if fsspec.open(triggers_path).fs.exists(triggers_path):
        triggers_df = pd.read_csv(triggers_path)
    else:
        triggers_df = pd.read_csv(fallback_triggers_path)
    return triggers_df
