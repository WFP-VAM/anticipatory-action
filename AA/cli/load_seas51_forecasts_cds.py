#!/usr/bin/env python3
# /// script
# dependencies = [
#   "cdsapi",
#   "xarray",
#   "numpy",
#   "pandas",
#   "rioxarray",
#   "odc-geo",
#   "rasterio",
#   "click",
#   "dask[distributed]",
#   "zarr",
#   "netcdf4",
# ]
# ///

"""
End-to-end loader for ECMWF SEAS5 / SEAS5.1 daily precipitation forecasts
using odc.geo.xr_reproject (HIP-compatible semantics).

WHAT THIS SCRIPT DOES
---------------------
- Downloads missing SEAS5 issue-year NetCDFs from CDS
- Uses a LARGE CDS bbox (for correct interpolation support)
- Normalizes SEAS5 time (forecast_reference_time + lead time)
- Reprojects + clips using a target GeoBox (like hip-analysis)
- Converts cumulative precipitation to daily values
- Stores the result as Zarr in an issue-month folder (e.g. 03/forecasts.zarr)

IMPORTANT DESIGN CHOICES
------------------------
- Reprojection is done BEFORE any spatial clipping (matches get_dataset)
- `issue` is a coordinate, NOT a dimension (memory-safe)
- Edge differences vs get_dataset are expected and acceptable

REQUIREMENTS
------------
- cdsapi
- xarray
- numpy
- pandas
- rioxarray
- odc-geo
- rasterio
- click
- dask[distributed]

USAGE
-----
Example (test mode, last 3 years, first 5 ensemble members):

    python load_seas51_forecasts_cds.py \
        --country tza \
        --output-dir ./seas5_cache \
        --issue-month 3 \
        --start-year 2024 \
        --end-year 2026
        
Or if using uv to manage the dependencies:
    
    uv run load_seas51_forecasts_cds.py \
      --country tza \
      --output-dir ./seas5_cache \
      --issue-month 3 \
      --start-year 2024 \
      --end-year 2026 \
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import logging

import cdsapi
import click
import rioxarray
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path
from rasterio.crs import CRS
from odc.geo import GeoBox
from odc.geo.xr import xr_reproject

from dask.distributed import Client


ANALYSIS_AREAS = {
    "tza": {
        "bbox": (29.3414, -11.7612, 40.4446, -0.9844),
    },
    "mwi": {
        "bbox": (32.625, -17.13, 35.9, -9.3),
    },
}


# ==============================================================================
# CONFIGURATION
# ==============================================================================

CACHE_DIR = Path("./seas5_cache")  # edit output dir
CACHE_DIR.mkdir(exist_ok=True)

# CDS parameters
DATASET = "seasonal-original-single-levels"
VARIABLE_CDS = "total_precipitation"
VARIABLE_OUT = "tp"

MODEL = "ecmwf"
SYSTEM = "51"

# Forecast properties
EXPECTED_FORECAST_LENGTH_MONTHS = 7
TARGET_RESOLUTION_DEG = 0.25  # degrees

# Large bbox for CDS reads (safe margin around East Africa)
CDS_BBOX = (
    20.0,  # min_lon
    -20.0,  # min_lat
    55.0,  # max_lon
    20.0,  # max_lat
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def list_issue_years(start_year: int = 1981) -> list[int]:
    """Return all SEAS5 issue years from start_year to present."""
    return list(range(start_year, datetime.utcnow().year + 1))


def download_seas5_issue_year(
    year: int,
    issue_month: int,
    bbox: tuple[float, float, float, float],
    target_file: Path,
):
    """
    Download one SEAS5 issue year from CDS.

    bbox: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
    """
    logging.info(f"Downloading SEAS5 from CDS: year={year}, issue={issue_month:02d}")

    client = cdsapi.Client()

    client.retrieve(
        DATASET,
        {
            "originating_centre": MODEL,
            "system": SYSTEM,
            "variable": [VARIABLE_CDS],
            "year": [str(year)],
            "month": [f"{issue_month:02d}"],
            "day": ["01"],
            "leadtime_hour": [
                "24",
                "48",
                "72",
                "96",
                "120",
                "144",
                "168",
                "192",
                "216",
                "240",
                "264",
                "288",
                "312",
                "336",
                "360",
                "384",
                "408",
                "432",
                "456",
                "480",
                "504",
                "528",
                "552",
                "576",
                "600",
                "624",
                "648",
                "672",
                "696",
                "720",
                "744",
                "768",
                "792",
                "816",
                "840",
                "864",
                "888",
                "912",
                "936",
                "960",
                "984",
                "1008",
                "1032",
                "1056",
                "1080",
                "1104",
                "1128",
                "1152",
                "1176",
                "1200",
                "1224",
                "1248",
                "1272",
                "1296",
                "1320",
                "1344",
                "1368",
                "1392",
                "1416",
                "1440",
                "1464",
                "1488",
                "1512",
                "1536",
                "1560",
                "1584",
                "1608",
                "1632",
                "1656",
                "1680",
                "1704",
                "1728",
                "1752",
                "1776",
                "1800",
                "1824",
                "1848",
                "1872",
                "1896",
                "1920",
                "1944",
                "1968",
                "1992",
                "2016",
                "2040",
                "2064",
                "2088",
                "2112",
                "2136",
                "2160",
                "2184",
                "2208",
                "2232",
                "2256",
                "2280",
                "2304",
                "2328",
                "2352",
                "2376",
                "2400",
                "2424",
                "2448",
                "2472",
                "2496",
                "2520",
                "2544",
                "2568",
                "2592",
                "2616",
                "2640",
                "2664",
                "2688",
                "2712",
                "2736",
                "2760",
                "2784",
                "2808",
                "2832",
                "2856",
                "2880",
                "2904",
                "2928",
                "2952",
                "2976",
                "3000",
                "3024",
                "3048",
                "3072",
                "3096",
                "3120",
                "3144",
                "3168",
                "3192",
                "3216",
                "3240",
                "3264",
                "3288",
                "3312",
                "3336",
                "3360",
                "3384",
                "3408",
                "3432",
                "3456",
                "3480",
                "3504",
                "3528",
                "3552",
                "3576",
                "3600",
                "3624",
                "3648",
                "3672",
                "3696",
                "3720",
                "3744",
                "3768",
                "3792",
                "3816",
                "3840",
                "3864",
                "3888",
                "3912",
                "3936",
                "3960",
                "3984",
                "4008",
                "4032",
                "4056",
                "4080",
                "4104",
                "4128",
                "4152",
                "4176",
                "4200",
                "4224",
                "4248",
                "4272",
                "4296",
                "4320",
                "4344",
                "4368",
                "4392",
                "4416",
                "4440",
                "4464",
                "4488",
                "4512",
                "4536",
                "4560",
                "4584",
                "4608",
                "4632",
                "4656",
                "4680",
                "4704",
                "4728",
                "4752",
                "4776",
                "4800",
                "4824",
                "4848",
                "4872",
                "4896",
                "4920",
                "4944",
                "4968",
                "4992",
                "5016",
                "5040",
                "5064",
                "5088",
                "5112",
                "5136",
                "5160",
            ],
            "area": [
                bbox[3],  # North
                bbox[0],  # West
                bbox[1],  # East
                bbox[2],  # South
            ],
            "data_format": "netcdf",
        },
        target_file.as_posix(),
    )


def build_target_geobox(
    bbox: tuple[float, float, float, float],
    resolution: float = 0.25,
) -> GeoBox:
    return GeoBox.from_bbox(
        bbox,
        crs=CRS.from_epsg(4326),
        resolution=resolution,
    )


def normalize_seas5_time(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize SEAS5 CDS dataset to RFH-compatible layout.

    - Build `time` explicitly as:
        forecast_reference_time + forecast_period
    - Keep `issue` as a SCALAR COORDINATE (NOT a dimension)
    - Ensure `time` is the only temporal dimension
    """

    # ------------------------------------------------------------------
    # Forecast reference time (seconds since Unix epoch)
    # ------------------------------------------------------------------
    frt = pd.to_datetime(ds.forecast_reference_time.values[0], unit="s")

    # ------------------------------------------------------------------
    # Forecast lead time in HOURS → absolute valid time
    # ------------------------------------------------------------------
    lead_hours = ds.forecast_period.values.astype("float64")
    time = frt + pd.to_timedelta(lead_hours, unit="h")

    # Assign proper time coordinate
    ds = ds.assign_coords(time=("forecast_period", time))

    # ------------------------------------------------------------------
    # Issue as SCALAR coordinate (critical: NOT a dimension)
    # ------------------------------------------------------------------
    ds = ds.assign_coords(issue=frt.strftime("%Y-%m"))

    # ------------------------------------------------------------------
    # Drop SEAS5-specific bookkeeping variables
    # ------------------------------------------------------------------
    ds = ds.drop_vars(
        ["forecast_reference_time", "forecast_period", "valid_time"],
        errors="ignore",
    )

    # ------------------------------------------------------------------
    # Rename forecast_period → time (single temporal axis)
    # ------------------------------------------------------------------
    ds = ds.swap_dims({"forecast_period": "time"})

    # ------------------------------------------------------------------
    # Safety: ensure monotonic time (MARS is not always strict)
    # ------------------------------------------------------------------
    ds = ds.sortby("time")

    return ds


def convert_cumsum_to_daily(tp: xr.DataArray) -> xr.DataArray:
    """
    Convert cumulative precipitation forecasts to daily values,
    reproducing RFH operational logic exactly.
    """

    # Convert cumulative to daily using diff grouped by issue
    tp_daily = tp.groupby("issue").apply(xr.DataArray.diff, dim="time")

    # Time correction:
    # - 1 day: accumulation over previous 24 hours
    # - 1 day: shift introduced by diff
    tp_daily["time"] = [
        pd.to_datetime(t) - pd.Timedelta(2, "d") for t in tp_daily.time.values
    ]

    # Drop a few trailing dates beyond the expected forecast horizon
    tp_daily = tp_daily.where(
        tp_daily.time.dt.month
        != (tp_daily.time.dt.month[0] + EXPECTED_FORECAST_LENGTH_MONTHS) % 12,
        drop=True,
    )

    # Convert units from meters to millimeters
    tp_daily = tp_daily * 1000.0
    tp_daily.attrs["units"] = "mm"

    return tp_daily


def build_target_geobox(
    bbox: tuple[float, float, float, float],
    resolution: float = TARGET_RESOLUTION_DEG,
) -> GeoBox:
    """
    Build a target GeoBox in EPSG:4326 at the requested resolution.
    """
    return GeoBox.from_bbox(
        bbox,
        crs=CRS.from_string("EPSG:4326"),
        resolution=resolution,
    )


def prepare_for_zarr(tp: xr.DataArray) -> xr.DataArray:
    """
    Prepare SEAS5 daily forecasts for persistent Zarr storage.

    - Rename ensemble dimension
    - Ensure dimension order
    - Ensure clean encoding
    """
    # Squeeze forecast_reference_tim dim (size 1)
    tp = tp.squeeze("forecast_reference_time")

    # Rename ensemble dimension
    if "number" in tp.dims:
        tp = tp.rename({"number": "ensemble"})

    # Ensure canonical dimension order
    tp = tp.transpose("time", "ensemble", "latitude", "longitude")

    # Clean encoding (important for Zarr stability)
    tp.encoding.clear()

    return tp


def store_forecasts_zarr(
    da: xr.DataArray,
    issue_month: int,
    root_dir: Path,
    overwrite: bool = True,
    chunks: dict | None = None,
):
    """
    Store SEAS5 daily forecasts as Zarr in issue-month subfolders.

    Directory layout:
      root_dir/
        01/forecasts.zarr
        02/forecasts.zarr
        ...
        12/forecasts.zarr

    Parameters
    ----------
    da : xr.DataArray
        Forecast data with dims (time, ensemble, latitude, longitude)
    issue_month : int
        Issue month (1–12)
    root_dir : Path
        Base directory for storage
    overwrite : bool
        If True, overwrite existing Zarr store
    chunks : dict, optional
        Dask chunking to apply before writing
        e.g. {"time": 30, "ensemble": 5}
    """

    issue_dir = root_dir / f"{issue_month:02d}"
    issue_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = issue_dir / "forecasts.zarr"

    # ------------------------------------------------------------------
    # Ensure canonical dimension names & order
    # ------------------------------------------------------------------
    if "number" in da.dims:
        da = da.rename({"number": "ensemble"})

    da = da.transpose("time", "ensemble", "latitude", "longitude")

    da = da.rename("tp")

    # ------------------------------------------------------------------
    # Chunking (VERY important for memory & performance)
    # ------------------------------------------------------------------
    if chunks is None:
        chunks = {
            "time": 30,  # ~1 month
            "ensemble": -1,
            "latitude": -1,
            "longitude": -1,
        }

    da = da.chunk(chunks)

    # ------------------------------------------------------------------
    # Clean encoding (prevents Zarr / NetCDF weirdness)
    # ------------------------------------------------------------------
    da.encoding.clear()

    # ------------------------------------------------------------------
    # Write Zarr
    # ------------------------------------------------------------------
    mode = "w" if overwrite else "a"

    logging.info(f"Writing forecasts to Zarr → {zarr_path}")

    da.to_zarr(
        zarr_path,
        mode=mode,
        consolidated=True,
        compute=True,
    )


# ==============================================================================
# MAIN LOADER
# ==============================================================================


def load_seas5_daily_time_series(
    analysis_bbox: tuple[float, float, float, float],
    issue_month: int,
    start_year: int,
    end_year: int,
    test_mode: bool = True,
) -> xr.DataArray:
    """
    Load ECMWF SEAS5 daily precipitation forecasts.

    Parameters
    ----------
    analysis_bbox : tuple
        Small bbox used for the final GeoBox (EPSG:4326)
    issue_month : int
        Issue month (1–12)
    start_year, end_year : int
        Year range to load
    test_mode : bool
        If True:
          - keep only last 3 years
          - keep first 5 ensemble members
    """

    # --------------------------------------------------------------
    # Folder: 03/ instead of issue_03/
    # --------------------------------------------------------------
    issue_dir = CACHE_DIR / f"{issue_month:02d}"
    issue_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------------
    # Target GeoBox (controls reprojection + clipping)
    # --------------------------------------------------------------
    target_geobox = build_target_geobox(analysis_bbox)

    yearly_arrays: list[xr.DataArray] = []

    years = list(range(start_year, end_year + 1))
    if test_mode:
        years = years[-3:]  # last 3 years only

    for year in years:
        logging.info(f"Processing year {year} (issue {issue_month:02d})")

        fname = issue_dir / f"seas5_{year}_issue{issue_month:02d}.nc"

        # ----------------------------------------------------------
        # Download missing years only
        # ----------------------------------------------------------
        if not fname.exists():
            download_seas5_issue_year(
                year=year,
                issue_month=issue_month,
                bbox=CDS_BBOX,
                target_file=fname,
            )
        else:
            logging.info(f"Using cached file: {fname.name}")

        # ----------------------------------------------------------
        # Load + normalize time
        # ----------------------------------------------------------
        ds = xr.open_dataset(fname, decode_cf=False)

        logging.info("Normalizing SEAS5 time")
        ds = normalize_seas5_time(ds)

        da = ds[VARIABLE_OUT]

        # ----------------------------------------------------------
        # Register CRS & spatial dims (required for xr_reproject)
        # ----------------------------------------------------------
        logging.info("Reprojecting and clipping using target GeoBox")
        if da.rio.crs is None:
            da = da.rio.write_crs("EPSG:4326")

        da = da.rio.set_spatial_dims(
            x_dim="longitude",
            y_dim="latitude",
            inplace=False,
        )

        # ----------------------------------------------------------
        # Reproject FIRST (this matches get_dataset)
        # This step also CLIPS to the GeoBox
        # ----------------------------------------------------------
        da = xr_reproject(
            da,
            target_geobox,
            resampling="bilinear",
        )

        # ----------------------------------------------------------
        # Testing restriction: first 3 ensemble members
        # ----------------------------------------------------------
        if test_mode:
            da = da.isel(number=slice(0, 3))

        yearly_arrays.append(da)

    # --------------------------------------------------------------
    # Concat on time ONLY
    # --------------------------------------------------------------
    tp = xr.concat(yearly_arrays, dim="time")

    # --------------------------------------------------------------
    # Convert cumulative → daily (AFTER reprojection)
    # --------------------------------------------------------------
    logging.info("Converting cumulative precipitation to daily values")
    tp_daily = convert_cumsum_to_daily(tp)

    # --------------------------------------------------------------
    # Prepare + store
    # --------------------------------------------------------------
    logging.info("Writing Zarr output")
    tp_daily = prepare_for_zarr(tp_daily)

    store_forecasts_zarr(
        tp_daily,
        issue_month=issue_month,
        root_dir=CACHE_DIR,
        overwrite=True,
    )

    return tp_daily


@click.command()
@click.option(
    "--country",
    type=str,
    required=True,
    help="Country (ISO3 code).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Base output directory for SEAS5 data and Zarr store.",
)
@click.option(
    "--issue-month",
    type=int,
    required=True,
    help="SEAS5 issue month (1–12).",
)
@click.option(
    "--start-year",
    type=int,
    required=True,
    help="First issue year to load.",
)
@click.option(
    "--end-year",
    type=int,
    required=True,
    help="Last issue year to load.",
)
def main(
    country: str, output_dir: Path, issue_month: int, start_year: int, end_year: int
):
    """
    CLI entry point.
    """

    logging.info("Starting Dask client (threads-only)")
    client = Client(
        processes=False,
        threads_per_worker=4,
        n_workers=1,
    )
    logging.info(f"Dask dashboard: {client.dashboard_link}")

    # Tanzania bbox
    analysis_bbox = ANALYSIS_AREAS[country]["bbox"]

    global CACHE_DIR
    CACHE_DIR = output_dir
    CACHE_DIR.mkdir(exist_ok=True, parents=True)

    logging.info(
        f"Running SEAS5 loader | issue={issue_month:02d} | "
        f"years={start_year}–{end_year}"
    )

    precip_daily = load_seas5_daily_time_series(
        analysis_bbox=analysis_bbox,
        issue_month=issue_month,
        start_year=start_year,
        end_year=end_year,
        test_mode=False,  # ← KEEP 3 first ensemble members to minimize running time for testing
    )

    logging.info("Processing finished successfully")
    logging.info(precip_daily)


if __name__ == "__main__":
    main()
