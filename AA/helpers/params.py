import datetime
import json
import logging
import os
from dataclasses import dataclass, field

import fsspec
import hdc.algo  # noqa: F401
import numpy as np
import pandas as pd
import geopandas as gpd
import yaml
import subprocess

from numba import types
from numba.typed import Dict
from collections import OrderedDict
from dataclasses import fields as dataclass_fields

from AA.helpers.read import read_roc_file


DRYSPELL_THRESHOLD = 2.0

AGGREGATES = {
    "spi": lambda x: x.sum("time", skipna=False),
    "dryspell": lambda x: ((x <= DRYSPELL_THRESHOLD) * 1)
    .astype(np.uint8)
    .hdc.algo.lroo(),
}

S3_OPS_DATA_PATH = "s3://dev-hip-jobs-ops/anticipatory-action/data/prod"


def load_config(iso: str, cli_json: str | None = None) -> dict:
    """
    Load configuration for the given ISO3 code.

    Priority:
    1) CLI parameter --config-json (must be valid JSON)
    2) Local file: ./config/{iso}_config.yaml (YAML or JSON)
    """
    cli_json = cli_json.strip() if cli_json else None

    if cli_json is not None:
        try:
            cfg = json.loads(cli_json)
            if not isinstance(cfg, dict):
                raise ValueError("--config-json must contain a JSON object.")
            if cfg:  # non-empty dict → use it, skip file
                logging.info("Loaded config from --config-json parameter.")
                return cfg
            # empty dict → fall through to file
            logging.info("--config-json is empty, falling back to file.")
        except json.JSONDecodeError as e:
            raise ValueError(f"--config-json contains invalid JSON: {e}")

    # Fallback to file
    iso_lower = iso.lower()
    config_path = f"./config/{iso_lower}_config.yaml"
    if not fsspec.open(config_path).fs.exists(config_path):
        raise FileNotFoundError(
            f"No config provided via --config-json, and no file exists at {config_path}"
        )
    with fsspec.open(config_path, mode="rt", encoding="utf-8") as f:
        text = f.read()
    try:
        cfg = json.loads(text)
        logging.info("Loaded config from file as JSON.")
        return cfg
    except json.JSONDecodeError:
        try:
            cfg = yaml.safe_load(text)
            logging.info("Loaded config from file as YAML.")
            return cfg
        except Exception as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}")


def get_git_commit_hash():
    """
    Retrieve the current Git commit hash.

    This function attempts to obtain the full SHA hash of the current Git HEAD
    using the `git rev-parse HEAD` command. It is primarily intended for run
    traceability and experiment reproducibility.

    Returns:
        str or None: The Git commit hash if available; otherwise None if Git is
        unavailable or the current directory is not a Git repository.
    """
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def sanitize_value(v):
    """
    Convert a parameter value into a JSON-serializable form.

    This function determines whether a value should be kept for inclusion in
    a configuration snapshot and, if so, converts it into a JSON-compatible
    representation.

    Values that are callable or large, non-serializable objects (e.g.
    pandas DataFrames) are excluded.

    Args:
        v: Any Python object representing a parameter value.

    Returns:
        tuple:
            - clean_value: A JSON-serializable representation of the value, or None.
            - keep: Boolean flag indicating whether the value should be included
              in the configuration snapshot.
    """
    if callable(v):
        return None, False

    if isinstance(v, pd.DataFrame):
        return None, False

    if isinstance(v, Dict):
        return dict(v), True

    if hasattr(v, "isoformat"):
        return v.isoformat(), True

    return v, True


def ordered_params_dict(params):
    """
    Extract and sanitize parameters from a dataclass in definition order.

    This function iterates over the fields defined in a dataclass instance,
    sanitizes each value for JSON serialization, and assembles them into an
    ordered dictionary preserving the original field order.

    Certain fields (e.g. raw configuration blobs) are explicitly excluded.

    Args:
        params: A dataclass instance containing run configuration parameters.

    Returns:
        OrderedDict: An ordered mapping of parameter names to sanitized values,
        suitable for serialization.
    """
    ordered = OrderedDict()

    for f in dataclass_fields(params):
        name = f.name

        if not hasattr(params, name):
            continue

        if name == "config_json":
            continue

        value = getattr(params, name)
        clean_value, keep = sanitize_value(value)

        if keep:
            ordered[name] = clean_value

    return ordered


def save_run_config(params, script_name: str):
    """
    Save a reproducible snapshot of run configuration to persistent storage.

    The configuration snapshot includes:
      - Git commit hash
      - Run timestamp
      - All sanitized dataclass parameters in definition order

    The snapshot is written as a JSON file under:
        <output_path>/<iso>/config/config-<script_name>.json

    Storage is handled via `fsspec`, allowing support for both local and
    remote filesystems (e.g. S3).

    Args:
        params: Dataclass instance containing run parameters. Must provide
            `output_path` and `iso` attributes.
        script_name: Name of the calling script, used to uniquely identify
            the configuration file.

    Returns:
        str: The full path to the written configuration JSON file.
    """
    fs, base_path = fsspec.url_to_fs(params.output_path)

    output_dir = os.path.join(base_path, params.iso, "config")
    fs.makedirs(output_dir, exist_ok=True)

    payload = OrderedDict()

    # ---- metadata ----
    payload["git_commit"] = get_git_commit_hash()
    payload["run_time"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # ---- parameters snapshot ----
    payload.update(ordered_params_dict(params))

    out_path = os.path.join(output_dir, f"config-{script_name}.json")

    with fs.open(out_path, "w", encoding="utf-8") as f:
        # IMPORTANT: preserve insertion order for traceability
        json.dump(payload, f, indent=2)

    logging.info(
        f"Saved {params.iso} config snapshot for traceability at {out_path}"
    )

    return out_path


@dataclass
class Params:
    """
    A class to store AA parameters.

    ...

    Attributes
    ----------
    iso : str
        country ISO code
    index : str
        name of index to process: can be "SPI" or "DRYSPELL"
    config_json: str
        optional JSON string with configuration parameters, takes precedence over config file
    issue : int
        issue month: month of interest for operational script
    issue_months : list
        issue months list: list of issue months to use for triggers selection and verification
    monitoring_year : int
        first year of season to monitor operationally (e.g. 2024 for 2024/2025 season)
    vulnerability : str
        vulnerability level, can be GT (General), NRT (Non-Regret) or TBD (To Be Determined)
    calibration_year: int
        last year of calibration period used for triggers selection (e.g. 2022 for 1981-2022)
    start_monitoring: int
        month from which the monitoring starts (e.g. 5 if the first predictions are produced in May)
    aggregate : callable
        method of aggregation corresponding to index
    min_index_period : int
        minimum length of indicator periods (ON, NDJ, JFMA...)
    max_index_period : int
        maximum length of indicator periods (ON, NDJ, JFMA...)
    start_season : int
        first month of the wet season
    end_season : int
        last month of the wet season
    hist_anomaly_start : datetime.datetime
        start date of historical time series used in anomaly computation
    hist_anomaly_stop : datetime.datetime
        end date of historical time series used in anomaly computation
    districts: list
        list of districts for which we want to compute triggers
    indicators: list
        list of indicators for which we want to compute triggers
    roc_df : pd.DataFrame
        dataframe containing information about districts to bias correct
    custom_shapefile : gpd.GeoDataFrame
        geodataframe with a custom shapefile in case the one in the VAM GeoAPI is not valid
    intensity_thresholds : dict
        thresholds defining different drought intensities used in probabilities computation
    districts_vulnerability : dict
        vulnerability class of triggers for each district: regret or non-regret
    tolerance: dict
        thresholds with tolerance for each category, used to compute false alarm with tolerance
    requirements: dict
        skill requirements for GT or NRT, should include Hit Rate, Success Rate, Failure Rate, Return Period
    windows: dict
        dictionary containing two dictionaries (window1, window2) containing indicators for each window (by province or not)
    save_zarr : bool
        save (and overwrite if exists) ds (obs or probs) for future trigger choice
    data_path : str
        data path where to read input data from (should include data folder)
    output_path : str
        output path where to store intermediate and final outputs (should include data folder)
    """

    iso: str
    index: str
    config_json: str | None = None
    issue: int = None
    issue_months: list = None
    vulnerability: str = None
    monitoring_year: int = 2024
    calibration_year: int = 2022
    start_monitoring: int = 5
    aggregate: callable = field(init=None)
    min_index_period: int = 2
    max_index_period: int = 3
    start_season: int = 10
    end_season: int = 6
    hist_anomaly_start: datetime.datetime = None
    hist_anomaly_stop: datetime.datetime = datetime.datetime(2018, 12, 31)
    districts: list = field(init=None)
    indicators: list = field(init=None)
    roc_df: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    custom_shapefile: gpd.GeoDataFrame = field(
        init=False, default_factory=gpd.GeoDataFrame
    )
    intensity_thresholds: dict = field(init=None)
    districts_vulnerability: dict = field(init=None)
    tolerance: dict = field(init=False)
    requirements: dict = field(init=None)
    windows: dict = field(init=False)
    save_zarr: bool = True
    data_path: str = S3_OPS_DATA_PATH
    output_path: str = S3_OPS_DATA_PATH

    def __post_init__(self):
        self.iso = self.iso.lower()
        self.index = self.index.lower()

        config = load_config(self.iso, cli_json=self.config_json)

        # Set attributes based on the config file
        for key, value in config.items():
            setattr(self, key, value)

        # Set the aggregate method
        self.aggregate = AGGREGATES[self.index]

        # Get districts list using vulnerability dictionary to avoid duplication of definitions
        self.districts = (
            list(self.districts_vulnerability.keys())
            if self.districts_vulnerability
            else None
        )

        # Read fbf roc dataframe if exists for triggers selection
        roc_path = f"{self.data_path}/{self.iso}/auc/roc.{self.index}.csv"
        if fsspec.open(roc_path).fs.exists(roc_path):
            self.roc_df = read_roc_file(roc_path, self)

        # Check if a custom shapefile is stored in the data folder and read it if it exists
        shapefile_path = f"{self.data_path}/data/{self.iso}/{self.iso}.geojson"

        if fsspec.open(shapefile_path).fs.exists(shapefile_path):
            try:
                gdf = gpd.read_file(shapefile_path)
                expected_col = "adm2_name"
                if expected_col not in gdf.columns:
                    raise KeyError(
                        f"Expected column '{expected_col}' not found in custom shapefile. "
                        f"Available columns: {list(gdf.columns)}"
                    )
                self.custom_shapefile = gdf.set_index(expected_col)
            except Exception as e:
                raise ValueError(
                    f"Failed to load custom shapefile for iso='{self.iso}' at {shapefile_path}. "
                    f"Ensure the file is a valid GeoJSON and contains an '{expected_col}' column."
                ) from e

        # Read the tolerance thresholds and store them as a dict
        self.tolerance = Dict.empty(key_type=types.unicode_type, value_type=types.f8)
        for k, v in config["tolerance"].items():
            self.tolerance[k] = v

        # When vulnerability is not None, set the requirements based on GT or NRT criteria
        self.load_vulnerability_requirements(self.vulnerability)

        # Load the windows for the current index
        self.windows = config["windows"][self.index]

        # Extract the indicators of interest
        if type(next(iter(self.windows.values()))) is dict:
            periods = np.unique(
                list((set().union(*next(iter(self.windows.values())).values())))
            )
        else:
            periods = np.unique(list((set().union(*self.windows.values()))))
        self.indicators = [self.index + "_" + ind for ind in periods]

    def get_windows(self, window_type):
        return self.windows.get(window_type, {})

    def load_vulnerability_requirements(self, vulnerability):
        if vulnerability not in [None, "GT", "NRT", "TBD"]:
            raise ValueError("vulnerability must be one of: GT, NRT, TBD")

        self.vulnerability = vulnerability
        self.requirements = Dict.empty(key_type=types.unicode_type, value_type=types.f8)

        if vulnerability not in [None, "TBD"]:
            config = load_config(self.iso)

            config_key = "general_t" if self.vulnerability == "GT" else "non_regret_t"
            for k, v in config[config_key].items():
                self.requirements[k] = v