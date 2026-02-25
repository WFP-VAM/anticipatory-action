import datetime
import json
import logging
import os
from dataclasses import dataclass, field

import fsspec
import hdc.algo  # noqa: F401
import numpy as np
import pandas as pd
import yaml
from numba import types
from numba.typed import Dict

from AA.helpers.utils import read_fbf_districts

DRYSPELL_THRESHOLD = 2.0

AGGREGATES = {
    "spi": lambda x: x.sum("time", skipna=False),
    "dryspell": lambda x: ((x <= DRYSPELL_THRESHOLD) * 1)
    .astype(np.uint8)
    .hdc.algo.lroo(),
}

S3_OPS_DATA_PATH = "s3://wfp-ops-userdata/amine.barkaoui/aa"


def load_config(iso: str) -> dict:
    """
    Load configuration for the given ISO3 code.

    Priority:
    1) AA_CONFIG_JSON environment variable (must contain a JSON document).
    2) Local file: ./config/{iso}_config.yaml (or .yml). Supports YAML or JSON content.
    Args:
        iso: ISO3 country code, e.g. "TZA"

    Returns:
        dict: Parsed configuration.

    Raises:
        FileNotFoundError: If neither env nor a file exists.
        ValueError: If parsing fails for available sources.
    """
    # --- 1) Env var ---
    env_val = os.environ.get("AA_CONFIG_JSON")
    if env_val is not None:
        try:
            cfg = json.loads(env_val)
            if not isinstance(cfg, dict):
                raise ValueError("AA_CONFIG_JSON must represent a JSON object (dict).")
            logging.info("Loaded config from environment variable AA_CONFIG_JSON.")
            return cfg
        except json.JSONDecodeError as e:
            # Warn and continue to file fallback
            logging.warning("AA_CONFIG_JSON is set but contains invalid JSON: %s", e)

    # --- 2) File fallback ---
    iso_lower = iso.lower()
    config_path = f"./config/{iso_lower}_config.yaml"
    config_file = (
        config_path if fsspec.open(config_path).fs.exists(config_path) else None
    )
    if not config_file:
        msg = (
            f"No configuration provided via AA_CONFIG_JSON and no file found. "
            f"Checked: {config_path}"
        )
        logging.error(msg)
        raise FileNotFoundError(msg)

    with fsspec.open(config_file, mode="rt", encoding="utf-8") as f:
        text = f.read()

    # Try JSON first (some teams store JSON in .yml files), then YAML.
    try:
        cfg = json.loads(text)
        if not isinstance(cfg, dict):
            raise ValueError(f"{config_file} must represent a JSON object (dict).")
        logging.info("Loaded config from file (JSON): %s", config_file)
        return cfg
    except json.JSONDecodeError:
        try:
            cfg = yaml.safe_load(text)
            if not isinstance(cfg, dict):
                raise ValueError(f"{config_file} must represent a mapping (dict).")
            logging.info("Loaded config from file (YAML): %s", config_file)
            return cfg
        except Exception as e:
            logging.error("Failed to parse YAML in %s: %s", config_file, e)
            raise ValueError(f"Invalid YAML in {config_file}: {e}") from e


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
    fbf_districts_df : pd.DataFrame
        dataframe containing information about districts to bias correct
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
    fbf_districts_df: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
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

        config = load_config(self.iso)

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
        fbf_districts_path = f"{self.data_path}/data/{self.iso}/auc/fbf.districts.roc.{self.index}.2022.csv"
        if fsspec.open(fbf_districts_path).fs.exists(fbf_districts_path):
            self.fbf_districts_df = read_fbf_districts(fbf_districts_path, self)

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
        self.indicators = [self.index + " " + ind for ind in periods]

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
