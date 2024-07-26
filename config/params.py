import os
import yaml
from dataclasses import dataclass, field

import hdc.algo
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd

from numba import types
from numba.typed import Dict

from AA.helper_fns import read_fbf_districts

DRYSPELL_THRESHOLD = 2.0

AGGREGATES = {
    "spi": lambda x: x.sum("time", skipna=False),
    "dryspell": lambda x: ((x <= DRYSPELL_THRESHOLD) * 1)
    .astype(np.uint8)
    .hdc.algo.lroo(),
}


def load_config(iso):
    config_file = f"./config/{iso}_config.yaml"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Configuration file for {iso} not found.")
    

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
    save_zarr : bool
        save (and overwrite if exists) ds (obs or probs) for future trigger choice
    data_path : str
        output path where to store intermediate and final outputs (should include data folder)
    districts: list
        list of districts for which we want to compute triggers
    fbf_districts_df : pd.DataFrame
        dataframe containing information about districts to bias correct
    intensity_thresholds : dict
        thresholds defining different drought intensities used in probabilities computation
    districts_vulnerability : dict
        vulnerability class of triggers for each district: regret or non-regret
    tolerance: dict
        thresholds with tolerance for each category, used to compute false alarm with tolerance
    general_t: dict
        skill requirements for General Triggers in terms of Hit Rate, Success Rate, Failure Rate, Return Period
    non_regret_t: dict
        skill requirements for Non Regret Triggers in terms of Hit Rate, Success Rate, Failure Rate, Return Period
    windows: dict
        dictionary containing two dictionaries (window1, window2) containing indicators for each window (by province or not)
    """

    iso: str
    index: str
    issue: int = None 
    issue_months: list = None
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
    save_zarr: bool = True
    data_path: str = "."
    districts: list = field(init=None)
    fbf_districts_df: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    intensity_thresholds: dict = field(init=None)
    districts_vulnerability: dict = field(init=None)
    tolerance: dict = field(init=False)
    general_t: dict = field(init=False)
    non_regret_t: dict = field(init=False)
    windows: dict = field(init=False)

    def __post_init__(self):
        self.iso = self.iso.lower()
        self.index = self.index.lower()

        config = load_config(self.iso)

        # Set attributes based on the config file
        for key, value in config.items():
            setattr(self, key, value)

        # +1 to monitoring year if operational issue month falls in 2nd part of cross-year season
        if isinstance(self.issue, int) and self.issue < self.start_monitoring:
            self.monitoring_year += 1

        # Set the aggregate method      
        self.aggregate = AGGREGATES[self.index]

        # Get districts list using vulnerability dictionary to avoid duplication of definitions
        self.districts = self.districts_vulnerability.keys()

        # Read fbf roc dataframe if exists for triggers selection
        fbf_districts_path = f"{self.data_path}/data/{self.iso}/auc/fbf.districts.roc.{self.index}.2022.csv"
        if os.path.exists(fbf_districts_path):
            self.fbf_districts_df = read_fbf_districts(fbf_districts_path, self)

        # Convert tolerance, general_t, and non_regret_t to typed dicts
        self.tolerance = Dict.empty(key_type=types.unicode_type, value_type=types.f8)
        for k, v in config["tolerance"].items():
            self.tolerance[k] = v

        self.general_t = Dict.empty(key_type=types.unicode_type, value_type=types.f8)
        for k, v in config["general_t"].items():
            self.general_t[k] = v

        self.non_regret_t = Dict.empty(key_type=types.unicode_type, value_type=types.f8)
        for k, v in config["non_regret_t"].items():
            self.non_regret_t[k] = v

        # Load the windows for the current index
        self.windows = config["windows"][self.index]

    def get_windows(self, window_type):
        return self.windows.get(window_type, {})