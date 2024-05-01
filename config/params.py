import datetime
import os
from dataclasses import dataclass, field

import hdc.algo
import numpy as np
import pandas as pd

from helper_fns import read_fbf_districts

DRYSPELL_THRESHOLD = 2.0

AGGREGATES = {
    "spi": lambda x: x.sum("time", skipna=False),
    "dryspell": lambda x: ((x <= DRYSPELL_THRESHOLD) * 1)
    .astype(np.uint8)
    .hdc.algo.lroo(),
}


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
    year : int
        first year of season of interest (e.g. 2022 for 2022/2023 season)
    aggregate : callable
        method of monthly of aggregation corresponding to index
    min_index_period : int
        minimum length of indicator periods (ON, NDJ, JFMA...)
    max_index_period : int
        maximum length of indicator periods (ON, NDJ, JFMA...)
    calibration_start : datetime.datetime
        start date of historical time series used in anomaly computation
    calibration_stop : datetime.datetime
        end date of historical time series used in anomaly computation
    save_zarr : bool
        save (and overwrite if exists) ds (obs or probs) for future trigger choice
    districts: list
        list of districts for which we want to compute triggers
    fbf_districts_df : pd.DataFrame
        dataframe containing information about districts to bias correct
    intensity_thresholds : dict
        thresholds defining different drought intensities used in probabilities computation
    districts_vulnerability : dict
        vulnerability class of triggers for each district: regret or non-regret
    """

    iso: str
    index: str
    issue: int or list = None
    year: int = 2022
    aggregate: callable = field(init=None)
    min_index_period: int = 2
    max_index_period: int = 3
    start_season: int = 10
    end_season: int = 6
    calibration_start: datetime.datetime = None
    calibration_stop: datetime.datetime = datetime.datetime(2018, 12, 31)
    save_zarr: bool = True
    districts: list = field(init=None)
    fbf_districts_df: pd.DataFrame = field(init=None)
    intensity_thresholds: dict = field(init=None)
    districts_vulnerability: dict = field(init=None)

    def __post_init__(self):
        self.iso = self.iso.upper()
        self.index = self.index.lower()
        self.aggregate = AGGREGATES[self.index]
        if self.iso == "MOZ":
            self.intensity_thresholds = {"Severo": -1, "Moderado": -0.85, "Leve": -0.68}
            self.districts_vulnerability = {
                "Chiure": "GT",  # "GT",
                "Caia": "GT",  # "NRT",
                "Changara": "GT",  # "GT",
                "Chemba": "GT",  # "GT",
                "Chibuto": "GT",  # "NRT",
                "Chicualacuala": "GT",  # "NRT",
                "Guija": "GT",  # "NRT",
                "Mabalane": "GT",  # "NRT",
                "Mapai": "GT",  # "NRT",
                "Marara": "GT",  # "GT",
                "Massingir": "GT",  # "NRT",
            }
            self.districts = self.districts_vulnerability.keys()
        else:
            self.intensity_thresholds = {"Severe": -1, "Moderate": -0.85, "Mild": -0.68}
        if self.issue is None:  # analytical / triggers
            self.issue = ["05", "06", "07", "08", "09", "10", "11", "12", "01", "02"]
        if os.path.exists(
            f"data/{self.iso}/outputs/Districts_FbF/{self.index}/fbf.districts.roc.{self.index}.{self.year}.txt"
        ):
            self.fbf_districts_df = read_fbf_districts(
                f"data/{self.iso}/outputs/Districts_FbF/{self.index}/fbf.districts.roc.{self.index}.{self.year}.txt",
                self,
            )
