# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: hdc
#     language: python
#     name: conda-env-hdc-py
# ---

# ### Check *triggers* differences after dates alignment by comparing coverage with reference outputs - Mozambique

# %cd ../..

# +
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# +
from config.params import Params

from AA.helper_fns import get_coverage
# -

# Get parameters 

country = "MOZ"

params = Params(iso=country, index='SPI')

# Get triggers

# +
refGT = pd.read_csv(f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.GT.csv")
refNRT = pd.read_csv(f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.NRT.csv")

ref = pd.read_csv(f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.pilots.csv")

# +
trigsGT = pd.read_csv(f"{params.data_path}/data/{params.iso}_align_dates/triggers/triggers.spi.dryspell.{params.calibration_year}.GT.csv")
trigsNRT = pd.read_csv(f"{params.data_path}/data/{params.iso}_align_dates/triggers/triggers.spi.dryspell.{params.calibration_year}.NRT.csv")

trigs = pd.read_csv(f"{params.data_path}/data/{params.iso}_align_dates/triggers/triggers.spi.dryspell.{params.calibration_year}.pilots.csv")
# -

# Compare coverages of Python GT and NRT triggers

# GENERAL TRIGGERS - NEW
columns = ["W1-Mild", "W1-Moderate", "W1-Severe", "W2-Mild", "W2-Moderate", "W2-Severe"]
get_coverage(trigsGT, params.districts, columns)

# GENERAL TRIGGERS - OLD
get_coverage(refGT, params.districts, columns)

# NON-REGRET TRIGGERS - NEW
get_coverage(trigsNRT, params.districts, columns)

# NON-REGRET TRIGGERS - OLD
get_coverage(refNRT, params.districts, columns)

# Get coverage of final output

get_coverage(trigs, params.districts, columns)


