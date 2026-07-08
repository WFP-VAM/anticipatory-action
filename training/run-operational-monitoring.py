# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: 'Python (pixi: aa-env)'
#     language: python
#     name: aa-env
# ---

# ## Run AA operational monitoring script
#
# #### (can be used for a more user-friendly experience or for training purposes)
#
# #### (note: having run entirely the `run_full_verification` notebook is a prerequisite to run this one)
#
# This notebook reads a forecasts dataset (corresponding to a specific issue month) and computes the corresponding probabilities. These probabilities are merged with the pre-computed triggers dataframe to be displayed on the dashboard.

# **Import required libraries and functions**

import os

if os.getcwd().split("\\")[-1] != "anticipatory-action":
    os.chdir("..")
os.getcwd()

# +
import pandas as pd
from hip.analysis.analyses.drought import get_accumulation_periods
from hip.analysis.aoi.analysis_area import AnalysisArea

from AA.cli.operational import run_full_index_pipeline
from AA.helpers.params import Params
from AA.helpers.read import read_forecasts, read_observations, read_triggers
# -

# **First, please define the country ISO code, the issue month and the index of interest**


country = (
    "ISO"  # Replace with the ISO code of the country you want to run the monitoring for
)
issue = 6
index = "SPI"  # 'SPI' or 'DRYSPELL'
data_path = "./data"  # anticipatory-action directory
output_path = "./data"


# Now, we will configure some parameters. Please feel free to edit the `{country}_config.yaml` file if you need to change the *monitoring_year* or any other relevant parameter.


params = Params(
    iso=country,
    issue=issue,
    index=index,
    data_path=data_path,
    output_path=output_path,
)


# ### Read data

# Let's start by getting the shapefile.

# +
area = AnalysisArea.from_admin_boundaries(
    iso3=params.iso.upper(),
    admin_level=2,
    resolution=0.25,
    datetime_range=f"1981-01-01/{params.monitoring_year + 1}-06-30",
)

gdf = area.get_dataset([area.BASE_AREA_DATASET])
gdf
# -


# Forecasts are easy to read using hip-analysis, called within the `read_forecasts` function. A caching system allows you not to read the data from HDC in case you already have it stored locally. 


# Downscaled ECMWF forecasts data reading
forecasts = read_forecasts(
    area,
    issue,
    f"{params.data_path}/{params.iso}/zarr/{str(issue).zfill(2)}/forecasts.zarr",
)
forecasts

# Rainfall forecasts averaged over time for control member
forecasts.isel(ensemble=0).mean("time").plot.imshow()


# The next cell reads the observations dataset. Please run it directly if you have the data stored in the specified path or have access to HDC.
#
#
# *Note:*
#
# If you previously ran the `run-full-verification` notebook, you probably already have the dataset stored locally. In that case, you can give its path as an argument to `read_observations`.


# Observations data reading
area.datetime_range = f"1981-01-01/{params.calibration_year}-{str(params.end_season).zfill(2)}-30"
observations = read_observations(
    area,
    f"{params.data_path}/{params.iso}/zarr/obs",
    params.index,
)


# Now that we got all the data we need, let's read the triggers file so we can merge the probabilities with it once we have them. This triggers file corresponds to the output of the `run-full-verification` notebook if we're in the first monitoring month. Then, we read the merged dataframe that already contains the probabilities from the previous months so we add the new probabilities to the existing merged dataframe.
#
# **Note:**
#
# This means that if you want to re-run the probabilities for the first monitoring month (e.g. May, June or July), you should delete or move the existing probabilities dataframes from the probs directory.


# Read triggers file
triggers_df = read_triggers(params)
triggers_df


# ### Run forecasts processing

# Before calculating the accumulation, anomaly etc..., we need to obtain the accumulation periods we will be focusing on. These depend on the issue month of the forecasts that we are currently processing.


# Get accumulation periods (DJ, JF, FM, DJF, JFM...)
accumulation_periods = get_accumulation_periods(
    forecasts,
    params.start_season,
    params.end_season,
    params.min_index_period,
    params.max_index_period,
)
accumulation_periods


# Now we know which periods we will be computing the drought probabilities on. And this will be done in the next cell, by calling the `run_full_index_pipeline` function on each of them. That function derives the accumulation, the anomaly, performs the bias correction and obtains the probabilities.


# Compute probabilities for each accumulation period
probs_merged_dataframes = [
    run_full_index_pipeline(
        forecasts,
        observations,
        params,
        triggers_df,
        area,
        period_name,
        period_months,
    )
    for period_name, period_months in accumulation_periods.items()
]


# We reorganise the dataframes and we are ready to save them.


# +
probs_df, merged_df = zip(*probs_merged_dataframes)

probs_dashboard = pd.concat(probs_df).drop_duplicates()

merged_db = pd.concat(merged_df)
merged_db = merged_db.sort_values(["prob_ready", "prob_set"])

# Check for duplicates and raise error if found
duplicate_cols = list(merged_db.columns.difference(["prob_ready", "prob_set"]))
duplicates_count = merged_db.duplicated(subset=duplicate_cols).sum()
if duplicates_count > 0:
    raise ValueError(
        f"Data integrity error: {duplicates_count} duplicate rows found in merged trigger data. "
        f"This indicates a problem with the trigger merging process."
    )

# Perform left merge to find rows in triggers_df that don't exist in merged_db
merge_result = triggers_df.merge(
    merged_db[duplicate_cols], on=duplicate_cols, how="left", indicator=True
)

# Get only rows that exist in triggers_df but not in merged_db
new_rows_from_triggers = triggers_df[merge_result["_merge"] == "left_only"]

# Append the new rows to merged_db
if not new_rows_from_triggers.empty:
    merged_db = pd.concat([merged_db, new_rows_from_triggers], ignore_index=True)

# Assert that final merged_db has same number of rows as original triggers_df
assert len(merged_db) == len(triggers_df), (
    f"Data integrity error: Final merged_db has {len(merged_db)} rows but original "
    f"triggers_df has {len(triggers_df)} rows. Expected them to be equal."
)

merged_db
# -


# ### Save drought probabilities

# Save all probabilities
probs_dashboard.to_csv(
    f"{params.data_path}/{params.iso}/probs/aa_probabilities_{params.index}_{params.issue}.csv",
    index=False,
)

# Save probabilities merged with triggers
merged_db.sort_values(["district", "index", "category"]).to_csv(
    f"{params.data_path}/{params.iso}/probs/aa_probabilities_triggers_pilots.csv",
    index=False,
)

# ### Update PRISM dashboard (local copy only — does not touch the shared PRISM bucket)
# NOTE: this notebook only ran with params.index = <SPI or DRYSPELL, whichever this run used>.
# If the PRISM dashboard needs both indicators updated, re-run the whole notebook (or at least
# this cell) once per index — with params.index set to "SPI" and again to "DRYSPELL" — so that
# both sets of probabilities/triggers get merged into the local PRISM copy below.
from AA.helpers.prism import update_prism_dashboard

local_prism_path = (
    f"{params.data_path}/{params.iso}/prism/aa_probabilities_triggers_{params.iso}.csv"
)

merged_prism_df = update_prism_dashboard(
    params.iso,
    sorted_merged_db,
    write_path=local_prism_path,
)
