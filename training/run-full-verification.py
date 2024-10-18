# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: aa-env
#     language: python
#     name: python3
# ---

# ## Run full AA drought verification
#
# #### (can be used for a more user-friendly experience or for training purposes)
#
# This notebook is intended to be self-sufficient for executing the entire workflow operationally ahead of the season and get the triggers using specific parameters and specific datasets. It is designed to be interactive, and does not require any direct interaction with another file, except for the configuration file. This will therefore be the main front-end for Anticipatory Action analysts.

# If you have not downloaded the data yet, please go to that link:
#
# https://data.earthobservation.vam.wfp.org/public-share/aa/zwe.zip

# **Import required libraries and functions**

# %cd ..

# +
import os
import logging
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config.params import Params

from AA.analytical import run_issue_verification
from AA.triggers import run_triggers_selection
from AA.helper_fns import get_coverage, read_observations, read_forecasts

from hip.analysis.aoi.analysis_area import AnalysisArea

from IPython.display import Markdown as md
# -

# **First, please define the country ISO code and the index of interest**


country = "MOZ"
index = "SPI"  # 'SPI' or 'DRYSPELL'


# Now, we will configure some parameters. Please feel free to edit the year of the last season considered. By default, it is equal to 2022. This means that for the purposes of evaluating and selecting triggers, the time series studied will end with the 2021-2022 season. This is the configuration chosen for monitoring the 2023-2024 season.
#
# Please also have a look at the `config/{iso}_config.yaml` file that contains all the defined parameters that are used in this workflow.
#
# *Note: if you change a parameter or a dataset, please make sure to manage correctly the different output paths so you don't overwrite previous results.*


params = Params(iso=country, index=index)


# ### Read data

# Let's start by getting the Zimbabwe shapefile.

# +
area = AnalysisArea.from_admin_boundaries(
    iso3=params.iso.upper(),
    admin_level=2,
    resolution=0.25,
    datetime_range=f"1981-01-01/{params.calibration_year}-06-30",
)

gdf = area.get_dataset([area.BASE_AREA_DATASET])
gdf
# -


# The next cell reads the observations dataset. Please run it directly if you have the data stored in the specified path or have access to HDC.
#
#
# *Notes for more advanced usage:*
#
# If you want to read a dataset that you have stored locally, you can give its path as an argument to `read_observations`. However, please make sure you have the right dimensions (grid spanning the whole country and daily timesteps since 1981) and that the band name is 'band'.
#
# If you want to read another dataset, that will be possible soon by specifying your key as an argument. For now, it is accessible via hip-analysis (see this [doc](https://wfp-vam.github.io/hip-analysis/reference/datasources/) to explore all the available datasets), but you need to replace the product name (*rfh_daily*) with the substitute product name in `AA.helper_fns.read_observations`.


# Observations data reading
observations = read_observations(
    area,
    f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/obs/observations.zarr",
)
observations

# As with observations, forecasts are easy to read using hip-analysis. When other datasets will be available there, it will also be possible to change these ECMWF forecasts to use another dataset from another source.
#
# If your dataset is not available via hip-analysis or if you already have stored locally the forecasts you would like to use, you can edit the path below and forecasts will be read in the analytical loop. Once again, make sure that your coordinates match those of the observations, that your forecasts are daily, and that you have 51 members. The name of the data variable must be 'tp'.


forecasts_folder_path = f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}"


# Congratulations! You've completed the part that requires the most energy during this process. Now all you have to do is run the different cells and check the results!

# ### Analytical processing
#
# The next part contains the analytical phase of the AA process.
#
# This calculates the probabilities from the forecasts and the anomalies from the observations for all the issue months and the entire time series in order to measure the ROC score with and without bias correction. Probabilities and anomalies are saved locally, so that they can be reused during the trigger selection phase.
#
# *Note1: the next cell can take several hours to run if looping through all issue months, so please make sure before launching it that you have started a new instance of JupyterHub if working on it so it doesn't get interrupted.*
#
# *Note2: if you want to re-run the workflow for issue months that you have already processed before, please delete the roc scores files in the `auc/split_by_issue` folder for the issue months of interest. Otherwise, the script will directly load the roc scores from the local files.*

# +
# Create directory for ROC scores df per issue month in case it doesn't exist
os.makedirs(
    f"{params.data_path}/data/{params.iso}/auc/split_by_issue",
    exist_ok=True,
)

# Define empty list for each issue month's ROC score dataframe
fbf_roc_issues = []

for issue in ["05", "06"]: # params.issue_months:

    forecasts = read_forecasts(
        area,
        issue,
        f"{forecasts_folder_path}/{issue}/forecasts.zarr",
    )
    logging.info(f"Completed reading of forecasts for the issue month {issue}")

    fbf_roc_issues.append(
        run_issue_verification(
            forecasts,
            observations,
            issue,
            params,
            gdf,
        )
    )
    logging.info(
        f"Completed analytical process for {params.index.upper()} over {country} country"
    )

fbf_roc = pd.concat(fbf_roc_issues)
display(fbf_roc)
# -

roc

# Let's have a look at how the computed probabilities data looks like.

xr.open_zarr(f"{forecasts_folder_path}/05/{params.index} ON/probabilities.zarr")

# We can also check how the CHIRPS-based anomalies that have been saved look like. They have been used to calculate the roc scores and will be used to select the triggers. 

xr.open_zarr(f"{forecasts_folder_path}/obs/{params.index} ON/observations.zarr")

# By running the next cell, you can save the dataframe containing the ROC scores. We commented it here so we don't overwrite the file with all the issue months with a file that only contains a few issue months. 


fbf_roc.to_csv(
   f"{params.data_path}/data/{params.iso}/auc/fbf.districts.roc.{params.index}.{params.calibration_year}.csv",
   index=False,
)

# Now we can read this dataframe locally to visualize the ROC scores.

roc = pd.read_csv(
    f"{params.data_path}/data/{params.iso}/auc/fbf.districts.roc.{params.index}.{params.calibration_year}.csv",
)

# +
display(
    md(
        f"This roc file shows {round(100 * roc.BC.sum() / len(roc), 1)} % of bias-corrected values."
    )
)
display(roc)

# Filter to include only 'AUC_best' scores and pivot the table
roc_pivot = roc.loc[(roc.district.isin(params.districts)) & (roc.category.isin(['Moderate']))].pivot_table(
    values="AUC_best", index="Index", columns="district"
)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(roc_pivot, annot=True, cmap="YlGnBu", cbar_kws={"label": "AUC_best"})
plt.title("AUC_best Scores Heatmap - Moderate")
plt.xlabel("District")
plt.ylabel("Index")
plt.show()
# -

# ### Triggers selection

# We've now come to the final part: the triggers optimization! All you have to do is execute the next cell and the calculations will take place automatically.

# Let's first define the vulnerability. We will run (if needed for at least one district) the triggers selection for two vulnerability levels: General Triggers & Non-Regret (or Emergency) Triggers.


vulnerability = "NRT"  # "GT"


run_triggers_selection(
    params, vulnerability
)


# Please have a look at the triggers dataset before any filtering by lead time / window.

xr.open_zarr(f"{params.data_path}/data/{params.iso}/triggers/triggers_{params.index}_{params.calibration_year}_{vulnerability}.zarr")

# Then, we keep the best pair for each lead time and the 4 best pairs of triggers per window of activation (in terms of Hit Rate first, and Failure Rate then).

# The triggers dataframe has been saved here: `"data/{iso}/triggers/triggers.aa.python.{index}.{calibration_year}.{vulnerability}.csv"`

triggers = pd.read_csv(
    f"{params.data_path}/data/{params.iso}/triggers/triggers.{params.index}.{params.calibration_year}.{vulnerability }.csv",
)
triggers


# ### Get and save final dataframe

# Now, you are done with the processing of one index (SPI or DRYSPELL). So you can rerun everything from the beginning with the other index. If you've already done it, you can run the next cell so it will merge all the different outputs to provide you with the very final dataframe that will be used operationally.

# The next cells merge SPI and DRYSPELL for each vulnerability level. So SPI is taken first, and if no SPI is available DRYSPELL is included. 

# Read all GT csvs
if os.path.exists(
    f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.{params.calibration_year}.GT.csv"
):
    spigt = pd.read_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.{params.calibration_year}.GT.csv"
    )
    drygt = pd.read_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.dryspell.{params.calibration_year}.GT.csv"
    )
    trigs_gt = pd.concat([spigt, drygt])
    trigs_gt["vulnerability"] = "GT"

    # Keep SPI by default and DRYSPELL when not available for GT
    gt_merged = pd.concat(
        [
            wcd.sort_values("index", ascending=False).head(4)
            for (d, c, w), wcd in trigs_gt.groupby(["district", "category", "window"])
        ]
    )

    # Save GT
    gt_merged.to_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.GT.csv",
        index=False,
    )

# Read all NRT csvs
if os.path.exists(
    f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.{params.calibration_year}.NRT.csv"
):
    spinrt = pd.read_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.{params.calibration_year}.NRT.csv"
    )
    drynrt = pd.read_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.dryspell.{params.calibration_year}.NRT.csv"
    )
    trigs_nrt = pd.concat([spinrt, drynrt])
    trigs_nrt["vulnerability"] = "NRT"

    # Keep SPI by default and DRYSPELL when not available for NRT
    nrt_merged = pd.concat(
        [
            wcd.sort_values("index", ascending=False).head(4)
            for (d, c, w), wcd in trigs_nrt.groupby(["district", "category", "window"])
        ]
    )

    # Save NRT
    nrt_merged.to_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.NRT.csv",
        index=False,
    )

# Now we read dataframes for both vulnerability levels if they exist and merge them according to the vulnerability defined for each district in the config file. 

# Read GT and NRT dataframes
if os.path.exists(
    f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.GT.csv"
):
    gt_merged = pd.read_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.GT.csv",
    )
if os.path.exists(
    f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.NRT.csv"
):
    nrt_merged = pd.read_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.NRT.csv",
    )

# Filter vulnerability based on district: merge GT and NRT
triggers_full = pd.DataFrame()
for d, v in params.districts_vulnerability.items():
    if v == "GT":
        if params.iso == "zwe":
            tmp = gt_merged.loc[
                (gt_merged.district == d) & (gt_merged.category == "Moderate")
            ]
        else:
            tmp = gt_merged.loc[gt_merged.district == d]
        triggers_full = pd.concat([triggers_full, tmp])
    else:
        if params.iso == "zwe":
            tmp = nrt_merged.loc[
                (nrt_merged.district == d) & (nrt_merged.category == "Normal")
            ]
        else:
            tmp = nrt_merged.loc[nrt_merged.district == d]
        triggers_full = pd.concat([triggers_full, tmp])


# Save final triggers file
triggers_full.to_csv(
    f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.pilots.csv",
    index=False,
)


triggers_full #.loc[triggers_full.issue_ready == 6]

# ### Visualize coverage


columns = ["W1-Mild", "W1-Moderate", "W1-Severe", "W2-Mild", "W2-Moderate", "W2-Severe"]
#columns = ["W1-Normal", "W2-Normal"]
get_coverage(triggers_full, triggers_full["district"].sort_values().unique(), columns)


# Great! The pre-season verification is complete. You can now proceed with the operational script and process the forecasts when they are ready to produce the alerts.

# Please find the final dataframe here!
#
# `data/{iso}/triggers/triggers.spi.dryspell.{params.year}.csv`
