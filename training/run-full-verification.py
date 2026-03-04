# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (Pixi)
#     language: python
#     name: pixi-kernel-python3
# ---

# ## Run full AA drought verification
#
# #### (can be used for a more user-friendly experience or for training purposes)
#
# This notebook is intended to be self-sufficient for executing the entire workflow operationally ahead of the season and get the triggers using specific parameters and specific datasets. It is designed to be interactive, and does not require any direct interaction with another file, except for the configuration file. This will therefore be the main front-end for Anticipatory Action analysts.

# If you have not downloaded the data yet, please download it from the link you should have received by email.

# **Import required libraries and functions**

import os
if os.getcwd().split("\\")[-1] != "anticipatory-action":
    os.chdir("..")
os.getcwd()

# +
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr
from hip.analysis.aoi.analysis_area import AnalysisArea
from IPython.display import Markdown as md

from AA.cli.analytical import run_issue_verification
from AA.helpers.params import Params
from AA.helpers.utils import get_coverage, read_forecasts, read_observations
from AA.cli.triggers import run_triggers_selection
# -

# **First, please define the country ISO code and the index of interest**


country = "ISO"
index = "SPI"  # 'SPI' or 'DRYSPELL'
data_path = "."  # current directory (anticipatory-action)
output_path = "."


# Now, we will configure some parameters. Please feel free to edit the year of the last season considered. By default, it is equal to 2022. This means that for the purposes of evaluating and selecting triggers, the time series studied will end with the 2021-2022 season. This is the configuration chosen for monitoring the 2023-2024 season.
#
# Please also have a look at the `config/{iso}_config.yaml` file that contains all the defined parameters that are used in this workflow.
#
# *Note: if you change a parameter or a dataset, please make sure to manage correctly the different output paths so you don't overwrite previous results.*


params = Params(iso=country, index=index, data_path=data_path, output_path=output_path)


# ### Read data

# Let's start by getting the shapefile.

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

# If your dataset is already stored locally, you can edit the path below and forecasts will be read in the analytical loop. Once again, make sure that your coordinates match those of the observations, that your forecasts are daily, and that you have 51 members. The name of the data variable must be `tp` for total precipitation.


forecasts_folder_path = (
    f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}"
)


# *Congratulations!* You've completed the part that requires the most energy during this process. Now all you have to do is run the different cells and check the results!

# ### Analytical processing
#
# The next part contains the analytical phase of the AA process.
#
# This calculates the **probabilities** from the forecasts and the **anomalies (SPI)** from the observations for all the issue months and the entire time series in order to measure the **ROC score** with and without bias correction. Probabilities and anomalies are saved locally, so that they can be reused during the trigger selection phase.
#
# *Note1: the next cell can take several hours to run if looping through all issue months.*
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

for issue in ["07"]:  # params.issue_months:
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
            area,
        )
    )
    logging.info(
        f"Completed analytical process for {params.index.upper()} over {country} country"
    )

fbf_roc = pd.concat(fbf_roc_issues)
display(fbf_roc)  # noqa: F821
# -

# Let's have a look at how the computed probabilities data looks like.

xr.open_zarr(f"{forecasts_folder_path}/07/{params.index} ON/probabilities.zarr").load()

# We can also check how the CHIRPS-based anomalies that have been saved look like. They have been used to calculate the roc scores and will be used to select the triggers.

xr.open_zarr(f"{forecasts_folder_path}/obs/{params.index} ON/observations.zarr").load()

# By running the next cell, you can save the dataframe containing the ROC scores. We commented it here so we don't overwrite the file with all the issue months with a file that only contains a few issue months.


# +
#fbf_roc.to_csv(
#    f"{params.data_path}/data/{params.iso}/auc/fbf.districts.roc.{params.index}.{params.calibration_year}.csv",
#    index=False,
#)
# -

# Now we can read this dataframe locally to visualize the ROC scores.

roc = pd.read_csv(
    f"{params.data_path}/data/{params.iso}/auc/fbf.districts.roc.{params.index}.{params.calibration_year}.csv",
)

# +
display(  # noqa: F821
    md(
        f"This roc file shows {round(100 * roc.BC.sum() / len(roc), 1)} % of bias-corrected values."
    )
)
display(roc)  # noqa: F821

# Filter to include only 'AUC_best' scores and pivot the table
roc_pivot = roc.loc[
    (roc.district.isin(params.districts)) & (roc.category.isin(["Moderate"]))
].pivot_table(values="AUC_best", index="Index", columns="district")

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

# The next cell allows to define that the trigger requirements are not clear at this point. It is still "TBD", so we will compute the metrics for all our candidates.


params.load_vulnerability_requirements("TBD")


run_triggers_selection(params)


# Then, we keep the best pair for each lead time and the 4 best pairs of triggers per window of activation (in terms of Hit Rate first, and Failure Rate then).

# The triggers dataframe has been saved here for each district: `"data/{iso}/triggers/triggers_metrics/triggers_metrics_tbd_{district}.csv"`
#
# Then, these dataframes can be explored in order to evaluate the trigger performance and attempt to find suitable triggers in terms of Hit Rate, Success Rate, and False Alarm Ratio.

triggers = pd.read_csv(
    f"{params.data_path}/data/{params.iso}/triggers/triggers_metrics/triggers_metrics_tbd_{params.districts[0]}.csv",
)
triggers


# Great! Now you can go through the trigger selection process: using the set-up tool, or by implementing a simple routine that carries out the filtering and ranking jobs. This type of function should be added soon in the codebase to facilitate this work. The pre-season verification and setup is then complete. 
#
# You can therefore proceed with the operational script and process the forecasts when they are ready to produce the alerts.

# Please store the final dataframe here!
#
# `data/{iso}/triggers/triggers.final.{monitoring_year}.pilots.csv`
