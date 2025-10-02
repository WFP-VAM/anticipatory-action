# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %cd ../

# +
import pandas as pd

from AA.analytical import run_issue_verification
from AA.helper_fns import (
    read_forecasts_locally,
    read_observations_locally,
)
from AA.triggers import run as run_triggers
from config.params import Params

# -

# ## Evaluate full AA drought pipeline using datasets/parameters of interest
#
# This notebook is intended to be self-sufficient for either: executing the entire workflow operationally ahead of the season or evaluating the system's performance and comparing it with the reference system (currently based on ecmwf forecasts downscaled with bilinear interpolation, chirps data resampled to 25km as observations, and bias correction performed with quantile mapping). It is designed to be interactive, and does not require any direct interaction with another file. This will therefore be the main front-end for Anticipatory Action analysts.
#
# Currently, this works for Mozambique and Zimbabwe. If there is a need to process another country, it is preferable for the time being to contact the Data Science team to configure a few parameters such as the wet season or the intensity thresholds that need to be specified for each new country.

# **First, please define the country ISO code and the index of interest**

country = "MOZ"  # 'ZWE'
index = "SPI"  # 'DRYSPELL'

# Now, we will configure some parameters. Please feel free to edit the year of the last season considered. By default, it is equal to 2022. This means that for the purposes of evaluating and selecting triggers, the time series studied will end with the 2021-2022 season. This is the configuration chosen for monitoring the 2023-2024 season.
#
# *Note: if you change a parameter or a dataset, please make sure to manage correctly the different output paths so you don't overwrite previous results.*

# +
params = Params(iso=country, index=index)

params.year = 2022  # first year of season of interest (e.g. 2022 for 2022/2023 season)
params.save_zarr = False  # set to True if need to run triggers script afterwards
# -

# ### Read data

# The next cell reads the observations dataset. If you want to read another dataset that is accessible via hip-analysis (see this [doc](https://wfp-vam.github.io/hip-analysis/reference/datasources/) to explore all the available datasets), you can simply replace the product name (*rfh_daily*) with the substitute product name.
#
# If you want to read a dataset that you have stored locally, you can use this command `xr.open_dataset(path, engine='netcdf4'`). However, please make sure you have the right dimensions (grid spanning the whole country and daily timesteps since 1981) and that the band name is 'precip'.

# TODO replace by HDC chirps by default
observations = read_observations_locally(f"data/{params.iso}/chirps")
observations


# As with observations, forecasts are easy to read thanks to hip-analysis (TODO). But it is also possible to change these forecasts using another dataset from another source. If this dataset is not available from hip-analysis, it is probably available from another API, such as CDS for example. In this case, you can download the dataset locally and read it here using `xr.concat(path, engine='netcdf4')`. Once again, make sure that your coordinates match those of the observations, that your forecasts are daily, and that you have 51 members. The name of the data variable must be 'tp'. Otherwise you can rename it as follows: `xx = xx.rename_vars({'current_name': 'tp'})`.


def get_forecasts(issue):
    # Please edit forecasts data reading here. As each issue month is read separately, please note
    # the use of {issue} (with issue being "01", "02" ...) so this may need to be slightly adapted
    # when reading a different dataset.

    return read_forecasts_locally(
        f"data/{params.iso}/forecast/Moz_SAB_tp_ecmwf_{issue}/*.nc"
    )


# Congratulations! You've completed the part that requires the most energy during this process. Now all you have to do is run the different cells and check the results!

# ### Analytical processing
#
# The next part contains the analytical phase of the AA process, also referred to as verification. It calculates the probabilities from the forecasts and the anomalies from the observations for all the issue months and the entire time series in order to measure the ROC score with and without bias correction. Probabilities and anomalies are saved locally (or not, depending on the save_zarr parameter value), so that they can be reused during the trigger selection phase.
#
# *Note: the next cell can take several hours to run, please make sure before launching it that you have started a new instance of JupyterHub so it doesn't get interrupted.*

# +
fbf_roc_issues = [
    run_issue_verification(
        get_forecasts(issue),
        observations,
        issue,
        params,
        params.gdf,
    )
    for issue in params.issue
]

fbf_roc = pd.concat(fbf_roc_issues)
print(fbf_roc)
# -

# By running the next cell, you can save the dataframe containing the ROC scores.

fbf_roc.to_csv(
    f"data/{params.iso}/outputs/Districts_FbF/{params.index}/fbf.districts.roc.{params.index}.{params.year}.txt",
    index=False,
)

# **TODO**: Add plots to visualize ROC scores and eventually compare with reference.

# **TODO**: Handle aggregated obs/probs paths so things don't get mixed up when several different inputs are tested locally. Currently, these paths are hard-coded as the scripts have been designed to be used operationally.

# ### Triggers selection per vulnerability

# We've now come to the final part: the triggers optimization! All you have to do is execute the next cell and the calculations will take place automatically.

run_triggers(country, index)  # TODO check if still bug before window

# The triggers dataframe has been saved here: `"data/MOZ/outputs/Plots/triggers.aa.python.{params.index}.{params.year}.{GT/NRT}.csv"`

trigs = pd.read_csv(
    f"data/MOZ/outputs/Plots/triggers.aa.python.{params.index}.{params.year}.NRT.csv",
)

# **TODO**: Handle properly GT/NRT runs (probably add parameter to run_triggers)

trigs

# **TODO**: Add coverage + plots to visualize HR/FR and compare.

# ### Get final dataframe

# Now, you are done with the processing of one index (SPI or DRYSPELL). So you can rerun everything from the beginning with the other index. If you've already done it, you can run the next cell so it will merge all the different outputs to provide you with the very final dataframe that will be used operationally.

# **TODO**: use merge-spi-dryspell-gt-nrt-triggers.py

# **TODO**: add some viz for final dataframe

# Please find the final dataframe here!
#
# `data/{country}/outputs/Plots/triggers.aa.python.spi.dryspell.{params.year}.csv`
