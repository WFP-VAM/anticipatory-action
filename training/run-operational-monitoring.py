#!/usr/bin/env python
# coding: utf-8

# ## Run AA operational monitoring script
# 
# #### (can be used for a more user-friendly experience or for training purposes)
# 
# #### (note: having run entirely the `run_full_verification` notebook is a prerequisite to run this one)
# 
# This notebook reads a forecasts dataset (corresponding to a specific issue month) and computes the corresponding probabilities. These probabilities are merged with the pre-computed triggers dataframe to be displayed on the dashboard.

# **Import required libraries and functions**

# In[1]:


get_ipython().run_line_magic('cd', '..')


# In[2]:


import os
import pandas as pd

from config.params import Params

from AA.helper_fns import read_observations, read_forecasts
from AA.operational import run_full_index_pipeline

from hip.analysis.aoi.analysis_area import AnalysisArea
from hip.analysis.analyses.drought import get_accumulation_periods


# **First, please define the country ISO code, the issue month and the index of interest**

# In[15]:


country = "MOZ"
issue = 6
index = "SPI"  # 'SPI' or 'DRYSPELL'


# Now, we will configure some parameters. Please feel free to edit the `{country}_config.yaml` file if you need to change the *monitoring_year* or any other relevant parameter.

# In[16]:


params = Params(iso=country, issue=issue, index=index)


# ### Read data

# Let's start by getting the shapefile.

# In[17]:


area = AnalysisArea.from_admin_boundaries(
    iso3=params.iso.upper(),
    admin_level=2,
    resolution=0.25,
    datetime_range=f"1981-01-01/{params.monitoring_year + 1}-06-30",
)

gdf = area.get_dataset([area.BASE_AREA_DATASET])
gdf


# The next cell reads the observations dataset. Please run it directly if you have the data stored in the specified path or have access to HDC.
# 
# 
# *Note:*
# 
# If you previously ran the `run-full-verification` notebook, you probably already have the dataset stored locally. In that case, you can give its path as an argument to `read_observations`.

# In[18]:


# Observations data reading
observations = read_observations(
    area,
    f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/obs/observations.zarr",
)


# As with observations, forecasts are easy to read using hip-analysis, called within the `read_forecasts` function.
# 
# Please note the *update* parameter that allows to re-load the data from HDC in order to get the latest updates. This means that if you are running this for the second time, you can set this parameter to **False**, so the data is read directly from the local file system.
# 
# For training purposes, we will also keep this parameter as False in order to avoid dealing with HDC credentials.

# In[19]:


forecasts = read_forecasts(
    area,
    issue,
    f"{params.data_path}/data/{params.iso}/zarr/2022/{str(issue).zfill(2)}/forecasts.zarr",
    update=False,  # True,
)
forecasts


# In[20]:


forecasts.isel(ensemble=0).mean("time").hip.viz.map(
    title=f"Rainfall forecasts (issue {issue}) average over time for control member"
)


# Now that we got all the data we need, let's read the triggers file so we can merge the probabilities with it once we have them. This triggers file corresponds to the output of the `run-full-verification` notebook if we're in May. Then, we read the merged dataframe that already contains the probabilities from the previous months so we add the new probabilities to the existing merged dataframe.
# 
# **Note:**
# 
# This means that if you want to re-run the probabilities for May, you should delete or move the existing probabilities dataframes from the probs directory.

# In[21]:


# Read triggers file
if os.path.exists(f"{params.data_path}/data/{params.iso}/probs/aa_probabilities_triggers_pilots.csv"):
    triggers_df = pd.read_csv(
        f"{params.data_path}/data/{params.iso}/probs/aa_probabilities_triggers_pilots.csv",
    )
else:
    triggers_df = pd.read_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.pilots.csv",
    )


# ### Run forecasts processing

# Before calculating the accumulation, anomaly etc..., we need to obtain the accumulation periods we will be focusing on. These depend on the issue month of the forecasts that we are currently processing.

# In[22]:


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

# In[23]:


# Compute probabilities for each accumulation period
probs_merged_dataframes = [
    run_full_index_pipeline(
        forecasts,
        observations,
        params,
        triggers_df,
        gdf,
        period_name,
        period_months,
    )
    for period_name, period_months in accumulation_periods.items()
]


# We reorganise the dataframes and we are ready to save them.

# In[24]:


probs_df, merged_df = zip(*probs_merged_dataframes)

probs_dashboard = pd.concat(probs_df).drop_duplicates()

merged_db = pd.concat(merged_df).sort_values(["prob_ready", "prob_set"])
merged_db = merged_db.drop_duplicates(
    merged_db.columns.difference(["prob_ready", "prob_set"]), keep="first"
)
merged_db = merged_db.sort_values(["district", "index", "category"])
merged_db


# ### Save drought probabilities

# In[25]:


# Save all probabilities
probs_dashboard.to_csv(
    f"{params.data_path}/data/{params.iso}/probs/aa_probabilities_{params.index}_{params.issue}.csv",
    index=False,
)


# In[26]:


# Save probabilities merged with triggers
merged_db.sort_values(["district", "index", "category"]).to_csv(
    f"{params.data_path}/data/{params.iso}/probs/aa_probabilities_triggers_pilots.csv",
    index=False,
)

