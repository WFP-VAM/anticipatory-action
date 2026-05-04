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
#     display_name: Python (pixi-aa)
#     language: python
#     name: pixi-aa
# ---

# ## Run full AA drought verification
#
# #### (can be used for a more user-friendly experience or for training purposes)
#
# This notebook is intended to be self-sufficient for executing the entire workflow operationally ahead of the season and get the triggers using specific parameters and specific datasets. It is designed to be interactive, and does not require any direct interaction with another file, except for the configuration file. This will therefore be the main front-end for Anticipatory Action analysts.

# If you have not downloaded the data yet, please download it from the link you should have received by email.

# **Import required libraries and functions**

# %load_ext autoreload
# %autoreload 2

# +
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr
from IPython.display import Markdown as md

from analysis_area import AnalysisArea

import os
if os.getcwd().split("/")[-1] != "anticipatory-action":
    os.chdir("../..")
os.getcwd()

from AA.helpers.params import Params
from AA.helpers.utils import get_coverage, read_forecasts, read_observations
from AA.cli.triggers import run_triggers_selection
# -

from hip.analysis import __version__
__version__

# **First, please define the country ISO code and the index of interest**


country = "ZMB"
index = "SPI"  # 'SPI' or 'DRYSPELL'
data_path = "s3://wfp-ops-userdata/amine.barkaoui/aa"  # current directory (anticipatory-action)
output_path = "s3://wfp-ops-userdata/amine.barkaoui/aa"


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

kitwe_gdf = gdf.loc[['Kitwe']]

area.add_dataset(kitwe_gdf, [area.BASE_AREA_DATASET])

area.get_dataset([area.BASE_AREA_DATASET])


# +
# Observations data reading
observations = read_observations(
    area,
    f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/obs/observations.zarr",
)

# Clip to Kitwe bounds
observations = observations.sel(latitude=slice(-12., -13.5), longitude=slice(27.5, 29))

observations
# -

forecasts_folder_path = (
    f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}"
)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### Rasterization visualization
# -

area = AnalysisArea.from_admin_boundaries(
    iso3=params.iso.upper(),
    admin_level=2,
    resolution=0.25,
    datetime_range=f"1981-01-01/{params.calibration_year}-06-30",
)

forecasts = read_forecasts(
    area,
    "07",
    f"{forecasts_folder_path}/07/forecasts.zarr",
)

data = forecasts.isel(time=slice(100, 1200), ensemble=0).mean('time')
data = data.sel(latitude=slice(-12., -13.5), longitude=slice(27.5, 29))

# ALL TOUCHED = FALSE
zone_ids, zones = area._resolve_zones(data, None, None, False)
(data.where(zones > -1)).plot.imshow()

# ALL TOUCHED = TRUE
zone_ids, zones = area._resolve_zones(data, None, None, True)
(data.where(zones > -1)).plot.imshow()

# +
import matplotlib.pyplot as plt

# Extract the raster of interest
raster = data.where(zones > -1)

# Create figure + axis
fig, ax = plt.subplots(figsize=(16, 4))

# Plot raster on the axis
raster.plot.imshow(ax=ax, cmap="viridis")

# Overlay the vector geometry
kitwe_gdf.boundary.plot(ax=ax, color="red", linewidth=2)

# Optional title
ax.set_title("Raster with Kitwe Boundary Overlay")

plt.show()
# -

# ### Analytical processing

# +
import fsspec

from hip.analysis.ops._statistics import evaluate_roc_forecasts
from hip.analysis.analyses.drought import get_accumulation_periods
from AA.cli.analytical import calculate_forecast_probabilities, get_verification_df

def compute_district_average(da, area):
    """
    Computes zonal statistics on an xarray DataArray for both observations and probabilities.

    Args:
        da : xarray.DataArray, Input DataArray (can be observations or probabilities).
        area : hip.analysis.aoi.analysis_area.AnalysisArea: object characterizing the area
            and admin level of interest.
    Returns: xarray.DataArray, DataArray with computed district averages.
    """
    # Ensure consistent time dimension
    if "year" in da.dims:
        da = da.rename({"year": "time"})

    # Determine dimensions to group by (exclude spatial dimensions)
    groupby_dim = set(da.dims) - {"latitude", "longitude", "time"}

    # Transpose dims to ensure equality of shapes
    da = da.transpose(..., *groupby_dim, "latitude", "longitude")

    # Compute zonal stats: handle different groupby dimensions lengths
    if len(groupby_dim) > 1:
        raise NotImplementedError(
            "Zonal stats with more than one groupby dimension are not supported."
        )
    elif len(groupby_dim) == 1:
        da_grouped = da.groupby(*groupby_dim).map(
            lambda da: area.zonal_stats(
                da.squeeze(groupby_dim), 
                stats=["mean"], 
                zone_ids=None, 
                zones=None,
                all_touched=True,
            )
            .query("zone != 'Administrative unit not available'")
            .to_xarray()["mean"]
        )
    else:
        da_grouped = (
            area.zonal_stats(
                da, 
                stats=["mean"], 
                zone_ids=None, 
                zones=None,
                all_touched=True,
            )
            .query("zone != 'Administrative unit not available'")
            .to_xarray()["mean"]
        )

    # Rename 'zone' to 'district' for consistency
    da_grouped = da_grouped.rename({"zone": "district"})

    # Ensure district is a string type
    da_grouped["district"] = da_grouped.district.astype(str)

    return da_grouped


def save_districts_results(
    observations,
    probabilities,
    probabilities_bc,
    area,
    issue,
    period_name,
    params,
):
    # Aggregate by district
    obs_district = compute_district_average(observations, area)
    probs_district = compute_district_average(probabilities, area)
    probs_bc_district = compute_district_average(probabilities_bc, area)

    # Convert the 'category' coordinate to string type
    probs_district["category"] = probs_district["category"].astype(str)
    probs_bc_district["category"] = probs_bc_district["category"].astype(str)

    # Define file paths
    obs_path = f"{params.output_path}/data/{params.iso}/zarr/{params.calibration_year}/obs/{params.index} {period_name}/observations_kitwe.zarr"
    probs_path = f"{params.output_path}/data/{params.iso}/zarr/{params.calibration_year}/{issue}/{params.index} {period_name}/probabilities_kitwe.zarr"
    probs_bc_path = f"{params.output_path}/data/{params.iso}/zarr/{params.calibration_year}/{issue}/{params.index} {period_name}/probabilities_bc_kitwe.zarr"

    obs_district.to_zarr(obs_path, mode="w")
    probs_district.to_zarr(probs_path, mode="w")
    probs_bc_district.to_zarr(probs_bc_path, mode="w")
    

def verify_index_across_districts(
    forecasts,
    observations,
    params,
    area,
    period_name,
    period_months,
    issue,
):
    """
    Run analytical / verification pipeline for a single issue month and a single index (period)

    Args:
        forecasts: xarray.Dataset, rainfall forecasts dataset for specific issue month
        observations: xarray.Dataset, rainfall observations dataset
        params: Params, parameters class
        area: hip.analysis.AnalysisArea object with aoi information
        period_name: str, name of index period (eg "ON")
        period_months: tuple, months of index period (eg (10, 11))
    Returns:
        fbf_issue_df: pandas.DataFrame, dataframe with roc scores for all districts, categories and specified issue month / period
    """

    probs, probs_bc, obs_values, obs_bool = calculate_forecast_probabilities(
        forecasts,
        observations,
        params,
        period_months,
        issue,
    )

    auc, auc_bc = evaluate_roc_forecasts(
        obs_bool,
        probs,
        probs_bc,
    )

    if params.save_zarr:
        save_districts_results(
            obs_values,
            probs,
            probs_bc,
            area,
            issue,
            period_name,
            params,
        )

    # Aggregate by district
    auc_district = compute_district_average(auc, area)
    auc_bc_district = compute_district_average(auc_bc, area)

    # Choose W/ or W/OUT BC based on AUROC
    fbf_index_df = get_verification_df(
        auc_district,
        auc_bc_district,
    )
    fbf_index_df["Index"] = f"{params.index.upper()} {period_name}"

    logging.info(
        f"Completed FbF ROC computation by district for the {params.index.upper()} {period_name} index"
    )

    return fbf_index_df

def run_issue_verification(forecasts, observations, issue, params, area):
    """
    Run analytical / verification pipeline for one issue month

    Args:
        observations: xarray.Dataset, rainfall observations dataset
        issue: str, issue month of forecasts to analyse
        params: Params, parameters class
        area: hip.analysis.AnalysisArea object with aoi information
    Returns:
        fbf_issue: pandas.DataFrame, dataframe with roc scores for all indexes, districts, categories and a specified issue month
    """

    fbf_path = f"{params.output_path}/data/{params.iso}/auc/split_by_issue_kitwe/fbf.districts.roc.{params.index}.{params.calibration_year}.{issue}.csv"

    if fsspec.open(fbf_path).fs.exists(fbf_path):
        logging.info(
            f"FbF ROC verification by district for the issue month {issue} read from disk"
        )

        return pd.read_csv(fbf_path)

    else:
        # Get accumulation periods (DJ, JF, FM, DJF, JFM...)
        accumulation_periods = get_accumulation_periods(
            forecasts,
            params.start_season,
            params.end_season,
            params.min_index_period,
            params.max_index_period,
        )

        fbf_indexes = [
            verify_index_across_districts(
                forecasts,
                observations,
                params,
                area,
                period_name,
                period_months,
                issue,
            )
            for period_name, period_months in accumulation_periods.items()
        ]

        fbf_issue = pd.concat(fbf_indexes)
        fbf_issue["issue"] = int(issue)

        fbf_issue.to_csv(
            fbf_path,
            index=False,
        )

        logging.info(
            f"FbF ROC verification by district for the issue month {issue} done"
        )

        return fbf_issue


# + jupyter={"outputs_hidden": true}
# Define empty list for each issue month's ROC score dataframe
fbf_roc_issues = []

params.save_zarr = True

for issue in params.issue_months:
    forecasts = read_forecasts(
        area,
        issue,
        f"{forecasts_folder_path}/{issue}/forecasts.zarr",
    )
    
    forecasts = forecasts.sel(latitude=slice(-12., -13.5), longitude=slice(27.5, 29))

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

xr.open_zarr(f"{forecasts_folder_path}/07/{params.index} ON/probabilities_kitwe.zarr").load()

# We can also check how the CHIRPS-based anomalies that have been saved look like. They have been used to calculate the roc scores and will be used to select the triggers.

xr.open_zarr(f"{forecasts_folder_path}/obs/{params.index} ON/observations_kitwe.zarr").load()

# By running the next cell, you can save the dataframe containing the ROC scores. We commented it here so we don't overwrite the file with all the issue months with a file that only contains a few issue months.


fbf_roc.to_csv(
    f"{params.data_path}/data/{params.iso}/auc/fbf.districts.roc.{params.index}.{params.calibration_year}.kitwe.csv",
    index=False,
)

# Now we can read this dataframe locally to visualize the ROC scores.

roc = pd.read_csv(
    f"{params.data_path}/data/{params.iso}/auc/fbf.districts.roc.{params.index}.{params.calibration_year}.kitwe.csv",
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
    (roc.district.isin(['Kitwe'])) & (roc.category.isin(["Moderate"]))
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


# +
import s3fs
import numpy as np
from tqdm import tqdm
from hip.analysis.analyses.drought import (concat_obs_levels,
                                           get_accumulation_periods)
from AA.helpers.utils import (create_flexible_dataarray,
                              format_triggers_df_for_dashboard,
                              merge_un_biased_probs, triggers_da_to_df)
from AA.helpers._triggers import run_pilot_districts_metrics

def read_aggregated_obs(path_to_zarr, params):
    fs, _, _ = fsspec.get_fs_token_paths(path_to_zarr)
    list_index_paths = fs.glob(f"{path_to_zarr}/{params.index} *")

    # Restore full S3 paths if needed
    if isinstance(fs, s3fs.core.S3FileSystem):
        list_index_paths = [
            f"s3://{fs._strip_protocol(path)}" for path in list_index_paths
        ]

    list_val_paths = [
        fs.sep.join([ind_path, "observations_kitwe.zarr"]) for ind_path in list_index_paths
    ]

    obs_val = xr.open_mfdataset(
        list_val_paths,
        engine="zarr",
        preprocess=lambda ds: ds["mean"],
        combine="nested",
        concat_dim="index",
    )
    obs_bool = concat_obs_levels(obs_val, levels=params.intensity_thresholds)

    obs = xr.Dataset({"bool": obs_bool, "val": obs_val})

    # Reformat time and index coords
    obs["time"] = [pd.to_datetime(t).year for t in obs.time.values]
    obs["index"] = [val_path.split(fs.sep)[-1] for val_path in list_index_paths]
    return obs

def read_aggregated_probs(path_to_zarr, params):
    fs, _, _ = fsspec.get_fs_token_paths(path_to_zarr)
    list_issue_paths = sorted(fs.glob(f"{path_to_zarr}/*"))[
        :-1
    ]  # Last one is the `obs` folder.
    list_index = {}

    for iss_path in list_issue_paths:
        list_index_paths = fs.glob(f"{iss_path}/{params.index} *")
        list_index_raw = [
            fs.sep.join([i, "probabilities_kitwe.zarr"]) for i in sorted(list_index_paths)
        ]
        list_index_bc = [
            fs.sep.join([i, "probabilities_bc_kitwe.zarr"]) for i in sorted(list_index_paths)
        ]
        index_names = [i.split(fs.sep)[-1] for i in sorted(list_index_paths)]

        # Restore full S3 paths if needed
        if isinstance(fs, s3fs.core.S3FileSystem):
            list_index_raw = [
                f"s3://{fs._strip_protocol(path)}" for path in list_index_raw
            ]
            list_index_bc = [
                f"s3://{fs._strip_protocol(path)}" for path in list_index_bc
            ]

        index_raw = xr.open_mfdataset(
            list_index_raw,
            engine="zarr",
            preprocess=lambda ds: ds["mean"],
            combine="nested",
            concat_dim="index",
        )
        index_bc = xr.open_mfdataset(
            list_index_bc,
            engine="zarr",
            preprocess=lambda ds: ds["mean"],
            combine="nested",
            concat_dim="index",
        )

        ds_index = xr.Dataset({"raw": index_raw, "bc": index_bc})
        ds_index["index"] = index_names
        list_index[int(iss_path.split(fs.sep)[-1])] = ds_index

    return xr.concat(list_index.values(), dim=pd.Index(list_index.keys(), name="issue"))

def run_triggers_selection(params):
    area = AnalysisArea.from_admin_boundaries(
        iso3=params.iso.upper(),
        admin_level=2,
        resolution=0.25,
        datetime_range=f"1981-01-01/{params.calibration_year}-06-30",
    )

    rfh = create_flexible_dataarray(params.start_season, params.end_season)
    periods = get_accumulation_periods(
        rfh, 0, 0, params.min_index_period, params.max_index_period
    )

    obs = read_aggregated_obs(
        f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/obs",
        params,
    )

    # Filter obs on indicators of interest
    obs = obs.sel(index=params.indicators)

    # Assign `lead_time`, `tolerance` and `return_period` as coordinates to enable
    # straightforward broadcasting and efficient use in vectorized functions via
    # `apply_ufunc` with `guvectorize`. These variables depend on others, but passing
    # a dict to `guvectorize` is impossible.
    obs = obs.assign_coords(
        lead_time=("index", [periods[i.split(" ")[-1]][0] for i in obs.index.values])
    )
    obs = obs.assign_coords(
        tolerance=("category", [params.tolerance[cat] for cat in obs.category.values])
    )
    if params.requirements:
        obs = obs.assign_coords(
            return_period=(
                "category",
                [
                    np.int64(
                        params.requirements["RP"]
                        + 1 * (cat[:3].lower() == "mod")
                        + 3 * (cat[:3].lower() == "sev")
                    )
                    for cat in obs.category.values
                ],
            )
        )
    logging.info(
        f"Completed reading of aggregated observations for the whole {params.iso.upper()} country"
    )

    probs_ds = read_aggregated_probs(
        f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}",
        params,
    )
    probs = xr.concat(
        [
            merge_un_biased_probs(probs_ds.raw, probs_ds.bc, params, i.split(" ")[-1])
            for i in probs_ds.index.values
        ],
        dim="index",
    )
    logging.info(
        f"Completed reading of aggregated probabilities for the whole {params.iso.upper()} country"
    )

    # Filter year dimension: temporary before harmonization with analytical script
    obs = obs.sel(time=probs.time.values).load()

    # Filter probs on indicators of interest
    probs = probs.sel(index=params.indicators)

    # Filter on specific categories to facilitate computation
    obs = obs.where(
        obs.category.isin(list(params.intensity_thresholds.keys())), drop=True
    )
    probs = probs.where(
        probs.category.isin(list(params.intensity_thresholds.keys())), drop=True
    )

    # Align couples of issue months inside apply_ufunc
    probs_ready = probs.sel(issue=np.uint8(params.issue_months)[:-1]).load()
    probs_set = probs.sel(issue=np.uint8(params.issue_months)[1:]).load()
    probs_set["issue"] = [i - 1 if i != 1 else 12 for i in probs_set.issue.values]

    if params.vulnerability in [None, "TBD"]:
        run_pilot_districts_metrics(
            obs=obs.compute(),
            probs_ready=probs_ready.compute(),
            probs_set=probs_set.compute(),
            params=params,
        )
        return

    # Chunk obs and probabilities datasets
    obs = obs.chunk(dict(time=-1, category=-1, index=1, district=1))
    probs_ready = probs_ready.chunk(
        dict(time=-1, category=-1, index=1, issue=1, district=1)
    )
    probs_set = probs_set.chunk(
        dict(time=-1, category=-1, index=1, issue=1, district=1)
    )

    # Persist chunked inputs before computation
    obs = persist_with_progress_bar(obs)
    probs_ready = persist_with_progress_bar(probs_ready)
    probs_set = persist_with_progress_bar(probs_set)

    # Run triggers computation
    logging.info(
        f"Starting computation of triggers for the whole {params.iso.upper()} country..."
    )
    trigs, score = run_ready_set_brute_selection(
        obs, probs_ready, probs_set, probs, params
    )

    # Reset cells of xarray of no interest as nan
    trigs = trigs.where(probs.prob.count("time") != 0, np.nan)
    score = score.where(probs.prob.count("time") != 0, np.nan)

    # Format trigs and score into a dataframe
    trigs_df = triggers_da_to_df(trigs, score).dropna()
    trigs_df = trigs_df.query("HR < 0")  # remove row when trigger not found (penalty)

    # Add window information depending on district
    trigs_df["Window"] = [
        get_window_district(area, row["index"].split(" ")[-1], row.district, params)
        for _, row in trigs_df.iterrows()
    ]

    # Filter per lead time
    df_leadtime = pd.concat(
        [
            g.sort_values(["index", "issue"]).sort_values("HR", kind="stable").head(2)
            for _, g in trigs_df.dropna()
            .sort_values("HR")
            .groupby(
                ["category", "district", "Window", "lead_time"],
                as_index=False,
                sort=False,
            )
        ]
    )

    # Keep 4 pairs of triggers per window of activation
    df_window = filter_triggers_by_window(
        df_leadtime,
        probs_ready,
        probs_set,
        obs,
        params,
    )

    # Format triggers dataframe for dashboard
    triggers = format_triggers_df_for_dashboard(df_window, params)

    triggers.to_csv(
        f"{params.output_path}/data/{params.iso}/triggers/triggers.{params.index}.{params.calibration_year}.{params.vulnerability}.csv",
        index=False,
    )

    logging.info(
        f"Triggers dataframe saved as a csv for {params.index} {params.vulnerability}"
    )


# +
params.districts = ['Kitwe']

fbf_districts_path = f"{params.data_path}/data/{params.iso}/auc/fbf.districts.roc.{params.index}.2022.kitwe.csv"
params.fbf_districts_df = pd.read_csv(fbf_districts_path)
    
run_triggers_selection(params)
# -


# Then, we keep the best pair for each lead time and the 4 best pairs of triggers per window of activation (in terms of Hit Rate first, and Failure Rate then).

# The triggers dataframe has been saved here for each district: `"data/{iso}/triggers/triggers_metrics/triggers_metrics_tbd_{district}.csv"`
#
# Then, these dataframes can be explored in order to evaluate the trigger performance and attempt to find suitable triggers in terms of Hit Rate, Success Rate, and False Alarm Ratio.

triggers = pd.read_csv(
    f"{params.data_path}/data/{params.iso}/triggers/triggers_metrics/triggers_metrics_tbd_{params.districts[0]}.csv",
)
triggers



