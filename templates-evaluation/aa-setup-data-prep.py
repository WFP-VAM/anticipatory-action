# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: hdc
#     language: python
#     name: conda-env-hdc-py
# ---

# %% [markdown]
# ### Imports

# %%
import os
import fsspec
import glob
import s3fs
import numpy as np
import hvplot.pandas
import hvplot.xarray
import xarray as xr
import pandas as pd
import panel as pn
import holoviews as hv
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import Markdown as md
from hip.analysis.analyses.drought import concat_obs_levels

pn.extension()
pn.extension("tabulator")


# %% [markdown]
# ###  Reading functions

# %%
def read_aggregated_probs(path_to_zarr, index):
    fs, _, _ = fsspec.get_fs_token_paths(path_to_zarr)
    list_issue_paths = sorted(fs.glob(f"{path_to_zarr}/*"))[
        :-1
    ]  # Last one is the `obs` folder.
    list_index = {}

    for iss_path in list_issue_paths:
        list_index_paths = fs.glob(f"{iss_path}/{index} *")
        list_index_raw = [
            fs.sep.join([i, "probabilities.zarr"]) for i in sorted(list_index_paths)
        ]
        list_index_bc = [
            fs.sep.join([i, "probabilities_bc.zarr"]) for i in sorted(list_index_paths)
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


# %%
def read_aggregated_obs(path_to_zarr, index, intensity_thresholds):
    fs, _, _ = fsspec.get_fs_token_paths(path_to_zarr)
    list_index_paths = fs.glob(f"{path_to_zarr}/{index} *")

    # Restore full S3 paths if needed
    if isinstance(fs, s3fs.core.S3FileSystem):
        list_index_paths = [
            f"s3://{fs._strip_protocol(path)}" for path in list_index_paths
        ]

    list_val_paths = [
        fs.sep.join([ind_path, "observations.zarr"]) for ind_path in list_index_paths
    ]

    obs_val = xr.open_mfdataset(
        list_val_paths,
        engine="zarr",
        preprocess=lambda ds: ds["mean"],
        combine="nested",
        concat_dim="index",
    )
    obs_bool = concat_obs_levels(obs_val, levels=intensity_thresholds)

    obs = xr.Dataset({"bool": obs_bool, "val": obs_val})

    # Reformat time and index coords
    obs["time"] = [pd.to_datetime(t).year for t in obs.time.values]
    obs["index"] = [val_path.split(fs.sep)[-1] for val_path in list_index_paths]
    return obs


# %% [markdown]
# ###  Define country and data path

# %%
COUNTRY = "TZA"

# Ideally we would move the aa folder that's in my bucket to a dedicated bucket like for LIA
DATA_PATH = f"/s3/scratch/amine.barkaoui/aa/data/{COUNTRY.lower()}"

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Read indicator analysis data

# %%
probs_spi = read_aggregated_probs(f"{DATA_PATH}/zarr/2022", "spi")
probs_dry = read_aggregated_probs(f"{DATA_PATH}/zarr/2022", "dryspell")

probs = xr.concat([probs_spi, probs_dry], 'index')

# %%
probs

# %%
intensity_thresholds = {'Normal': -0.44, 'Mild': -0.68, 'Moderate': -0.85, 'Severe': -1}

chirps_spi = read_aggregated_obs(f"{DATA_PATH}/zarr/2022/obs", "spi", intensity_thresholds)
chirps_dry = read_aggregated_obs(f"{DATA_PATH}/zarr/2022/obs", "dryspell", intensity_thresholds)

chirps_anomaly = xr.concat([chirps_spi, chirps_dry], 'index')

# %%
chirps_anomaly

# %%
roc = pd.concat([
    pd.read_csv(f"{DATA_PATH}/auc/fbf.districts.roc.spi.2022.csv"),
    pd.read_csv(f"{DATA_PATH}/auc/fbf.districts.roc.dryspell.2022.csv"),    
])

# %%
roc

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Visualize indicator performance analysis

# %%
# ROC scores
display(
    md(
        f"**This roc file shows {round(100 * roc.BC.sum() / len(roc), 1)} % of bias-corrected values.**"
    )
)
display(roc)

# Filter to include only 'AUC_best' scores and pivot the table
roc_pivot = roc.pivot_table(values="AUC_best", index="Index", columns="district")

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(roc_pivot, annot=False, cmap="YlGnBu", cbar_kws={"label": "AUC_best"})
plt.title("AUC_best Scores Heatmap - Below Normal")
plt.xlabel("District")
plt.ylabel("Index")
plt.title("ROC scores (best between raw and bc)")
plt.show()

# %%
# Probabilities
issue_widget = pn.widgets.Select(name="Issue Month", options=sorted(probs.issue.values.tolist()))
district_widget = pn.widgets.Select(name="District", options=sorted(probs.district.values.tolist()))
category_widget = pn.widgets.Select(name="Category", options=sorted(probs.category.values.tolist()))
index_widget = pn.widgets.Select(name="Index", options=sorted(probs.sel(issue=issue_widget.value).index.values.tolist()))
     
@pn.depends(issue_widget, index_widget, district_widget, category_widget)
def plot_timeseries(issue, index, district, category):
    sel = probs.sel(issue=issue, index=index, district=district, category=category)

    raw_plot = sel["raw"].hvplot(label="Raw", color="blue")
    bc_plot = sel["bc"].hvplot(label="Bias-Corrected", color="green")

    return (raw_plot * bc_plot).opts(show_grid=True, legend_position="top_left")

###
# Here please make sure to select an index within the 7-month leadtime of the issue month
###

pn.Column(
    pn.Row(issue_widget, index_widget),
    pn.Row(district_widget, category_widget),
    plot_timeseries
)

# %%
# Observations
district_widget = pn.widgets.Select(name="District", options=sorted(probs.district.values.tolist()))
category_widget = pn.widgets.Select(name="Category", options=sorted(probs.category.values.tolist()))
index_widget = pn.widgets.Select(name="Index", options=sorted(probs.sel(issue=issue_widget.value).index.values.tolist()))
     
@pn.depends(index_widget, district_widget, category_widget)
def plot_obs_timeseries(index, district, category):
    sel = chirps_anomaly.sel(index=index, district=district, category=category)
    df = sel[["val", "bool"]].to_dataframe().reset_index()

    line = df.hvplot.line(x="time", y="val", color="blue", label="val")

    points_df = df[df["bool"] == 1]
    points = points_df.hvplot.scatter(
        x="time", y="val", color="red", size=10, marker="o", label="bool=1"
    )

    threshold = intensity_thresholds.get(category, None) * 1000
    if threshold is not None:
        hline = hv.HLine(threshold).opts(color="red", line_dash="dashed", line_width=2)
        return (line * points * hline).opts(show_grid=True, legend_position="top_left")

    return (line * points).opts(show_grid=True, legend_position="top_left")


pn.Column(
    pn.Row(index_widget, district_widget, category_widget),
    plot_obs_timeseries
)

# %% [markdown]
# ###  Read triggers data

# %%
triggers = pd.concat([
    *[pd.read_csv(f) for f in glob.glob(f"{DATA_PATH}/triggers/triggers_metrics/*")],
])

# %%
triggers

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Visualize trigger metrics

# %%
index_options = sorted(triggers["index"].unique())
district_options = sorted(triggers["district"].unique())
category_options = sorted(triggers["category"].unique())

index_widget = pn.widgets.Select(name="Index", options=index_options)
district_widget = pn.widgets.Select(name="District", options=district_options)
category_widget = pn.widgets.Select(name="Category", options=category_options)

@pn.depends(
    index_widget, district_widget, category_widget,
)
def filtered_table(index, district, category):
    filtered = triggers[
        (triggers["index"] == index) &
        (triggers["district"] == district) &
        (triggers["category"] == category)
    ]
    return pn.widgets.Tabulator(filtered, pagination="local", page_size=10, width=1000)

pn.Column(
    pn.Row(index_widget, district_widget, category_widget),
    filtered_table
)

# %% [markdown]
# ### Format to dataframe and save to parquet

# %%
probs_df = probs.to_dataframe().dropna().reset_index()
chirps_df = chirps_anomaly.to_dataframe().dropna().reset_index()

# %%
probs_df['index'] = probs_df['index'].str.upper()
chirps_df['index'] = chirps_df['index'].str.upper()
triggers['index'] = triggers['index'].str.upper()

# %%
# Rename time column as year column
probs_df = probs_df.rename(columns={'time': 'year'})
chirps_df = chirps_df.rename(columns={'time': 'year'})

# %%
# Unscale CHIRPS-based SPI
chirps_df['val'] = chirps_df.val / 1000

# %%
# Change types
probs_df['issue'] = probs_df.issue.astype(np.uint8)
probs_df['year'] = probs_df.year.astype(np.uint16)
chirps_df['year'] = chirps_df.year.astype(np.uint16)
chirps_df['bool'] = chirps_df['bool'].astype(bool)
roc['BC'] = roc.BC.astype(bool)
roc['issue'] = roc.issue.astype(np.uint8)

# %%
triggers['issue_ready'] = triggers.issue_ready.astype(np.uint8)
triggers['issue_set'] = triggers.issue_set.astype(np.uint8)
triggers['lead_time'] = triggers.lead_time.astype(np.uint8)
triggers['FN'] = triggers.FN.astype(np.uint8)
triggers['FP'] = triggers.FP.astype(np.uint8)
triggers['FPtol'] = triggers.FPtol.astype(np.uint8)
triggers['TN'] = triggers.TN.astype(np.uint8)
triggers['TP'] = triggers.TP.astype(np.uint8)
triggers['RP'] = triggers.RP.astype(np.uint8)

# %%
outdir = "s3://wfp-ops-userdata/amine.barkaoui/aa/data/setup-tool/tza"
#os.makedirs(outdir, exist_ok=True)

probs_df.to_parquet(f"{outdir}/probs.parquet", index=False)
chirps_df.to_parquet(f"{outdir}/obs.parquet", index=False)
roc.to_parquet(f"{outdir}/roc.parquet", index=False)
triggers.to_parquet(f"{outdir}/triggers.parquet", index=False)
