# ---
# jupyter:
#   jupytext:
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

# %cd ..

# +
import logging
import os

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from AA.analytical import run_issue_verification
from AA.helper_fns import (
    create_flexible_dataarray,
    format_triggers_df_for_dashboard,
    merge_un_biased_probs,
    triggers_da_to_df,
)
from AA.triggers import (
    filter_triggers_by_window,
    find_optimal_triggers,
    get_window_district,
    read_aggregated_obs,
    read_aggregated_probs,
)
from config.params import Params
from hip.analysis.analyses.drought import get_accumulation_periods
from hip.analysis.aoi.analysis_area import AnalysisArea
from odc.geo.xr import xr_reproject

AWS_PROFILE = os.getenv("AWS_PROFILE", None)

s3fs_client = s3fs.S3FileSystem(profile=AWS_PROFILE)

logging.basicConfig(level="INFO")
logging.info(f"Using AWS_PROFILE: {AWS_PROFILE}")


# +
params = Params(iso="TEST", index="SPI")

vulnerability = "GT"
# -

# Create directory for ROC scores df per issue month in case it doesn't exist
os.makedirs(
    f"{params.data_path}/data/{params.iso}/auc/split_by_issue",
    exist_ok=True,
)

area = AnalysisArea.from_admin_boundaries(
    iso3="MOZ",
    datetime_range="1981-01-01/2021-12-31",
    resolution=0.25,
    admin_level=2,
)

# Forecasts data reading
store = s3fs.S3Map(root="s3://hip-analysis-tests/fixtures/seas5.zarr", s3=s3fs_client)
ds_forecasts = xr.open_zarr(store)

# Observations data reading
store = s3fs.S3Map(root="s3://hip-analysis-tests/fixtures/chirps.zarr", s3=s3fs_client)
observations = xr.open_zarr(store)["CHIRPS-RFH_DAILY"]
observations = xr_reproject(observations, ds_forecasts.odc.geobox)
observations.attrs["nodata"] = np.nan

# +
fbf_roc_issues = []

for issue in ["06", "07"]:

    forecasts = ds_forecasts[f"ECMWF-RFH_FORECASTS_SEAS5_ISSUE{int(issue)}_DAILY"]
    forecasts.attrs["nodata"] = np.nan
    forecasts = forecasts.dropna(dim="time", how="all")

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
        f"Completed analytical process for {params.index.upper()} over {params.iso} country"
    )

fbf_roc = pd.concat(fbf_roc_issues)
# -

fbf_roc.to_csv(
    f"{params.data_path}/data/{params.iso}/auc/fbf.districts.roc.{params.index}.{params.calibration_year}.csv",
    index=False,
)

params = Params(iso="TEST", index="SPI")

# +
gdf = area.get_dataset([area.BASE_AREA_DATASET])
admin1 = area.get_admin_boundaries(iso3="MOZ", admin_level=1).drop(
    ["geometry", "adm0_Code"], axis=1
)
admin1.columns = ["Code_1", "adm1_name"]
gdf = pd.merge(gdf, admin1, how="left", left_on=["adm1_Code"], right_on=["Code_1"])

rfh = create_flexible_dataarray(params.start_season, params.end_season)
periods = get_accumulation_periods(
    rfh, 0, 0, params.min_index_period, params.max_index_period
)

obs = read_aggregated_obs(
    f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/obs",
    params,
)
obs = obs.assign_coords(
    lead_time=("index", [periods[i.split(" ")[-1]][0] for i in obs.index.values])
)
obs = obs.load()
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

# Trick to align couples of issue months inside apply_ufunc
probs_ready = probs.sel(issue=np.uint8(params.issue_months)[:-1]).load()
probs_set = probs.sel(issue=np.uint8(params.issue_months)[1:]).load()
probs_set["issue"] = [i - 1 if i != 1 else 12 for i in probs_set.issue.values]

# +
# Distribute computation of triggers
logging.info(
    f"Starting computation of triggers on the whole {params.iso.upper()} country..."
)
trigs, score = xr.apply_ufunc(
    find_optimal_triggers,
    obs.bool,
    obs.val,
    probs_ready.prob,
    probs_set.prob,
    obs.lead_time,
    probs.issue,
    obs.category,
    vulnerability,
    params,
    vectorize=True,
    join="outer",
    input_core_dims=[["time"], ["time"], ["time"], ["time"], [], [], [], [], []],
    output_core_dims=[["trigger"], []],
    dask="parallelized",
    keep_attrs=True,
)
trigs["trigger"] = ["trigger1", "trigger2"]

trigs["category"] = trigs.category.astype(str)
trigs["district"] = trigs.district.astype(str)
trigs["index"] = trigs.index.astype(str)

score["category"] = score.category.astype(str)
score["district"] = score.district.astype(str)
score["index"] = score.index.astype(str)

# +
# Reset cells of xarray of no interest as nan
trigs = trigs.where(probs.prob.count("time") != 0, np.nan)
score = score.where(probs.prob.count("time") != 0, np.nan)

# Format trigs and score into a dataframe
trigs_df = triggers_da_to_df(trigs, score).dropna()
trigs_df = trigs_df.query("HR < 0")  # remove row when trigger not found (penalty)

# Add window information depending on district
trigs_df["Window"] = [
    get_window_district(gdf, row["index"].split(" ")[-1], row.district, params)
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

# Only focus on June issue month for this test
df_leadtime = df_leadtime[df_leadtime.issue == 6]

# Keep 4 pairs of triggers per window of activation
df_window = filter_triggers_by_window(
    df_leadtime,
    probs_ready,
    probs_set,
    obs,
    vulnerability,
    params,
)

# Format triggers dataframe for dashboard
triggers = format_triggers_df_for_dashboard(df_window, params)
# -

triggers

ref = pd.read_csv("tests/integration_triggers_ref.csv")

ref

pd.testing.assert_frame_equal(triggers, ref, check_like=True)
