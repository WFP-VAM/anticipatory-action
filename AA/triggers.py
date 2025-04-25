import logging

import click

logging.basicConfig(level="INFO", force=True)

import warnings

warnings.simplefilter(action="ignore")

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr
from hip.analysis.analyses.drought import (concat_obs_levels,
                                           get_accumulation_periods)
from hip.analysis.aoi.analysis_area import AnalysisArea
from hip.analysis.compute.utils import start_dask
from numba import jit, types
from numba.typed import Dict
from tqdm import tqdm

from AA._triggers_ready_set import (objective, run_pilot_districts_metrics,
                                    run_ready_set_brute_selection)
from AA.helper_fns import (create_flexible_dataarray,
                           format_triggers_df_for_dashboard,
                           merge_un_biased_probs, triggers_da_to_df)
from config.params import Params


@click.command()
@click.argument("country", required=True, type=str)
@click.argument("index", default="SPI")
@click.argument("vulnerability", default="GT")
def run(country, index, vulnerability):
    client = start_dask(n_workers=1)

    params = Params(iso=country, index=index, vulnerability=vulnerability)

    run_triggers_selection(params)


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
    obs = obs.sel(index=[params.index + " " + k for k in periods.keys()])
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
    probs = probs.sel(index=[params.index + " " + k for k in periods.keys()])
    logging.info(
        f"Completed reading of aggregated probabilities for the whole {params.iso.upper()} country"
    )

    # Filter year dimension: temporary before harmonization with analytical script
    obs = obs.sel(time=probs.time.values).load()

    # Filter on indicators of interest
    obs = obs.sel(index=params.indicators)
    probs = probs.sel(index=params.indicators)

    # Align couples of issue months inside apply_ufunc
    probs_ready = probs.sel(
        issue=np.uint8(params.issue_months)[:-1]
    ).load()  # use start/end season here
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
        f"{params.data_path}/data/{params.iso}/triggers/triggers.{params.index}.{params.calibration_year}.{params.vulnerability}.validate.csv",
        index=False,
    )

    logging.info(
        f"Triggers dataframe saved as a csv for {params.index} {params.vulnerability}"
    )


def filter_triggers_by_window(df_leadtime, probs_ready, probs_set, obs, params):
    def _sel_row(da, row, index, issue=None):
        da_sel = da.sel(district=row.district.unique(), index=index)
        if "issue" in da.dims:
            da_sel = da_sel.sel(issue=issue)
        if "category" in da.dims:
            da_sel = da_sel.sel(category=row.category.unique())
        return da_sel

    def _get_top_pairs_per_window(tdf, n_to_keep=4):
        for (ind, iss), sub_tdf in tdf.groupby(["index", "issue"]):
            t = sub_tdf.sort_values("trigger").trigger_value.values
            issue = sub_tdf.issue.unique()
            out = np.empty(9, dtype=np.float64)
            stats = objective(
                t,
                _sel_row(obs.val, tdf, ind).values[0],
                _sel_row(obs.bool, tdf, ind).values[0][0],
                _sel_row(probs_ready.prob, tdf, ind, issue).values[0][0][0],
                _sel_row(probs_set.prob, tdf, ind, issue).values[0][0][0],
                _sel_row(obs, tdf, ind).lead_time.values,
                _sel_row(probs_ready.prob, tdf, ind, issue).issue.values[0],
                _sel_row(obs, tdf, ind).tolerance.values,
                0,
                _sel_row(obs, tdf, ind).return_period.values,
                np.float64(params.requirements["HR"]),
                np.float64(params.requirements["SR"]),
                np.float64(params.requirements["FR"]),
                1e6,
                10e-3,
                1e-6,
                np.empty(9, dtype=np.int8),
                out,
            )
            hr, fr = stats[4], stats[7]
            tdf.loc[(tdf["index"] == ind) & (tdf.issue == iss), "HR"] = hr
            tdf.loc[(tdf["index"] == ind) & (tdf.issue == iss), "FR"] = fr
        if len(tdf) < (2 * n_to_keep):  # more than two pairs otherwise no need
            return tdf
        else:
            best_four = (
                tdf.sort_values("lead_time")
                .sort_values("FR")
                .sort_values("HR", ascending=False)
                .head(2 * n_to_keep)
            )
            return best_four

    triggers_window_list = [
        _get_top_pairs_per_window(r)
        for _, r in df_leadtime.groupby(["district", "category", "Window"])
    ]

    return pd.concat(triggers_window_list)


def get_window_district(area, indicator, district, params):
    gdf = area.get_dataset([area.BASE_AREA_DATASET])
    admin1 = area.get_admin_boundaries(iso3=params.iso, admin_level=1).drop(
        ["geometry", "adm0_Code"], axis=1
    )
    admin1.columns = ["Code_1", "adm1_name"]
    shp = pd.merge(gdf, admin1, how="left", left_on=["adm1_Code"], right_on=["Code_1"])

    province = shp.loc[shp.Name == district].adm1_name.unique()[0]

    # Get window1 and window2 definitions
    window1 = params.get_windows("window1")
    window2 = params.get_windows("window2")

    if isinstance(window1, dict):
        if indicator in window1[province]:
            return "Window 1"
        elif indicator in window2[province]:
            return "Window 2"
        else:
            return np.nan
    else:
        if indicator in window1:
            return "Window 1"
        elif indicator in window2:
            return "Window 2"
        else:
            return np.nan


def read_aggregated_obs(path_to_zarr, params):
    list_index_paths = glob.glob(f"{path_to_zarr}/{params.index} *")
    list_val_paths = [os.path.join(l, "observations.zarr") for l in list_index_paths]

    obs_val = xr.open_mfdataset(
        list_val_paths,
        engine="zarr",
        preprocess=lambda ds: ds["mean"],
        combine="nested",
        concat_dim="index",
    )
    obs_bool = concat_obs_levels(obs_val, levels=params.intensity_thresholds)

    obs = xr.Dataset({"bool": obs_bool, "val": obs_val})
    obs["time"] = [pd.to_datetime(t).year for t in obs.time.values]
    obs["index"] = [os.path.split(os.path.dirname(i))[-1] for i in list_val_paths]
    return obs


def read_aggregated_probs(path_to_zarr, params):
    list_issue_paths = sorted(glob.glob(f"{path_to_zarr}/*"))[
        :-1
    ]  # Last one is the `obs` folder.
    list_index = {}

    for l in list_issue_paths:
        list_index_raw = [
            os.path.join(i, "probabilities.zarr")
            for i in sorted(glob.glob(f"{l}/{params.index} *"))
        ]
        list_index_bc = [
            os.path.join(i, "probabilities_bc.zarr")
            for i in sorted(glob.glob(f"{l}/{params.index} *"))
        ]
        index_names = [os.path.split(os.path.dirname(i))[-1] for i in list_index_raw]

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
        list_index[int(os.path.split(l)[-1])] = ds_index

    return xr.concat(list_index.values(), dim=pd.Index(list_index.keys(), name="issue"))


if __name__ == "__main__":
    # From AA repository:
    # $ python triggers.py MOZ SPI NRT

    run()
