import click
import logging

logging.basicConfig(level="INFO")

import warnings

warnings.simplefilter(action="ignore")

from numba import jit
from dask.distributed import Client

import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
from scipy.optimize import brute
from sklearn.metrics import confusion_matrix

from config.params import Params

from AA.helper_fns import (
    triggers_da_to_df,
    merge_un_biased_probs,
)

from hip.analysis.analyses.drought import (
    get_accumulation_periods,
    concat_obs_levels,
)


@click.command()
@click.argument("country", required=True, type=str)
@click.argument("index", default="SPI")
def run(country, index):
    client = Client()

    params = Params(iso=country, index=index)

    rfh = xr.DataArray(
        np.arange(1, 9),
        coords=dict(
            time=(
                ["time"],
                pd.date_range(
                    f"{params.start_season}/1/1990",
                    f"{params.end_season + 1}/28/1991",
                    freq="M",
                ),
            )
        ),
    )
    periods = get_accumulation_periods(
        rfh, 0, 0, params.min_index_period, params.max_index_period
    )

    obs = read_aggregated_obs(
        f"data/{params.iso}/outputs/zarr/obs/2022",
        params,
    )
    obs = obs.assign_coords(
        lead_time=("index", [periods[i.split(" ")[-1]][0] for i in obs.index.values])
    )
    obs = obs.assign_coords(
        vulnerability=(
            "district",
            [params.districts_vulnerability[d] for d in obs.district.values],
        )
    )
    logging.info(
        f"Completed reading of aggregated observations for the whole {params.iso} country"
    )

    probs_ds = read_aggregated_probs(
        f"data/{params.iso}/outputs/zarr/2022",
        params,
    )
    probs = xr.concat(
        [
            merge_un_biased_probs(probs_ds, probs_ds, params, i.split(" ")[-1])
            for i in probs_ds.index.values
        ],
        dim="index",
    )
    logging.info(
        f"Completed reading of aggregated probabilities for the whole {params.iso} country"
    )

    # Filter year/time dimension: temporary before harmonization with analytical script
    obs = obs.sel(year=probs.year.values).load()
    obs = obs.sel(time=probs.year.values).load()

    # Trick to align couples of issue months inside apply_ufunc
    probs_ready = probs.sel(
        issue=np.uint8(params.issue)[:-1]
    ).load()  # use start/end season here
    probs_set = probs.sel(issue=np.uint8(params.issue)[1:]).load()
    probs_set["issue"] = [i - 1 if i != 1 else 12 for i in probs_set.issue.values]

    # Distribute computation of triggers
    trigs, score = xr.apply_ufunc(
        find_optimal_triggers,
        obs.bool,
        obs.val,
        probs_ready.prob,
        probs_set.prob,
        obs.lead_time,
        probs.issue,
        obs.category,
        obs.vulnerability,
        vectorize=True,
        join="outer",
        input_core_dims=[["year"], ["time"], ["year"], ["year"], [], [], [], []],
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

    trigs.to_zarr(
        f"data/MOZ/outputs/Plots/triggers_{params.index}_{params.year}_NRT.zarr", mode="w"
    )
    score.to_zarr(
        f"data/MOZ/outputs/Plots/score_{params.index}_{params.year}_NRT.zarr", mode="w"
    )
    logging.info(f"Triggers and score datasets saved as a back-up")

    # Reset cells of xarray of no interest as nan
    trigs = trigs.where(probs.prob.count("year") != 0, np.nan)
    score = score.where(probs.prob.count("year") != 0, np.nan)

    # Format trigs and score into a dataframe
    trigs_df = triggers_da_to_df(trigs, score).dropna()
    trigs_df = trigs_df.loc[
        trigs_df['HR'] < 0
    ]  # remove row when trigger not found (penalty)

    # Add window information depending on district
    trigs_df["Window"] = [
        get_window_district("MOZ", row["index"].split(" ")[-1], row.district)
        for _, row in trigs_df.iterrows()
    ]

    # Filter per lead time
    df_leadtime = pd.concat(
        [
            g.sort_values(["index", "issue"]).sort_values("HR", kind="stable").head(2)
            for _, g in trigs_df.sort_values("HR").groupby(
                ["category", "district", "Window", "lead_time"],
                as_index=False,
                sort=False,
            )
        ]
    )

    # Keep two pairs of triggers per window of activation
    df_window = filter_triggers_by_window(
        df_leadtime,
        probs_ready,
        probs_set,
        obs,
    )

    df_window.to_csv(
        "data/MOZ/outputs/Plots/triggers.aa.python.{params.index}.{params.year}.NRT.csv",
        index=False,
    )

    client.close()


# Define some constants
TOLERANCE = dict(Leve=0, Moderado=-0.44, Severo=-0.68)
GENERAL_T = dict(HR=0.5, SR=0.65, FR=0.35, RP=dict(Leve=4, Moderado=5, Severo=7))
NON_REGRET_T = dict(HR=0.65, SR=0.55, FR=0.45, RP=dict(Leve=3, Moderado=4, Severo=6))


def find_optimal_triggers(
    observations_bool,
    observations_val,
    prob_issue0,
    prob_issue1,
    lead_time,
    issue,
    category,
    vulnerability,
):
    """
    Calculate optimal triggers by minimizing Hit Rate (1) and False Alarm Ratio (2) for two consecutive months

    Args:
        observations_bool: list or np.array, rainfall observations (categorical) for specific index
        observations_val: list or np.array, rainfall observations (numerical) for specific index
        prob_issue0: list or np.array, rainfall forecasts for specific index and issue month
        prob_issue1: list or np.array, rainfall forecasts for specific index and issue month (+1)
        lead_time: int, first month of index period
        issue: int, issue month of forecasts
        category: str, category name
        vulnerability: str, "GT" for General Triggers or "NRT" for Non-Regret Triggers
    Returns:
        best_triggers: np.array, couple of optimal triggers
        best_score: float, score corresponding to selected triggers
    """

    # Define grid
    threshold_range = (0.0, 1.0)
    grid = (
        slice(threshold_range[0], threshold_range[1], 0.01),
        slice(threshold_range[0], threshold_range[1], 0.01),
    )

    # Launch research
    best_triggers, best_score, _, _ = brute(
        objective,
        grid,
        args=(
            observations_val,
            observations_bool,
            prob_issue0,
            prob_issue1,
            lead_time,
            issue,
            category,
            vulnerability,
        ),
        full_output=True,
        finish=None,
    )

    return best_triggers, best_score


def objective(
    t,
    obs_val,
    obs_bool,
    prob_issue0,
    prob_issue1,
    leadtime,
    issue,
    category,
    vulnerability,
    end_season=5,
    penalty=1e6,
    alpha=10e-3,
    sorting=False,
):
    if leadtime <= end_season:
        obs_val = obs_val[1:]
        obs_bool = obs_bool[1:]
        prob_issue0 = prob_issue0[:-1]
        prob_issue1 = prob_issue1[:-1]

    t = np.array(t).reshape(2, 1)

    prediction = np.logical_and(prob_issue0 > t[0, :], prob_issue1 > t[1, :])

    cm = confusion_matrix(obs_bool, prediction, labels=[0, 1])
    _, false, fn, hits = cm.ravel()

    number_actions = np.sum(prediction)

    far = false / (false + hits)
    false_tol = np.sum(prediction & (obs_val > TOLERANCE[category]))
    hit_rate = hits / (hits + fn)
    success_rate = hits + false - false_tol
    failure_rate = false_tol

    freq = number_actions / len(obs_val)
    return_period = np.round(1 / freq if freq != 0 else 0, 0)

    requirements = GENERAL_T if vulnerability == "GT" else NON_REGRET_T

    constraints = [
        hit_rate >= requirements["HR"],
        success_rate >= (requirements["SR"] * number_actions),
        failure_rate <= (requirements["FR"] * number_actions),
        return_period >= requirements["RP"][category],
        (leadtime - (issue + 1)) % 12 > 1,
    ]

    if not sorting:
        if not (all(constraints)):
            return penalty
        else:
            return -hit_rate + alpha * far
    else:
        return -hit_rate, failure_rate / number_actions


def filter_triggers_by_window(df_leadtime, probs_ready, probs_set, obs):
    def sel_row(da, row, index, issue=None):
        da_sel = da.sel(district=row.district.unique(), index=index)
        if "issue" in da.dims:
            da_sel = da_sel.sel(issue=issue)
        if "category" in da.dims:
            da_sel = da_sel.sel(category=row.category.unique())
        return da_sel

    def get_two_pairs_per_window(tdf):
        for (ind, iss), sub_tdf in tdf.groupby(["index", "issue"]):
            t = sub_tdf.sort_values("trigger").trigger_value.values
            issue = sub_tdf.issue.unique()
            hr, fr = objective(
                t,
                sel_row(obs.val, tdf, ind).values[0],
                sel_row(obs.bool, tdf, ind).values[0][0],
                sel_row(probs_ready.prob, tdf, ind, issue).values[0][0][0],
                sel_row(probs_set.prob, tdf, ind, issue).values[0][0][0],
                sel_row(obs, tdf, ind).lead_time.values,
                sel_row(probs_ready.prob, tdf, ind, issue).issue.values,
                str(sel_row(obs.bool, tdf, ind).category.values[0]),
                str(sel_row(obs.bool, tdf, ind).vulnerability.values),
                sorting=True,
            )
            tdf.loc[(tdf["index"] == ind) & (tdf.issue == iss), "HR"] = hr
            tdf.loc[(tdf["index"] == ind) & (tdf.issue == iss), "FR"] = fr
        if len(tdf) < 4:  # more than two pairs otherwise no need
            return tdf
        else:
            best_four = (
                tdf.sort_values("lead_time").sort_values("FR").sort_values("HR").head(4)
            )
            return best_four

    triggers_window_list = [
        get_two_pairs_per_window(r)
        for _, r in df_leadtime.groupby(["district", "category", "Window"])
    ]

    return pd.concat(triggers_window_list)


def get_window_district(iso, index, district):
    if iso == "MOZ":
        if district == "Chiure":
            if index in ["DJ", "DJF", "JF", "JFM", "FM"]:
                return "Window1"
            elif index in ["FMA", "MA", "MAM", "AM", "AMJ", "MJ"]:
                return "Window2"
            else:
                return np.nan
        elif district in ["Changara", "Marara", "Caia", "Chemba"]:
            if index in ["ND", "NDJ", "DJ", "DJF", "JF"]:
                return "Window1"
            elif index in ["JFM", "FM", "FMA", "MA", "MAM", "AM"]:
                return "Window2"
            else:
                return np.nan
        else:
            assert district in [
                "Chicualacuala",
                "Guija",
                "Massingir",
                "Chibuto",
                "Mabalane",
                "Mapai",
            ]
            if index in ["ON", "OND", "ND", "NDJ", "DJ"]:
                return "Window1"
            elif index in ["DJF", "JF", "JFM", "FM", "FMA", "MA"]:
                return "Window2"
            else:
                return np.nan


def read_aggregated_obs(path_to_zarr, params):
    list_index_paths = glob.glob(f"{path_to_zarr}/{params.index} *")
    list_val_paths = [os.path.join(l, "observations.zarr") for l in list_index_paths]

    obs_val = xr.open_mfdataset(
        list_val_paths,
        engine="zarr",
        preprocess=lambda ds: ds["precip"],
        combine="nested",
        concat_dim="index",
    )
    obs_bool = concat_obs_levels(obs_val, levels=params.intensity_thresholds)

    obs = xr.Dataset({"bool": obs_bool, "val": obs_val})
    obs["time"] = [pd.to_datetime(t).year for t in obs.time.values]
    obs["index"] = [i.split("\\")[-2] for i in list_val_paths]
    return obs


def read_aggregated_probs(path_to_zarr, params):
    list_issue_paths = glob.glob(f"{path_to_zarr}/*")
    list_index = {}
    for l in list_issue_paths:
        list_index_raw = [
            os.path.join(i, "probabilities.zarr")
            for i in glob.glob(f"{l}/{params.index} *")
        ]
        list_index_bc = [
            os.path.join(i, "probabilities_bc.zarr")
            for i in glob.glob(f"{l}/{params.index} *")
        ]
        index_names = [i.split("\\")[-2] for i in list_index_raw]

        try:
            index_raw = xr.open_mfdataset(
                list_index_raw,
                engine="zarr",
                preprocess=lambda ds: ds["tp"],
                combine="nested",
                concat_dim="index",
            )
            index_bc = xr.open_mfdataset(
                list_index_bc,
                engine="zarr",
                preprocess=lambda ds: ds["scen"],
                combine="nested",
                concat_dim="index",
            )

            ds_index = xr.Dataset({"raw": index_raw, "bc": index_bc})
            ds_index["index"] = index_names
            list_index[int(l.split("\\")[-1])] = ds_index
        except:
            continue

    return xr.concat(list_index.values(), dim=pd.Index(list_index.keys(), name="issue"))


if __name__ == "__main__":
    # From AA repository:
    # $ python triggers.py MOZ SPI

    run()
