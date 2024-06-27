import logging

import click

logging.basicConfig(level="INFO")

import warnings

warnings.simplefilter(action="ignore")

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr
from config.params import Params
from hip.analysis.analyses.drought import concat_obs_levels, get_accumulation_periods
from hip.analysis.aoi.analysis_area import AnalysisArea
from numba import jit

from AA.helper_fns import (
    create_flexible_dataarray,
    merge_un_biased_probs,
    triggers_da_to_df,
    format_triggers_df_for_dashboard,
)


@click.command()
@click.argument("country", required=True, type=str)
@click.argument("index", default="SPI")
@click.argument("vulnerability", default="GT")
def run(country, index, vulnerability):
    params = Params(iso=country, index=index)

    run_triggers_selection(params, vulnerability)


def run_triggers_selection(params, vulnerability):
    # have function that takes obs / probs and returns triggers
    area = AnalysisArea.from_admin_boundaries(
        iso3=params.iso.upper(),
        admin_level=2,
        resolution=0.25,
        datetime_range=f"1981-01-01/{params.calibration_year}-06-30",
    )

    gdf = area.get_dataset([area.BASE_AREA_DATASET])
    admin1 = area.get_admin_boundaries(iso3=params.iso, admin_level=1).drop(
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
    obs = obs.sel(time=probs.year.values).load()

    # Trick to align couples of issue months inside apply_ufunc
    probs_ready = probs.sel(
        issue=np.uint8(params.issue_months)[:-1]
    ).load()  # use start/end season here
    probs_set = probs.sel(issue=np.uint8(params.issue_months)[1:]).load()
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
        vulnerability,
        params,
        vectorize=True,
        join="outer",
        input_core_dims=[["time"], ["time"], ["year"], ["year"], [], [], [], [], []],
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
        f"{params.data_path}/data/{params.iso}/triggers/triggers_{params.index}_{params.calibration_year}_{vulnerability}.zarr",
        mode="w",
    )
    score.to_zarr(
        f"{params.data_path}/data/{params.iso}/triggers/score_{params.index}_{params.calibration_year}_{vulnerability}.zarr",
        mode="w",
    )
    logging.info(f"Triggers and score datasets saved as a back-up")

    # Reset cells of xarray of no interest as nan
    trigs = trigs.where(probs.prob.count("year") != 0, np.nan)
    score = score.where(probs.prob.count("year") != 0, np.nan)

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

    triggers.to_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.{params.index}.{params.calibration_year}.{vulnerability}.csv",
        index=False,
    )

    logging.info(
        f"Triggers dataframe saved as a csv for {params.index} {vulnerability}"
    )


@jit(nopython=True, cache=True)
def _compute_confusion_matrix(true, pred):
    # TODO move to hip-analysis
    """
    Computes a confusion matrix using numpy for two np.arrays
    true and pred.

    Results are identical (and similar in computation time) to:
    "from sklearn.metrics import confusion_matrix"

    However, this function avoids the dependency on sklearn and
    allows to use numba in nopython mode.
    """

    K = 2  # len(np.unique(true)) # Number of classes
    result = np.zeros((K, K))

    for i in range(len(true)):
        result[true[i]][pred[i]] += 1

    return result


@jit(
    nopython=True,
    cache=True,
)
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
    tolerance,
    general_req,
    non_regret_req,
    end_season=6,
    penalty=1e6,
    alpha=10e-3,
    sorting=False,
    eps=1e-6,
):
    # Align obs and probs when leadtime between Jan and end_season
    # TODO align these time steps at the analytical level
    if leadtime <= end_season:
        obs_val = obs_val[1:]
        obs_bool = obs_bool[1:]
        prob_issue0 = prob_issue0[:-1]
        prob_issue1 = prob_issue1[:-1]

    prediction = np.logical_and(prob_issue0 > t[0], prob_issue1 > t[1]).astype(np.int16)

    cm = _compute_confusion_matrix(obs_bool.astype(np.int16), prediction)
    _, false, fn, hits = cm.astype(np.int16).ravel()

    number_actions = np.sum(prediction)

    if hits + false == 0:  # avoid divisions by zero
        return [penalty, penalty] if sorting else [penalty]

    false_alarm_rate = false / (false + hits + eps)
    false_tol = np.sum(prediction & (obs_val > tolerance[category]))
    hit_rate = np.round(hits / (hits + fn + eps), 3)
    success_rate = hits + false - false_tol
    failure_rate = false_tol

    freq = number_actions / len(obs_val)
    return_period = np.round(1 / freq if freq != 0 else 0, 0)

    requirements = general_req if vulnerability == "GT" else non_regret_req
    req_RP = (
        requirements["RP"]
        + 1 * (category[:3].lower() == "mod")
        + 3 * (category[:3].lower() == "sev")
    )

    constraints = np.array(
        [
            hit_rate >= requirements["HR"],
            success_rate >= (requirements["SR"] * number_actions),
            failure_rate <= (requirements["FR"] * number_actions),
            return_period >= req_RP,
            (leadtime - (issue + 1)) % 12 > 1,
        ]
    ).astype(np.int16)

    if sorting:
        return [-hit_rate, failure_rate / (number_actions + eps)]
    else:
        if np.all(constraints):
            return [-hit_rate + alpha * false_alarm_rate]
        else:
            return [penalty]


@jit(nopython=True)
def _make_grid(arraylist):
    n = len(arraylist)
    k = arraylist[0].shape[0]
    a2d = np.zeros((n, k, k))
    for i in range(n):
        a2d[i] = arraylist[i]
    return a2d


@jit(nopython=True)
def _meshxy(x, y):
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for i in range(y.size):
        for j in range(x.size):
            xx[i, j] = x[i]  # change to x[j] if indexing xy
            yy[i, j] = y[j]  # change to y[i] if indexing xy
    return xx, yy


@jit(nopython=True)
def brute_numba(func, ranges, args=()):
    # TODO Move to hip-analysis
    """
    Numba-compatible implementation of scipy.optimize.brute designed only for 2d minimizations.
    Minimize a function over a given range by brute force.

    Uses the “brute force” method, i.e., computes the function's value at each point of a
    multidimensional grid of points, to find the global minimum of the function.

    Args:
        func: callable, objective function to be minimized. Must be in the form f(x, *args),
                        where x is the argument in the form of a 1-D array and args is a tuple
                        of any additional fixed parameters needed to completely specify the
                        function.
        ranges: tuple,  each component of the ranges tuple must be a numpy.array. The program uses
                        these to create the grid of points on which the objective function will be
                        computed.
        args: tuple, optional, any additional fixed parameters needed to completely specify the
                        function.
    Returns:
        xmin: numpy.ndarray, a 1-D array containing the coordinates of a point at which the
                        objective function had its minimum value.
        Jmin: float, function values at the point xmin.
    """
    assert len(ranges) == 2

    x, y = _meshxy(*ranges)
    grid = _make_grid([x, y])

    # obtain an array of parameters that is iterable by a map-like callable
    inpt_shape = np.array(grid.shape)
    grid = np.reshape(grid, (inpt_shape[0], np.prod(inpt_shape[1:]))).T

    # iterate over input arrays and evaluate func (1D array)
    Jout = np.array(
        [func(np.asarray(candidate).flatten(), *args)[0] for candidate in grid]
    )

    # identify index of minimizer in 1D array
    indx = np.argmin(Jout)

    # reshape to recover 2D grid
    Jout = np.reshape(Jout, (inpt_shape[1], inpt_shape[2]))
    grid = np.reshape(grid.T, (inpt_shape[0], inpt_shape[1], inpt_shape[2]))

    # identify index of minimizer in grid
    Nshape = np.shape(Jout)
    Nindx = np.empty(2, dtype=np.uint8)
    Nindx[1] = indx % Nshape[1]
    indx = indx // Nshape[1]
    Nindx[0] = indx % Nshape[0]

    # get candidate value that minimizes func
    xmin = np.array([grid[k][Nindx[0], Nindx[1]] for k in range(2)])

    # retrieve func minimum when evaluated on ranges
    Jmin = Jout[Nindx[0], Nindx[1]]

    return xmin, Jmin


def find_optimal_triggers(
    observations_bool,
    observations_val,
    prob_ready,
    prob_set,
    lead_time,
    issue,
    category,
    vulnerability,
    params,
):
    """
    Find the optimal triggers pair by evaluating the objective function on each couple of
    values of a 100 * 100 grid and selecting the minimizer.

    Args:
        observations_bool: np.array, time series of categorical observations for the
                        specified category
        observations_val: np.array, time series of the observed rainfall values (or SPI)
        prob_ready: np.array, time series of forecasts probabilities for the ready month
        prob_ready: np.array, time series of forecasts probabilities for the set month
        lead_time: int, lead time month
        issue: int, issue month
        category: str, intensity level
        vulnerability: str, should be either 'GT' for General Triggers or 'NRT' for Non-
                        Regret Triggers
        params: Params, configuration parameters dictionary
    Returns:
        best_triggers: np.array, array of size 2 containing best triggers for Ready / Set
        best_score: int, score (mainly hit rate) corresponding to the best triggers
    """

    # Define grid
    threshold_range = (0.0, 1.0)
    grid = (
        np.arange(threshold_range[0], threshold_range[1], step=0.01),
        np.arange(threshold_range[0], threshold_range[1], step=0.01),
    )

    # Launch research
    best_triggers, best_score = brute_numba(
        objective,
        grid,
        args=(
            observations_val,
            observations_bool,
            prob_ready,
            prob_set,
            lead_time,
            issue,
            category,
            vulnerability,
            params.tolerance,
            params.general_t,
            params.non_regret_t,
        ),
    )

    return best_triggers, best_score


def filter_triggers_by_window(
    df_leadtime, probs_ready, probs_set, obs, vulnerability, params
):
    def sel_row(da, row, index, issue=None):
        da_sel = da.sel(district=row.district.unique(), index=index)
        if "issue" in da.dims:
            da_sel = da_sel.sel(issue=issue)
        if "category" in da.dims:
            da_sel = da_sel.sel(category=row.category.unique())
        return da_sel

    def get_top_pairs_per_window(tdf, n_to_keep=4):
        for (ind, iss), sub_tdf in tdf.groupby(["index", "issue"]):
            t = sub_tdf.sort_values("trigger").trigger_value.values
            issue = sub_tdf.issue.unique()
            stats = tuple(
                objective(
                    t,
                    sel_row(obs.val, tdf, ind).values[0],
                    sel_row(obs.bool, tdf, ind).values[0][0],
                    sel_row(probs_ready.prob, tdf, ind, issue).values[0][0][0],
                    sel_row(probs_set.prob, tdf, ind, issue).values[0][0][0],
                    sel_row(obs, tdf, ind).lead_time.values,
                    sel_row(probs_ready.prob, tdf, ind, issue).issue.values[0],
                    str(sel_row(obs.bool, tdf, ind).category.values[0]),
                    vulnerability,
                    params.tolerance,
                    params.general_t,
                    params.non_regret_t,
                    sorting=True,
                )
            )
            hr, fr = stats[0], stats[1]
            tdf.loc[(tdf["index"] == ind) & (tdf.issue == iss), "HR"] = hr
            tdf.loc[(tdf["index"] == ind) & (tdf.issue == iss), "FR"] = fr
        if len(tdf) < (2 * n_to_keep):  # more than two pairs otherwise no need
            return tdf
        else:
            best_four = (
                tdf.sort_values("lead_time")
                .sort_values("FR")
                .sort_values("HR")
                .head(2 * n_to_keep)
            )
            return best_four

    triggers_window_list = [
        get_top_pairs_per_window(r)
        for _, r in df_leadtime.groupby(["district", "category", "Window"])
    ]

    return pd.concat(triggers_window_list)


def get_window_district(shp, indicator, district, params):
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
        preprocess=lambda ds: ds["band"],
        combine="nested",
        concat_dim="index",
    )
    obs_bool = concat_obs_levels(obs_val, levels=params.intensity_thresholds)

    obs = xr.Dataset({"bool": obs_bool, "val": obs_val})
    obs["time"] = [pd.to_datetime(t).year for t in obs.time.values]
    obs["index"] = [i.split("\\")[-2] for i in list_val_paths]
    return obs


def read_aggregated_probs(path_to_zarr, params):
    list_issue_paths = glob.glob(f"{path_to_zarr}/*")[:-1]  # remove obs folder
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
    # $ python triggers.py MOZ SPI NRT

    run()
