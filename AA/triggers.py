import click
import logging

logging.basicConfig(level="INFO")

import warnings

warnings.simplefilter(action="ignore")

from numba import jit
from numba.core import types
from numba.typed import Dict
from dask.distributed import Client

import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

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
@click.argument("vulnerability", default="GT")
def run(country, index, vulnerability):
    client = Client()

    params = Params(iso=country, index=index)

    gdf = gpd.read_file(
        f"data/{params.iso}/shapefiles/moz_admbnda_2019_SHP/moz_admbnda_adm2_2019.shp"
    )

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
        f"data/{params.iso}/outputs/zarr/obs/2022_all_districts",
        params,
    )
    obs = obs.where(obs.index != f"{index} JDJ", drop=True)
    obs = obs.assign_coords(
        lead_time=("index", [periods[i.split(" ")[-1]][0] for i in obs.index.values])
    )
    logging.info(
        f"Completed reading of aggregated observations for the whole {params.iso} country"
    )

    probs_ds = read_aggregated_probs(
        f"data/{params.iso}/outputs/zarr/2022_all_districts",
        params,
        
    )
    # TODO fix in get_accumulation_periods
    probs_ds = probs_ds.where(probs_ds.index != f"{index} JDJ", drop=True)
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
        vulnerability,
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
        f"data/MOZ/outputs/Plots/all_districts/triggers_{params.index}_{params.year}_{vulnerability}.zarr", mode="w"
    )
    score.to_zarr(
        f"data/MOZ/outputs/Plots/all_districts/score_{params.index}_{params.year}_{vulnerability}.zarr", mode="w"
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
        get_window_district("MOZ", gdf, row["index"].split(" ")[-1], row.district)
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
        vulnerability,
    )

    df_window.to_csv(
        "data/MOZ/outputs/Plots/triggers.aa.python.{params.index}.{params.year}.{vulnerability}.all.districts.csv",
        index=False,
    )

    client.close()


# Define some constants
# The Dict.empty() constructs a typed dictionary.
TOLERANCE = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.f8,
)
TOLERANCE['Leve'] = 0; TOLERANCE['Moderado'] = -0.44; TOLERANCE['Severo'] = -0.68

GENERAL_T = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.f8,
)
GENERAL_T['HR'] = 0.5; GENERAL_T['SR'] = 0.65; GENERAL_T['FR'] = 0.35; GENERAL_T['RP'] = 4.

NON_REGRET_T = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.f8,
)
NON_REGRET_T['HR']=0.65; NON_REGRET_T['SR']=0.55; NON_REGRET_T['FR']=0.45; NON_REGRET_T['RP'] = 3.


@jit(nopython=True, cache=True)
def _compute_confusion_matrix(true, pred):
  # TODO move to hip-analysis
  '''
  Computes a confusion matrix using numpy for two np.arrays
  true and pred.

  Results are identical (and similar in computation time) to: 
    "from sklearn.metrics import confusion_matrix"

  However, this function avoids the dependency on sklearn and 
  allows to use numba in nopython mode.
  '''

  K = 2 #len(np.unique(true)) # Number of classes 
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
    end_season=5,
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

    if hits + false == 0: # avoid divisions by zero
       return [penalty]
    
    false_alarm_rate = false / (false + hits + eps)
    false_tol = np.sum(prediction & (obs_val > tolerance[category]))
    hit_rate = hits / (hits + fn + eps)
    success_rate = hits + false - false_tol
    failure_rate = false_tol
    
    freq = number_actions / len(obs_val)
    return_period = np.round(1 / freq if freq != 0 else 0, 0)
    
    requirements = general_req if vulnerability == "GT" else non_regret_req
    req_RP = requirements['RP'] + 1 * (category[0]=='M') + 3 * (category[0]=='S')
    
    constraints = np.array([
        hit_rate >= requirements["HR"],
        success_rate >= (requirements["SR"] * number_actions),
        failure_rate <= (requirements["FR"] * number_actions),
        return_period >= req_RP,
        (leadtime - (issue + 1)) % 12 > 1,
    ]).astype(np.int16)
    
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
    return(a2d)


@jit(nopython=True)
def _meshxy(x, y):
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for i in range(y.size):
        for j in range(x.size):
            xx[i,j] = x[i]  # change to x[j] if indexing xy
            yy[i,j] = y[j]  # change to y[i] if indexing xy
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
    Jout = np.array([
        func(np.asarray(candidate).flatten(), *args)[0]
        for candidate in grid
    ])
    
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
        vulnerability: str, should be either 'GT' for General Triggers or 'NRT' for 'Non-
                        Regret Triggers'
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
            TOLERANCE,
            GENERAL_T,
            NON_REGRET_T,
        ),
    )

    return best_triggers, best_score


def filter_triggers_by_window(df_leadtime, probs_ready, probs_set, obs, vulnerability):
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
                sel_row(probs_ready.prob, tdf, ind, issue).issue.values[0],
                str(sel_row(obs.bool, tdf, ind).category.values[0]),
                vulnerability,
                TOLERANCE,
                GENERAL_T,
                NON_REGRET_T,
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


WINDOW1_INDEXES_MOZ = {
    'Cabo_Delgado': ['DJ', 'DJF', 'JF', 'JFM', 'FM'], 
    'Gaza': ['ON', 'OND', 'ND', 'NDJ', 'DJ'], 
    'Inhambane': ['ON', 'OND', 'ND', 'NDJ', 'DJ'], 
    'Manica': ['ND', 'NDJ', 'DJ', 'DJF', 'JF'], 
    'Maputo': ['ON', 'OND', 'ND', 'NDJ', 'DJ'],
    'Maputo City': ['ON', 'OND', 'ND', 'NDJ', 'DJ'], 
    'Nampula': ['DJ', 'DJF', 'JF', 'JFM', 'FM'], 
    'Niassa': ['DJ', 'DJF', 'JF', 'JFM', 'FM'], 
    'Sofala': ['ND', 'NDJ', 'DJ', 'DJF', 'JF'],  
    'Tete': ['ND', 'NDJ', 'DJ', 'DJF', 'JF'],  
    'Zambezia': ['ND', 'NDJ', 'DJ', 'DJF', 'JF'], 
}

WINDOW2_INDEXES_MOZ = {
    'Cabo_Delgado': ['FMA', 'MA', 'MAM', 'AM', 'AMJ', 'MJ'], 
    'Gaza': ['DJF', 'JF', 'JFM', 'FM', 'FMA', 'MA'], 
    'Inhambane': ['DJF', 'JF', 'JFM', 'FM', 'FMA', 'MA'], 
    'Manica': ['JFM', 'FM', 'FMA', 'MA', 'MAM', 'AM'], 
    'Maputo': ['DJF', 'JF', 'JFM', 'FM', 'FMA', 'MA'], 
    'Maputo City': ['DJF', 'JF', 'JFM', 'FM', 'FMA', 'MA'], 
    'Nampula': ['FMA', 'MA', 'MAM', 'AM', 'AMJ', 'MJ'], 
    'Niassa': ['FMA', 'MA', 'MAM', 'AM', 'AMJ', 'MJ'], 
    'Sofala': ['JFM', 'FM', 'FMA', 'MA', 'MAM', 'AM'], 
    'Tete': ['JFM', 'FM', 'FMA', 'MA', 'MAM', 'AM'], 
    'Zambezia': ['JFM', 'FM', 'FMA', 'MA', 'MAM', 'AM'], 
}

def get_window_district(iso, shp, index, district):
    province = shp.loc[shp.ADM2_PT == district].ADM1_PT.unique()[0]
    if iso == "MOZ":
        if index in WINDOW1_INDEXES_MOZ[province]:
            return "Window1"
        elif index in WINDOW2_INDEXES_MOZ[province]:
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
    # $ python triggers.py MOZ SPI NRT

    run()
