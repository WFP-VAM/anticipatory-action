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

# +
# %cd ../
import logging
import numpy as np 
import pandas as pd
import xarray as xr

from hip.analysis.analyses.drought import get_accumulation_periods

from config.params import Params
from dask.distributed import Client
from triggers import read_aggregated_obs, read_aggregated_probs, merge_un_biased_probs
# -

# ### Load data

# +
client = Client()

params = Params(iso='MOZ', index='SPI')

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
# -

# ### Numba-optimized way

# +
from numba.core import types
from numba.typed import Dict

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

# +
from numba import jit

@jit(nopython=True, cache=True)
def _compute_confusion_matrix(true, pred):
  '''
  Computes a confusion matrix using numpy for two np.arrays
  true and pred.

  Results are identical (and similar in computation time) to: 
    "from sklearn.metrics import confusion_matrix"

  However, this function avoids the dependency on sklearn and 
  allows to use numba in nopython mode.
  '''

  K = len(np.unique(true)) # Number of classes 
  result = np.zeros((K, K))

  for i in range(len(true)):
    result[true[i]][pred[i]] += 1

  return result

@jit(
    nopython=True, 
    cache=True,
)
def objective_numba(
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
):
    if leadtime <= end_season:
        obs_val = obs_val[1:]
        obs_bool = obs_bool[1:]
        prob_issue0 = prob_issue0[:-1]
        prob_issue1 = prob_issue1[:-1]
    
    prediction = np.logical_and(prob_issue0 > t[0], prob_issue1 > t[1]).astype(np.int16)

    cm = _compute_confusion_matrix(obs_bool, prediction)
    _, false, fn, hits = cm.astype(np.int16).ravel()

    number_actions = np.sum(prediction)

    if hits + false == 0: # avoid divisions by zero
       return [penalty]
    
    far = false / (false + hits)
    false_tol = np.sum(prediction & (obs_val > tolerance[category]))
    hit_rate = hits / (hits + fn)
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
        return [-hit_rate, failure_rate / number_actions]
    else:
      if np.all(constraints):
          return [-hit_rate + alpha * far]
      else:
          return [penalty]


# +
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
    
    # iterate over input arrays
    Jout = np.array([
        func(np.asarray(candidate).flatten(), *args)[0]
        for candidate in grid
    ])
    
    indx = np.argmin(Jout)

    Jout = np.reshape(Jout, (inpt_shape[1], inpt_shape[2]))
    grid = np.reshape(grid.T, (inpt_shape[0], inpt_shape[1], inpt_shape[2]))
    
    Nshape = np.shape(Jout)
    Nindx = np.empty(2, dtype=np.uint8)    
    Nindx[1] = indx % Nshape[1]
    indx = indx // Nshape[1]
    Nindx[0] = indx % Nshape[0]
    indx = indx // Nshape[0]
    
    xmin = np.array([grid[k][Nindx[0], Nindx[1]] for k in range(2)])

    Jmin = Jout[Nindx[0], Nindx[1]]

    return xmin, Jmin


# -

def find_optimal_triggers_numba(
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
        objective_numba,
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


# +
# %%time 

# Distribute computation of triggers
trigs, score = xr.apply_ufunc(
    find_optimal_triggers_numba,
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
# -

# ### Validation of numba implementation: comparison with previous Python results (older version of objective function and scipy brute function)

trigs_ref = xr.open_zarr('data/MOZ/outputs/Plots/triggers_spi_2022_GT.zarr').bool
score_ref = xr.open_zarr('data/MOZ/outputs/Plots/score_spi_2022_GT.zarr').bool

xr.testing.assert_equal(trigs_ref, trigs.assign_coords({'trigger': trigs_ref.trigger.values}))

xr.testing.assert_equal(score_ref, score)
