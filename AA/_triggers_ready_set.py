import logging
import os

import numpy as np
import xarray as xr
from hip.analysis.compute.utils import persist_with_progress_bar
from numba import guvectorize
from tqdm import tqdm


@guvectorize(
    ["void(int8[:], int8[:], int8[:], int8[:])"], "(n), (n), (m) -> (m)", nopython=True
)
def compute_confusion_matrix(true, pred, out_shape, result):
    """
    Computes a confusion matrix using numpy for two np.arrays
    true and pred.

    Results are identical (and similar in computation time) to:
    "from sklearn.metrics import confusion_matrix"

    However, this function avoids the dependency on sklearn and
    allows to use numba in nopython mode.
    """
    # TODO move to hip-analysis
    for i in range(len(true)):
        result[true[i] * 2 + pred[i]] += 1


@guvectorize(
    [
        "void(float64[:], float64[:], int8[:], float64[:], float64[:], int64, int64, float64, int64, float64, float64, float64, float64, float64, float64, float64, int8[:], float64[:])"
    ],
    "(k), (n), (n), (n), (n), (), (), (), (), (), (), (), (), (), (), (), (m) -> (m)",
    nopython=True,
)
def objective(
    t,
    obs_val,
    obs_bool,
    prob_issue0,
    prob_issue1,
    leadtime,
    issue,
    tolerance,
    filter_constraints,
    min_return_period,
    min_hit_rate,
    min_success_rate,
    max_failure_rate,
    penalty,
    alpha,
    eps,
    out_shape,
    result,
):
    # Convert ready / set into single prediction
    prediction = np.logical_and(prob_issue0 > t[0], prob_issue1 > t[1]).astype(np.int8)

    # Get confusion matrix
    conf_matrix = np.zeros(4, dtype=np.int8)
    compute_confusion_matrix(
        obs_bool.astype(np.int8),
        prediction,
        np.empty(4, dtype=np.int8),
        conf_matrix,
    )
    misses, false, fn, hits = (
        conf_matrix[0],
        conf_matrix[1],
        conf_matrix[2],
        conf_matrix[3],
    )

    # Avoid divisions by zero
    if hits + false == 0:
        result[:] = np.array([penalty] * 9)

    # Compute metrics
    number_actions = np.sum(prediction)
    false_alarm_rate = false / (false + hits + eps)
    false_tol = np.sum(prediction & (obs_val > tolerance))
    hit_rate = np.round(hits / (hits + fn + eps), 3)
    success_rate = hits + false - false_tol
    failure_rate = false_tol
    return_period = np.round(
        len(obs_val) / number_actions if number_actions != 0 else 0, 0
    )

    if filter_constraints:
        constraints = np.array(
            [
                hit_rate >= min_hit_rate,
                success_rate >= (min_success_rate * number_actions),
                failure_rate <= (max_failure_rate * number_actions),
                return_period >= min_return_period,
                (leadtime - (issue + 1)) % 12 > 1,
            ]
        ).astype(np.int16)

        if np.all(constraints):
            result[:] = np.array([-hit_rate + alpha * false_alarm_rate] + 8 * [np.nan])
        else:
            result[:] = np.array([penalty] + 8 * [np.nan])

    else:
        result[:] = np.array(
            [
                misses,
                false,
                fn,
                hits,
                hit_rate,
                false_alarm_rate,
                success_rate / (number_actions + eps),
                failure_rate / (number_actions + eps),
                return_period,
            ]
        )


@guvectorize(
    [
        "(float64[:], int8[:], float64[:], float64[:], int64, int64, float64, int64, int64, float64, float64, float64, int8[:], float64[:])"
    ],
    "(m), (m), (m), (m), (), (), (), (), (), (), (), (), (p) -> (p)",
    nopython=True,
)
def brute_numba_vec(
    observations_val,
    observations_bool,
    prob_ready,
    prob_set,
    lead_time,
    issue,
    tolerance,
    vulnerability,
    min_return_period,
    min_hit_rate,
    min_success_rate,
    max_failure_rate,
    out_shape,
    result,
):
    """
    Vectorized version of brute_numba using guvectorize, to evaluate the objective function over a grid.
    """
    xmin, ymin = 0, 0
    min_value = 1e6

    # Iterate over the index pairs directly
    for i, j in zip(range(0, 100), range(0, 100)):
        t1, t2 = i / 100, j / 100  # Generate (t1, t2) on the fly

        out = np.empty(9, dtype=np.float64)
        objective(
            np.array([t1, t2], dtype=np.float64),  # Avoids storing grid_flat
            observations_val,
            observations_bool,
            prob_ready,
            prob_set,
            lead_time,
            issue,
            tolerance,
            vulnerability,
            min_return_period,
            min_hit_rate,
            min_success_rate,
            max_failure_rate,
            1e6,
            10e-3,
            1e-6,
            np.empty(9, dtype=np.int8),
            out,
        )
        if out[0] < min_value:
            xmin, ymin = t1, t2
            min_value = out[0]

    # Store the result: x, y, and minimum function value
    result[0] = xmin
    result[1] = ymin
    result[2] = min_value


def find_optimal_triggers_parallel(
    observations_val,
    observations_bool,
    prob_ready,
    prob_set,
    lead_time,
    issue,
    tolerance,
    vulnerability,
    min_return_period,
    min_hit_rate,
    min_success_rate,
    max_failure_rate,
):
    """
    Find the optimal triggers pair by evaluating the objective function on each couple of
    values of a 100 * 100 grid and selecting the minimizer.

    Args:
        observations_val: np.array, time series of the observed rainfall values (or SPI)
        observations_bool: np.array, time series of categorical observations for the
                        specified category
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
    result = np.ones(3, dtype=np.float64) * -1

    # Launch research
    brute_numba_vec(
        observations_val,
        observations_bool.astype(np.int8),
        prob_ready,
        prob_set,
        lead_time,
        issue,
        tolerance,
        vulnerability,
        min_return_period,
        min_hit_rate,
        min_success_rate,
        max_failure_rate,
        np.empty(3, dtype=np.int8),
        result,
    )

    best_triggers = result[:2]
    best_score = result[2]

    return best_triggers, best_score


def run_ready_set_brute_selection(obs, probs_ready, probs_set, probs, params):
    """
    Run the trigger optimization using xarray's apply_ufunc and Dask parallelization.

    Parameters
    ----------
    obs : xarray.Dataset
        Dataset containing observed values ('val'), boolean event occurrence ('bool'),
        lead time ('lead_time'), tolerance ('tolerance'), and return period ('return_period').
    probs_ready : xarray.Dataset
        Dataset containing the forecast probability used for readiness ('prob').
    probs_set : xarray.Dataset
        Dataset containing the forecast probability used for activation ('prob').
    probs : xarray.Dataset
        Dataset containing the forecast issue time ('issue').
    params : object
        Params class containing 'vulnerability' and 'requirements' (with 'HR', 'SR', 'FR').

    Returns
    -------
    triggers : xarray.DataArray
        Array of optimal triggers pairs.
    score : xarray.DataArray
        Optimal score (hit rate + alpha * false_alarm_rate) for each configuration.
    """
    triggers, score = xr.apply_ufunc(
        find_optimal_triggers_parallel,
        obs.val,
        obs.bool,
        probs_ready.prob,
        probs_set.prob,
        obs.lead_time,
        probs.issue,
        obs.tolerance,
        np.int64(params.vulnerability in ["GT", "NRT"]),
        obs.return_period,
        np.float64(params.requirements["HR"]),
        np.float64(params.requirements["SR"]),
        np.float64(params.requirements["FR"]),
        vectorize=True,
        join="outer",
        input_core_dims=[
            ["time"],
            ["time"],
            ["time"],
            ["time"],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        ],
        output_core_dims=[["trigger"], []],
        output_sizes={"trigger": 2},
        output_dtypes=[np.int8, np.float64],
        dask="parallelized",
        keep_attrs=True,
    )

    triggers = persist_with_progress_bar(triggers)
    score = persist_with_progress_bar(score)

    triggers["trigger"] = ["trigger1", "trigger2"]

    triggers["category"] = triggers.category.astype(str)
    triggers["district"] = triggers.district.astype(str)
    triggers["index"] = triggers.index.astype(str)

    score["category"] = score.category.astype(str)
    score["district"] = score.district.astype(str)
    score["index"] = score.index.astype(str)

    return triggers, score


def evaluate_grid_metrics(
    obs,
    probs_ready,
    probs_set,
):
    """
    Evaluate all metrics over the entire grid using apply_ufunc.
    The list of metrics is:

    Args:
        obs: xarray.Dataset, containing numerical and categorical observations
        prob_ready: xarray.DataArray, forecast probabilities for the ready month
        prob_set: xarray.DataArray, forecast probabilities for the set month

    Returns:
        metrics_da: xarray.DataArray, structured array with grid evaluations for all metrics
    """
    # Ensure category alignment
    probs_ready = probs_ready.reindex(category=obs.category.coords["category"].values)
    probs_set = probs_set.reindex(category=obs.category.coords["category"].values)

    # Define the threshold grid
    thresholds = np.arange(0.0, 1.0, step=0.01)
    grid_ready, grid_set = np.meshgrid(thresholds, thresholds, indexing="ij")

    # Combine thresholds into pairs
    grid_thresholds = np.stack(
        [grid_ready, grid_set], axis=-1
    )  # Shape: (ready, set, 2)

    grid_da = xr.DataArray(
        grid_thresholds,
        dims=["ready", "set", "threshold"],
        coords={"ready": thresholds, "set": thresholds, "threshold": ["ready", "set"]},
    )

    out_shape = xr.DataArray(np.empty(9, dtype=np.int8), dims=["metric"])

    # Apply the `objective` function across the threshold grid
    metrics_da = xr.apply_ufunc(
        objective,
        grid_da,
        obs.val,
        obs.bool,
        probs_ready.prob,
        probs_set.prob,
        obs.lead_time,
        probs_ready.issue,
        obs.tolerance,
        np.int64(0),  # vulnerability
        np.int64(-1),  # min_return_period
        np.float64(-1),  # min_hit_rate
        np.float64(-1),  # min_success_rate
        np.float64(10),  # max_failure_rate
        np.float64(1e6),  # penalty
        np.float64(10e-3),  # alpha
        np.float64(1e-6),  # eps
        out_shape,
        input_core_dims=[
            ["threshold"],
            ["time"],
            ["time"],
            ["time"],
            ["time"],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            ["metric"],
        ],
        output_core_dims=[["metric"]],
        output_sizes={"metric": 9},
        output_dtypes=[np.float64],
        vectorize=True,
        join="outer",
        exclude_dims={"metric"},
    )

    metrics_da["metric"] = ["TN", "FP", "FN", "TP", "HR", "FAR", "SR", "FR", "RP"]

    metrics_da["district"] = metrics_da.district.astype(str)
    metrics_da["category"] = metrics_da.category.astype(str)
    metrics_da["metric"] = metrics_da.metric.astype(str)

    return metrics_da


def get_trigger_metrics_dataframe(obs, probs_ready, probs_set, data_path):
    """
    Compute trigger metrics for a single district and save the results as a CSV file.

    Parameters:
    -----------
    obs : xarray.DataArray
        Observations dataset containing `bool`, `val`, `lead_time`, and `category` variables.
    probs_ready : xarray.DataArray
        Dataset containing readiness probabilities with `prob` and `issue` variables.
    probs_set : xarray.DataArray
        Dataset containing set probabilities with `prob` variable.
    data_path : str
        Output folder path.
    """
    grid_metrics_da = evaluate_grid_metrics(obs, probs_ready, probs_set)

    grid_metrics_df = grid_metrics_da.to_dataframe(name="value").reset_index()
    grid_metrics_df = grid_metrics_df.loc[grid_metrics_df.value < 1e5]

    grid_metrics_df["issue_set"] = grid_metrics_df.issue + 1
    grid_metrics_df = grid_metrics_df.rename(columns={"issue": "issue_ready"})

    # Pivot to structured format
    grid_metrics_df = grid_metrics_df.pivot(
        index=[
            "ready",
            "set",
            "index",
            "district",
            "category",
            "issue_ready",
            "lead_time",
            "issue_set",
        ],
        columns="metric",
        values="value",
    ).reset_index()

    # Define output path and save
    output_path = (
        f"{data_path}/triggers_metrics_tbd_{grid_metrics_df.district.unique()[0]}.csv"
    )
    grid_metrics_df.to_csv(output_path, index=False)

    logging.info(f"Metrics saved to {output_path}")


def run_pilot_districts_metrics(obs, probs_ready, probs_set, params):
    """
    Loop through pilot districts and compute trigger metrics.

    Parameters:
    -----------
    obs : xarray.DataArray
        Observational dataset containing `bool`, `val`, `lead_time`, and `category` variables.
    probs_ready : xarray.DataArray
        Dataset containing readiness probabilities with `prob` and `issue` variables.
    probs_set : xarray.DataArray
        Dataset containing set probabilities with `prob` variable.
    params : object
        Configuration parameters including `data_path`, `iso`, and `districts_vulnerability`.
    """
    logging.info(
        f"Starting computation of metrics for {params.iso.upper()} pilot districts..."
    )

    folder_path = f"{params.data_path}/data/{params.iso}/triggers/triggers_metrics"
    os.makedirs(folder_path, exist_ok=True)

    districts = params.districts if params.districts else obs.district.values

    for district in tqdm(districts):
        logging.info(f"Processing district: {district}")
        get_trigger_metrics_dataframe(
            obs.sel(district=district),
            probs_ready.sel(district=district),
            probs_set.sel(district=district),
            folder_path,
        )
