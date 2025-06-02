import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
from hip.analysis.compute.utils import persist_with_progress_bar
from numba import guvectorize
from tqdm import tqdm

ALPHA = 1e-3
EPS = 1e-6


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
    result.fill(0)
    for i in range(len(true)):
        result[true[i] * 2 + pred[i]] += 1


@guvectorize(
    [
        "void(float64[:], float64[:], int8[:], float64[:], float64[:], int64, int64, float64, int64, float64, float64, float64, float64, float64[:], int8[:], int8[:], float64[:])"
    ],
    "(k), (n), (n), (n), (n), (), (), (), (), (), (), (), (), (m), (l), (j) -> (m)",
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
    conf_matrix,
    constraints,
    result,
):
    """
    Compute the objective function value for a given pair of thresholds.

    This function evaluates a set of thresholds (t) applied to two forecast probability time series
    (prob_issue0 and prob_issue1) to generate binary predictions. It then computes the confusion
    matrix (misses, false alarms, false negatives, hits) and derives key performance metrics
    such as hit rate, false alarm rate, success rate, failure rate, and return period.

    If `filter_constraints` is True, it applies a set of operational constraints
    (minimum hit rate, minimum success rate, maximum failure rate, minimum return period, and lead time constraint)
    to determine if the thresholds are acceptable. If all constraints are satisfied,
    it minimizes a combination of hit rate and false alarm rate; otherwise, it assigns a high penalty value.

    If `filter_constraints` is False, it returns the full set of metrics without penalization.

    Args:
        t: np.ndarray, Array of size 2 containing triggers for 'ready' and 'set' forecasts.
        obs_val: np.ndarray, Array of observed continuous anomaly values.
        obs_bool: np.ndarray, Array of observed binary event occurrences (0 or 1).
        prob_issue0: np.ndarray, Forecast probabilities for the 'ready' stage.
        prob_issue1: np.ndarray, Forecast probabilities for the 'set' stage.
        leadtime: int, Indicator period lead time (month).
        issue: int, Forecast issue month.
        tolerance: float, Tolerance threshold for acceptable false alarms.
        filter_constraints: int, If 1, constraints are applied to filter acceptable triggers.
        min_return_period: int, Minimum acceptable return period for actions (in years).
        min_hit_rate: float, Minimum acceptable hit rate.
        min_success_rate: float, Minimum acceptable success rate (hits relative to actions taken).
        max_failure_rate: float, Maximum acceptable failure rate (tolerance-exceeding false alarms relative to actions taken).
        penalty: np.ndarray, Array of penalty values assigned when constraints are not satisfied.
        conf_matrix: np.ndarray, Array to store computed confusion matrix elements [misses, false alarms, false negatives, hits].
        constraints: np.ndarray, Array to temporarily store the boolean results of constraints evaluation.
        result: np.ndarray, Array to store the computed objective score or the list of metrics.


    Notes:
        - The first output when `filter_constraints=True` is a scalar combining hit rate and false alarm rate.
        - When `filter_constraints=False`, the full confusion matrix and performance metrics are returned.
        - Designed for use with Numba `guvectorize` to allow fast parallel evaluation across a large grid of thresholds.
    """
    # Convert ready / set into single prediction
    prediction = np.logical_and(prob_issue0 > t[0], prob_issue1 > t[1]).astype(np.int8)

    # Get confusion matrix
    compute_confusion_matrix(
        obs_bool,
        prediction,
        conf_matrix,
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
        result[:] = penalty

    else:
        # Compute metrics
        number_actions = np.sum(prediction)
        false_alarm_rate = false / (false + hits + EPS)
        false_tol = np.sum(prediction & (obs_val > tolerance))
        hit_rate = np.round(hits / (hits + fn + EPS), 3)
        success_rate = hits + false - false_tol
        failure_rate = false_tol
        return_period = np.round(
            len(obs_val) / number_actions if number_actions != 0 else 0, 0
        )

        # When we need to select the optimal pair of triggers
        if filter_constraints:
            constraints[0] = hit_rate >= min_hit_rate
            constraints[1] = success_rate >= (min_success_rate * number_actions)
            constraints[2] = failure_rate <= (max_failure_rate * number_actions)
            constraints[3] = return_period >= min_return_period
            constraints[4] = (leadtime - (issue + 1)) % 12 > 1

            if np.all(constraints):
                result[0] = -hit_rate + ALPHA * false_alarm_rate
            else:
                result[0] = penalty[0]

        # When we need to retrieve all the metrics
        else:
            result[0] = misses
            result[1] = false
            result[2] = fn
            result[3] = hits
            result[4] = hit_rate
            result[5] = false_alarm_rate
            result[6] = success_rate / (number_actions + EPS)
            result[7] = failure_rate / (number_actions + EPS)
            result[8] = return_period


@guvectorize(
    [
        "(float64[:], int8[:], float64[:], float64[:], int64, int64, float64, int64, int64, float64, float64, float64, int8[:], float64[:])"
    ],
    "(m), (m), (m), (m), (), (), (), (), (), (), (), (), (p) -> (p)",
    nopython=True,
)
def brute_force(
    observations_val,
    observations_bool,
    prob_ready,
    prob_set,
    lead_time,
    issue,
    tolerance,
    filter_constraints,
    min_return_period,
    min_hit_rate,
    min_success_rate,
    max_failure_rate,
    out_shape,
    result,
):
    """
    Evaluate the objective function for a set of thresholds using vectorized operations.

    This function computes the objective function across an array of forecast probabilities
    (`prob_issue0`, `prob_issue1`) and observed values (`obs_val`, `obs_bool`).
    The objective function is evaluated for a set of candidate trigger pairs (t1 and t2 ranging
    between 0 and 1 with a 0.01 step). The trigger pair with the lowest score is extracted as well as the score value.

    The function is optimized for parallel execution with `Numba`'s `guvectorize` decorator to perform
    calculations efficiently across a grid of inputs.

    Args:
        prob_issue0: np.ndarray, Forecast probabilities for the 'ready' stage.
        prob_issue1: np.ndarray, Forecast probabilities for the 'set' stage.
        obs_val: np.ndarray, Array of observed continuous anomaly values.
        obs_bool: np.ndarray, Array of observed binary event occurrences (0 or 1).
        leadtime: int, Indicator period lead time (month).
        issue: int, Forecast issue month.
        tolerance: float, Tolerance threshold for acceptable false alarms.
        filter_constraints: int, Flag to apply operational constraints (1 for yes, 0 for no).
        min_return_period: int, Minimum acceptable return period (in years).
        min_hit_rate: float, Minimum acceptable hit rate.
        min_success_rate: float, Minimum acceptable success rate.
        max_failure_rate: float, Maximum acceptable failure rate.
        out_shape: np.ndarray, Array with the same size as result used as a trick to define the dimension of result in the decorator.
        result: np.ndarray, Array to store computed objective function value.

    Returns:
        None: The `result` array is updated in place with the objective value.

    Notes:
        - This function uses `Numba`'s `guvectorize` to enable fast parallel processing.
        - It computes various performance metrics (hit rate, false alarm rate, success rate, failure rate) based on the thresholds.
        - Constraints (if enabled) can penalize thresholds that don't satisfy operational limits.
    """
    xmin, ymin = 0, 0
    min_value = 1e6

    t = np.empty(2, dtype=np.float64)
    out = np.full(9, np.nan, dtype=np.float64)
    penalty_array = np.full(9, 1e6, dtype=np.float64)
    conf_matrix = np.zeros(4, dtype=np.int8)
    constraints = np.zeros(5, dtype=np.int8)

    # Iterate over the index pairs directly
    for i in range(100):
        for j in range(100):
            t[0], t[1] = i / 100, j / 100

            objective(
                t,
                observations_val,
                observations_bool,
                prob_ready,
                prob_set,
                lead_time,
                issue,
                tolerance,
                filter_constraints,
                min_return_period,
                min_hit_rate,
                min_success_rate,
                max_failure_rate,
                penalty_array,
                conf_matrix,
                constraints,
                out,
            )
            if out[0] < min_value:
                xmin, ymin = t[0], t[1]
                min_value = out[0]

    result[0] = xmin
    result[1] = ymin
    result[2] = min_value


def find_optimal_triggers(
    observations_val,
    observations_bool,
    prob_ready,
    prob_set,
    lead_time,
    issue,
    tolerance,
    filter_constraints,
    min_return_period,
    min_hit_rate,
    min_success_rate,
    max_failure_rate,
    output_shape,
):
    """
    Find the optimal triggers pair by evaluating the objective function on each couple of
    values of a 100 * 100 grid and selecting the minimizer.

    Args:
        observations_val: np.ndarray, Time series of the observed rainfall values (or SPI).
        observations_bool: np.ndarray, Time series of categorical observations for the specified category.
        prob_ready: np.ndarray, Time series of forecast probabilities for the ready month.
        prob_set: np.ndarray, Time series of forecast probabilities for the set month.
        lead_time: int, Lead time month.
        issue: int, Issue month.
        tolerance: float, Tolerance threshold for acceptable false alarms.
        filter_constraints: int, Flag to apply operational constraints (1 for yes, 0 for no).
        min_return_period: int, Minimum acceptable return period (in years).
        min_hit_rate: float, Minimum acceptable hit rate.
        min_success_rate: float, Minimum acceptable success rate.
        max_failure_rate: float, Maximum acceptable failure rate.
        output_shape: np.ndarray, Array with expected output size for numba compilation.

    Returns:
        best_triggers: np.ndarray, Array of size 2 containing best triggers for Ready / Set.
        best_score: float, Score (mainly hit rate) corresponding to the best triggers.
    """

    result = -np.ones(
        3, dtype=np.float64
    )  # fill with -1 to easily detect unexpected values

    brute_force(
        observations_val,
        observations_bool,
        prob_ready,
        prob_set,
        lead_time,
        issue,
        tolerance,
        filter_constraints,
        min_return_period,
        min_hit_rate,
        min_success_rate,
        max_failure_rate,
        output_shape,
        result,
    )

    best_triggers = result[:2]
    best_score = result[2]

    return best_triggers, best_score


def run_ready_set_brute_selection(obs, probs_ready, probs_set, probs, params):
    """
    Run the trigger optimization using xarray's apply_ufunc and Dask parallelization.

    Args:
        obs: xarray.Dataset, dataset containing observed values ('val'), boolean event occurrence ('bool'), lead time ('lead_time'), tolerance ('tolerance'), and return period ('return_period').
        probs_ready: xarray.Dataset, dataset containing the forecast probability used for readiness ('prob').
        probs_set: xarray.Dataset, dataset containing the forecast probability used for activation ('prob').
        probs: xarray.Dataset, dataset containing the forecast issue time ('issue').
        params: object, Params class containing 'vulnerability' and 'requirements' (with 'HR', 'SR', 'FR').

    Returns:
        triggers: xarray.DataArray, array of optimal trigger pairs.
        score: xarray.DataArray, optimal score (- hit rate + alpha * false alarm rate) for each configuration.
    """
    triggers, score = xr.apply_ufunc(
        find_optimal_triggers,
        obs.val,
        obs.bool.astype(np.int8),
        probs_ready.prob,
        probs_set.prob,
        obs.lead_time,
        probs.issue,
        obs.tolerance,
        np.int64(params.vulnerability in ["GT", "NRT"]),
        obs.return_period,
        kwargs={
            "output_shape": np.empty(3, dtype=np.int8),
            "min_hit_rate": np.float64(params.requirements["HR"]),
            "min_success_rate": np.float64(params.requirements["SR"]),
            "max_failure_rate": np.float64(params.requirements["FR"]),
        },
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
        ],
        output_core_dims=[["trigger"], []],
        output_sizes={"trigger": 2},
        output_dtypes=[np.float64, np.float64],
        dask="parallelized",
        keep_attrs=True,
    )

    triggers["trigger"] = ["trigger1", "trigger2"]

    triggers = persist_with_progress_bar(triggers)
    score = persist_with_progress_bar(score)

    return triggers, score


def evaluate_grid_metrics(
    obs,
    probs_ready,
    probs_set,
):
    """
    Evaluate all metrics over the entire grid using apply_ufunc.
    The list of metrics is:
        [Correct Rejections, False Positives, False Negatives, Hits, Hit Rate, False Alarm Rate, Success Rate, Failure Rate, Return Period]

    Args:
        obs: xarray.Dataset, containing numerical and categorical observations, with dimensions (district, time, category, index)
        probs_ready: xarray.Dataset, forecast probabilities for the ready month, with dimensions (district, time, category, index, issue)
        probs_set: xarray.Dataset, forecast probabilities for the set month

    Returns:
        metrics_da: xarray.DataArray, structured array with grid evaluations for all metrics
    """
    # Ensure category alignment between observations and probabilities
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

    # Initialize empty placeholders for results from xr.apply_ufunc
    penalty = xr.DataArray(np.full(9, 1e6, dtype=np.float64), dims=["metric"])
    conf_matrix = xr.DataArray(np.zeros(4, dtype=np.int8), dims=["metric"])
    constraints = xr.DataArray(np.empty(5, dtype=np.int8), dims=["metric"])

    # Apply the `objective` function across the threshold grid
    metrics_da = xr.apply_ufunc(
        objective,
        grid_da,
        obs.val,
        obs.bool.astype(np.int8),
        probs_ready.prob,
        probs_set.prob,
        obs.lead_time,
        probs_ready.issue,
        obs.tolerance,
        np.int64(0),  # set filter_constraints to 0 to prevent trigger selection
        np.int64(-1),  # min_return_period (inactive)
        np.float64(-1),  # min_hit_rate (inactive)
        np.float64(-1),  # min_success_rate (inactive)
        np.float64(10),  # max_failure_rate (allow high)
        penalty,
        conf_matrix,
        constraints,
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
            ["metric"],
            ["metric"],
            ["metric"],
        ],
        output_core_dims=[["metric"]],
        output_sizes={"metric": 9},
        output_dtypes=[np.float64],
        vectorize=True,
        join="outer",
        exclude_dims={"metric"},
    )

    # Assign metric names
    metrics_da = metrics_da.assign_coords(
        metric=["TN", "FP", "FN", "TP", "HR", "FAR", "SR", "FR", "RP"]
    )

    return metrics_da


def save_metrics_df(grid_metrics_da, data_path):
    """
    Convert grid metrics from a DataArray to a pivoted DataFrame and save it as a CSV file.

    Args:
        grid_metrics_da: xarray.DataArray, containing the computed trigger metrics.
        data_path: str, output folder path.

    Returns:
        output_path: str, output file path.
    """
    grid_metrics_df = grid_metrics_da.to_dataframe(name="value").reset_index()
    grid_metrics_df = grid_metrics_df.loc[grid_metrics_df.value < 1e5]

    grid_metrics_df["issue_set"] = grid_metrics_df.issue + 1
    grid_metrics_df = grid_metrics_df.rename(columns={"issue": "issue_ready"})

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

    output_path = (
        f"{data_path}/triggers_metrics_tbd_{grid_metrics_df.district.unique()[0]}.csv"
    )
    grid_metrics_df.to_csv(output_path, index=False)
    return output_path


def get_trigger_metrics_dataframe(obs, probs_ready, probs_set, data_path):
    """
    Compute trigger metrics for a single district and save the results as a CSV file.

    Args:
        obs: xarray.DataArray, observations dataset containing 'bool', 'val', 'lead_time', and 'category' variables.
        probs_ready: xarray.DataArray, dataset containing readiness probabilities with 'prob' and 'issue' variables.
        probs_set: xarray.DataArray, dataset containing set probabilities with 'prob' variable.
        data_path: str, output folder path to save the CSV file.

    Returns:
        None
    """
    grid_metrics_da = evaluate_grid_metrics(obs, probs_ready, probs_set)

    output_path = save_metrics_df(grid_metrics_da, data_path)

    logging.info(f"Metrics saved to {output_path}")


def run_pilot_districts_metrics(obs, probs_ready, probs_set, params):
    """
    Loop through pilot districts and save trigger metrics in CSV.

    Args:
        obs: xarray.DataArray, observational dataset containing 'bool', 'val', 'lead_time', and 'category' variables.
        probs_ready: xarray.DataArray, dataset containing readiness probabilities with 'prob' and 'issue' variables.
        probs_set: xarray.DataArray, dataset containing set probabilities with 'prob' variable.
        params: object, configuration parameters including 'data_path', 'iso', and 'districts_vulnerability'.

    Returns:
        None
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


def evaluate_top_pairs(sub_df, obs, probs_ready, probs_set, params):

    def _select_row(da, district, index, category=None, issue=None):
        sel = da.sel(district=district, index=index)
        if "issue" in da.dims and issue is not None:
            sel = sel.sel(issue=issue)
        if "category" in da.dims and category is not None:
            sel = sel.sel(category=category)
        return sel

    out = np.empty(9, dtype=np.float64)
    penalty_array = np.full(9, 1e6, dtype=np.float64)
    conf_matrix = np.zeros(4, dtype=np.int8)
    constraints = np.zeros(5, dtype=np.int8)

    for (ind, iss), sub_tdf in sub_df.groupby(["index", "issue"]):
        t = sub_tdf.sort_values("trigger").trigger_value.values

        district = sub_tdf.district.values[0]
        category = sub_tdf.category.values[0]
        issue = sub_tdf.issue.values[0]

        stats = objective(
            t,
            _select_row(obs.val, district, ind, category).values,
            _select_row(obs.bool, district, ind, category).values.astype(np.int8),
            _select_row(probs_ready.prob, district, ind, category, issue).values,
            _select_row(probs_set.prob, district, ind, category, issue).values,
            _select_row(obs, district, ind, category).lead_time.values,
            _select_row(probs_ready.prob, district, ind, category, issue).issue.values,
            _select_row(obs, district, ind, category).tolerance.values,
            0,  # we don't need to check the requirements here
            _select_row(obs, district, ind, category).return_period.values,
            np.float64(params.requirements["HR"]),
            np.float64(params.requirements["SR"]),
            np.float64(params.requirements["FR"]),
            penalty_array,
            conf_matrix,
            constraints,
            out,
        )

        hr, fr = stats[4], stats[7]  # HR and FR at indices 4 and 7 of the metrics list
        sub_df.loc[(sub_df["index"] == ind) & (sub_df.issue == iss), "HR"] = hr
        sub_df.loc[(sub_df["index"] == ind) & (sub_df.issue == iss), "FR"] = fr

    return sub_df


def filter_triggers_by_window(df_leadtime, probs_ready, probs_set, obs, params):
    """
    Filters and selects the best trigger pairs for each window by evaluating the trigger values.

    Args:
        df_leadtime: pd.DataFrame, DataFrame containing lead time information and trigger values.
        probs_ready: xarray.Dataset, dataset containing readiness probabilities.
        probs_set: xarray.Dataset, dataset containing set probabilities.
        obs: xarray.DataArray, dataset containing observation values.
        params: object, configuration parameters including requirements for HR, SR, and FR.

    Returns:
        pd.DataFrame, DataFrame containing the best trigger pairs for each window.
    """

    triggers_window_list = [
        evaluate_top_pairs(r, obs, probs_ready, probs_set, params)
        for _, r in df_leadtime.groupby(["district", "category", "Window"])
    ]

    return pd.concat(triggers_window_list)


def get_window_district(area, indicator, district, params):
    """
    Determines which window (Window 1 or Window 2) an indicator belongs to for a given district.
    """
    # Get the dataset containing the area information
    gdf = area.get_dataset([area.BASE_AREA_DATASET])

    # Retrieve the administrative boundaries for level 1 (province) based on the ISO country code
    admin1 = area.get_admin_boundaries(iso3=params.iso, admin_level=1).drop(
        ["geometry", "adm0_Code"], axis=1
    )
    admin1.columns = ["Code_1", "adm1_name"]
    shp = pd.merge(gdf, admin1, how="left", left_on=["adm1_Code"], right_on=["Code_1"])

    # Find the province corresponding to the district name
    province = shp.loc[shp.Name == district].adm1_name.unique()[0]

    # Get the window definitions (window1 and window2) from the params
    window1 = params.get_windows("window1")
    window2 = params.get_windows("window2")

    # If window definitions are dictionaries, check if the indicator is in the province's windows
    if isinstance(window1, dict):
        if indicator in window1[province]:
            return "Window 1"
        elif indicator in window2[province]:
            return "Window 2"
        else:
            return np.nan
    else:
        # If window definitions are lists, check for the indicator in both windows
        if indicator in window1:
            return "Window 1"
        elif indicator in window2:
            return "Window 2"
        else:
            return np.nan
