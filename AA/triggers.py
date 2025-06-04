import logging
import warnings

import click
import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from hip.analysis.analyses.drought import concat_obs_levels, get_accumulation_periods
from hip.analysis.aoi.analysis_area import AnalysisArea
from hip.analysis.compute.utils import persist_with_progress_bar, start_dask

from AA._triggers_ready_set import (
    filter_triggers_by_window,
    get_window_district,
    run_pilot_districts_metrics,
    run_ready_set_brute_selection,
)
from AA.helper_fns import (
    create_flexible_dataarray,
    format_triggers_df_for_dashboard,
    merge_un_biased_probs,
    triggers_da_to_df,
)
from config.params import Params

logging.basicConfig(level="INFO", force=True)
warnings.simplefilter(action="ignore")


@click.command()
@click.argument("country", required=True, type=str)
@click.argument("index", default="SPI")
@click.argument("vulnerability", default="TBD")
@click.option(
    "--data-path",
    required=True,
    type=str,
    help="Root directory for data files.",
)
@click.option(
    "--output-path",
    required=False,
    type=str,
    default=None,
    help="Root directory for output files. Defaults to data-path if not provided.",
)
def run(country, index, vulnerability, data_path, output_path):
    client = start_dask(n_workers=1)
    logging.info("+++++++++++++")
    logging.info(f"Dask dashboard: {client.dashboard_link}")
    logging.info("+++++++++++++")

    params = Params(
        iso=country,
        index=index,
        vulnerability=vulnerability,
        data_path=data_path,
        output_path=output_path,
    )

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


def read_aggregated_obs(path_to_zarr, params):
    fs, _, _ = fsspec.get_fs_token_paths(f"{path_to_zarr}/{params.index} *")
    list_index_paths = fs.glob(f"{path_to_zarr}/{params.index} *")
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
    obs_bool = concat_obs_levels(obs_val, levels=params.intensity_thresholds)

    obs = xr.Dataset({"bool": obs_bool, "val": obs_val})

    # Reformat time and index coords
    obs["time"] = [pd.to_datetime(t).year for t in obs.time.values]
    obs["index"] = [val_path.split(fs.sep)[-1] for val_path in list_val_paths]
    return obs


def read_aggregated_probs(path_to_zarr, params):
    fs, _, _ = fsspec.get_fs_token_paths(f"{path_to_zarr}/*")
    list_issue_paths = sorted(fs.glob(f"{path_to_zarr}/*"))[
        :-1
    ]  # Last one is the `obs` folder.
    list_index = {}

    for iss_path in list_issue_paths:
        list_index_paths = fs.glob(f"{iss_path}/{params.index} *")
        list_index_raw = [
            fs.sep.join([i, "probabilities.zarr"]) for i in sorted(list_index_paths)
        ]
        list_index_bc = [
            fs.sep.join([i, "probabilities_bc.zarr"]) for i in sorted(list_index_paths)
        ]
        index_names = [i.split(fs.sep)[-1] for i in sorted(list_index_paths)]

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


if __name__ == "__main__":
    # From AA repository:
    # $ pixi run python -m AA.triggers MOZ SPI TBD --data-path "C:/path/to/data"

    run()
