import datetime
import glob
import logging
import os

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from hip.analysis.compute.utils import persist_with_progress_bar

PORTUGUESE_CATEGORIES = dict(
    Normal="Normal", Mild="Leve", Moderate="Moderado", Severe="Severo"
)


def create_flexible_dataarray(start_season, end_season):
    # Create the start and end dates
    start_date = datetime.datetime(1990, start_season, 1)
    end_date = datetime.datetime(1991, end_season + 1, 28)

    # Generate the date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="M")

    # Create the DataArray
    data_array = xr.DataArray(
        np.arange(1, len(date_range) + 1),  # Create a range of values for demonstration
        coords=dict(time=(["time"], date_range)),
        dims="time",
    )

    return data_array


def triggers_da_to_df(triggers_da, score_da):
    """Converts trigger and score DataArrays to a merged DataFrame.

    This function processes two xarray DataArrays containing trigger values and scores,
    converts them to pandas DataFrames, and merges them based on specified indices.

    Args:
        triggers_da (xarray.DataArray): DataArray containing trigger information.
        score_da (xarray.DataArray): DataArray containing score information.

    Returns:
        pandas.DataFrame: A DataFrame combining trigger values and scores, with duplicates removed.
    """
    # Convert triggers to DataFrame, clean and set index
    triggers_df = (
        triggers_da.rename("trigger_value")
        .to_dataframe()
        .drop(columns=["spatial_ref", "return_period", "tolerance"], errors="ignore")
        .dropna()
        .reset_index()
        .set_index(["index", "category", "district", "issue"])
    )

    # Convert score to DataFrame, clean and set index
    score_df = (
        score_da.rename("HR")
        .to_dataframe()
        .reset_index()
        .drop(columns=["lead_time", "return_period", "tolerance"], errors="ignore")
        .set_index(["district", "category", "issue", "index"])
    )

    # Join triggers with score
    triggers_df = triggers_df.join(score_df)

    # Reset index and remove duplicates
    return triggers_df.reset_index().drop_duplicates()


def compute_district_average(da, area):
    """
    Computes zonal statistics on an xarray DataArray for both observations and probabilities.
    Uses all_touched=False as the primary rasterization strategy.
    For districts absent from all_touched=False (too small to contain any pixel center),
    falls back to all_touched=True. Districts that are lost by all_touched=True but present
    in all_touched=False are always kept from the latter.
    Districts still missing from both (e.g. their only candidate pixel is won by a
    neighboring zone) are rescued by rasterizing that district's geometry alone,
    with no competing zones, over a clipped bbox.
    Args:
        da: xarray DataArray with spatial dimensions (latitude, longitude) and optionally
            a time dimension and one additional groupby dimension.
        area: Area object with zonal_stats() and get_dataset() methods.
    Returns:
        xarray DataArray with a district dimension containing zonal means.
    """
    # Ensure consistent time dimension
    if "year" in da.dims:
        da = da.rename({"year": "time"})
    # Determine dimensions to group by (exclude spatial dimensions)
    groupby_dim = set(da.dims) - {"latitude", "longitude", "time"}
    # Transpose dims to ensure equality of shapes
    da = da.transpose(..., *groupby_dim, "latitude", "longitude")
    if len(groupby_dim) > 1:
        raise NotImplementedError(
            "Zonal stats with more than one groupby dimension are not supported."
        )

    def _zonal_stats(data, *, zone_ids=None, all_touched=False):
        """Helper that returns a clean DataArray with district dimension."""
        out = area.zonal_stats(
            data,
            stats=["mean"],
            zone_ids=zone_ids,
            zones=None,
            all_touched=all_touched,
        )
        return (
            out.query("zone != 'Administrative unit not available'")
            .to_xarray()["mean"]
            .rename({"zone": "district"})
            .assign_coords(district=lambda x: x.district.astype(str))
        )

    expected_districts = set(
        area.get_dataset([area.BASE_AREA_DATASET]).index.astype(str)
    )

    def _rasterize_single_district(data_clip, geom, all_touched=True):
        """Rasterize a single district geometry onto data_clip's grid, with
        value 0 where covered and -1 (nodata) elsewhere — no other district
        competes for any pixel."""
        from rasterio.features import rasterize
        from rasterio.transform import from_origin

        lat = data_clip.latitude.values
        lon = data_clip.longitude.values
        lat_res = abs(lat[1] - lat[0])
        lon_res = abs(lon[1] - lon[0])
        lat_desc = lat[0] > lat[-1]

        west = lon.min() - lon_res / 2
        north = (lat[0] if lat_desc else lat[-1]) + lat_res / 2
        transform = from_origin(west, north, lon_res, lat_res)

        lat_for_burn = lat if lat_desc else lat[::-1]
        burned = rasterize(
            [(geom, 0)],
            out_shape=(len(lat_for_burn), len(lon)),
            transform=transform,
            fill=-1,
            all_touched=all_touched,
            dtype="int32",
        )
        if not lat_desc:
            burned = burned[::-1, :]

        return xr.DataArray(
            burned,
            coords={"latitude": lat, "longitude": lon},
            dims=["latitude", "longitude"],
        )

    def _rescue_missing_districts(data, missing_districts):
        """
        For districts missing from both all_touched=False and all_touched=True
        rasterizations (typically small districts whose only candidate pixel
        was assigned to a neighboring zone), recompute by rasterizing that
        district's geometry alone onto a clipped bbox, so no neighboring
        district can compete for the shared pixel.
        """
        gdf = area.get_dataset([area.BASE_AREA_DATASET])
        results = []
        for district in missing_districts:
            geom = gdf.loc[district].geometry
            minx, miny, maxx, maxy = geom.bounds

            lat_res = abs(data.latitude.values[1] - data.latitude.values[0])
            lon_res = abs(data.longitude.values[1] - data.longitude.values[0])
            pad_lat, pad_lon = lat_res * 1.5, lon_res * 1.5

            lat_desc = data.latitude.values[0] > data.latitude.values[-1]
            lat_slice = (
                slice(maxy + pad_lat, miny - pad_lat)
                if lat_desc
                else slice(miny - pad_lat, maxy + pad_lat)
            )
            lon_slice = slice(minx - pad_lon, maxx + pad_lon)

            data_clip = data.sel(latitude=lat_slice, longitude=lon_slice)
            if data_clip.latitude.size == 0 or data_clip.longitude.size == 0:
                logging.warning(
                    f"District '{district}': clipped bbox has no pixels, skipping rescue."
                )
                continue

            zones_da = _rasterize_single_district(data_clip, geom, all_touched=True)
            if not bool((zones_da == 0).any()):
                logging.warning(
                    f"District '{district}': geometry doesn't touch any pixel "
                    f"even with all_touched=True, skipping rescue."
                )
                continue

            out = area.zonal_stats(
                data_clip,
                stats=["mean"],
                zone_ids=[district],
                zones=zones_da,  # pre-built, single-district — bypasses _resolve_zones
                all_touched=True,
            )
            da_rescued = (
                out.query("zone != 'Administrative unit not available'")
                .to_xarray()["mean"]
                .rename({"zone": "district"})
                .assign_coords(district=lambda x: x.district.astype(str))
            )
            results.append(da_rescued)

        if not results:
            return None
        return xr.concat(results, dim="district")

    def _with_fallback(data):
        da_false = _zonal_stats(data, all_touched=False)
        da_true = _zonal_stats(data, all_touched=True)
        false_districts = set(da_false.district.values)
        true_districts = set(da_true.district.values)

        # Districts lost by all_touched=True — always keep all_touched=False values
        only_in_false = false_districts - true_districts
        if only_in_false:
            logging.warning(
                f"{len(only_in_false)} district(s) lost by all_touched=True, "
                f"keeping all_touched=False values: {only_in_false}"
            )

        # Districts completely absent from both rasterizations
        missing_from_both = expected_districts - false_districts - true_districts
        da_rescued = None
        if missing_from_both:
            logging.warning(
                f"{len(missing_from_both)} district(s) missing from both "
                f"rasterizations: {missing_from_both}"
            )
            # TEMPORARY: skip rescue for Tanzania (adm0_Code 257) — already
            # handled by the dedicated Wete/Micheweni patching. Remove this
            # guard once that patch is folded into (or replaced by) this rescue.
            adm0_code = area.get_dataset([area.BASE_AREA_DATASET]).adm0_Code.unique()[0]
            if adm0_code != 257:
                da_rescued = _rescue_missing_districts(data, missing_from_both)
                if da_rescued is not None:
                    still_missing = missing_from_both - set(da_rescued.district.values)
                    if still_missing:
                        logging.warning(
                            f"{len(still_missing)} district(s) still missing after "
                            f"single-geometry rescue: {still_missing}"
                        )
                else:
                    logging.warning(
                        f"No district(s) could be rescued out of: {missing_from_both}"
                    )
            else:
                logging.info(
                    "Skipping single-geometry rescue for TZA (adm0_Code 257) — "
                    "handled separately by existing Wete/Micheweni patching."
                )

        # Only take from all_touched=True what is strictly absent from all_touched=False
        # (small districts with no pixel centers inside them)
        only_in_true = true_districts - false_districts
        pieces = [da_false]
        if only_in_true:
            pieces.append(da_true.sel(district=list(only_in_true)))
        if da_rescued is not None:
            pieces.append(da_rescued)

        if len(pieces) == 1:
            return pieces[0]
        return xr.concat(pieces, dim="district")

    if len(groupby_dim) == 1:
        gb = list(groupby_dim)[0]
        da_grouped = da.groupby(gb).map(lambda s: _with_fallback(s.squeeze(gb)))
    else:
        da_grouped = _with_fallback(da)
    return da_grouped


def post_process_aggregated_observations(ds, iso):
    """
    Post-process district-aggregated observations.

    Tanzania-specific fix:
    Wete and Micheweni are smaller than a single 0.25° observation pixel.
    Their rasterized zones frequently contain only NaN values, while the
    overlapping valid observation pixel is assigned to Chake Chake during
    rasterization. When Wete/Micheweni are missing and Chake Chake is
    available, use Chake Chake as a proxy.
    """
    if iso == "tza":
        ds = ds.copy()

        chake = ds.sel(district="Chake Chake")
        wete = ds.sel(district="Wete")
        micheweni = ds.sel(district="Micheweni")

        # Sanity check: only fill when both districts are missing and
        # Chake Chake has data.
        fill_mask = wete.isnull() & micheweni.isnull() & chake.notnull()

        ds.loc[dict(district="Wete")] = xr.where(
            fill_mask,
            chake,
            wete,
        )

        ds.loc[dict(district="Micheweni")] = xr.where(
            fill_mask,
            chake,
            micheweni,
        )

    return ds


def merge_un_biased_probs(probs_district, probs_bc_district, params, period_name):
    # Get roc_df data in xarray format
    roc_df = params.roc_df
    roc_df = roc_df.loc[roc_df["Index"] == f"{params.index.upper()} {period_name}"]
    roc_df = roc_df[["district", "category", "issue", "BC"]]

    # If params.roc_df has Portuguese category names, ensure these are English
    CATEGORY_TRANSLATIONS = {"Leve": "Mild", "Moderado": "Moderate", "Severo": "Severe"}
    roc_df["category"] = roc_df["category"].apply(
        lambda x: CATEGORY_TRANSLATIONS.get(x, x)
    )

    roc_da = roc_df.set_index(["district", "category", "issue"]).to_xarray().BC
    roc_da = roc_da.expand_dims(dim={"index": [f"{params.index}_{period_name}"]})

    # Combination of both probabilities datasets
    probs_merged = (1 - roc_da) * probs_district + roc_da * probs_bc_district

    probs_merged = probs_merged.to_dataset(name="prob")

    return probs_merged


def format_triggers_df_for_dashboard(triggers, params):
    triggers["index"] = triggers["index"].str.upper()
    triggers.loc[(triggers.trigger == "trigger2") & (triggers.issue == 12), "issue"] = 0
    triggers.loc[triggers.trigger == "trigger2", "issue"] = (
        triggers.loc[triggers.trigger == "trigger2"].issue.values + 1
    )

    triggers["prob"] = np.nan
    triggers["HR"] = triggers["HR"].abs()

    if "season" not in triggers.columns:
        triggers["season"] = (
            f"{params.monitoring_year}-{str(params.monitoring_year + 1)[-2:]}"
        )
        triggers["date"] = [
            params.monitoring_year if r.issue >= 5 else params.monitoring_year + 1
            for _, r in triggers.iterrows()
        ]
        triggers["date"] = [
            pd.to_datetime(f"{r.issue}-1-{r.date}") for _, r in triggers.iterrows()
        ]

    def substract(issue):
        return 2 if issue == 1 else 1

    triggers["mready"] = [
        r.issue if r.trigger == "trigger1" else (r.issue - substract(int(r.issue))) % 13
        for _, r in triggers.iterrows()
    ]

    triggers_pivot = triggers.pivot_table(
        index=["district", "index", "category", "Window", "mready"],
        columns="trigger",
        values=["trigger_value", "prob", "issue"],
    ).reset_index()
    triggers_pivot.columns = [
        "district",
        "index",
        "category",
        "window",
        "mready",
        "issue_ready",
        "issue_set",
        "trigger_ready",
        "trigger_set",
    ]
    triggers_pivot = triggers_pivot.drop("mready", axis=1)

    return triggers_pivot


def get_coverage(triggers_df, districts: list, columns: list):
    cov = pd.DataFrame(
        columns=columns,
        index=districts,
    )
    for d, _ in cov.iterrows():
        val = []
        for w in triggers_df["window"].unique():
            for c in triggers_df["category"].unique():
                val.append(
                    len(
                        triggers_df[
                            (triggers_df["window"] == w)
                            & (triggers_df["category"] == c)
                            & (triggers_df["district"] == d)
                        ]
                    )
                )
        cov.loc[d] = val

    print(
        f"The coverage is {round(100 * np.sum(cov.values > 0) / np.size(cov.values), 1)} %"
    )
    return cov


def load_trigger_with_reference(params, variant_folder=None):
    """
    Load trigger data and reference trigger data for comparison.

    Args:
        params: An object containing parameters such as data_path, iso, and calibration_year.
        variant_folder (str, optional): If provided, modifies the data path to load from an alternative directory.

    Returns:
        dict: A dictionary containing DataFrames for GT, NRT, and pilots triggers and reference triggers.
    """
    base_path = f"{params.data_path}/data/{variant_folder or params.iso}"

    files = ["GT", "NRT", "pilots"]

    triggers = {}
    for file in files:
        trigger_path = f"{base_path}/triggers/triggers.spi.dryspell.{params.calibration_year}.{file}.csv"
        ref_path = f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.{file}.csv"

        triggers[f"triggers_{file}"] = pd.read_csv(trigger_path)
        triggers[f"reference_{file}"] = pd.read_csv(ref_path)

    return triggers


def merge_probabilities_triggers_dashboard(probs, triggers, params, period):
    # Format probabilities
    probs_df = probs.to_dataframe().reset_index()
    probs_df["prob"] = [np.round(p, 2) for p in probs_df.prob.values]
    probs_df["index"] = probs_df["index"].str.replace("_", " ").str.upper()
    probs_df["aggregation"] = np.repeat(
        f"{params.index.upper()} {len(period)}", len(probs_df)
    )

    triggers_merged = triggers.copy()

    # Create prob columns if reading empty triggers df
    if "prob_ready" not in triggers_merged.columns:
        triggers_merged["prob_ready"] = np.nan
        triggers_merged["prob_set"] = np.nan

    # Fill in probabilities columns matching with triggers
    target_index = f"{params.index.upper()} {period}"

    # Drop all rows that do not related to the target_index
    triggers_merged = triggers_merged[triggers_merged["index"] == target_index]

    for idx, row in triggers_merged.iterrows():
        if row.issue_ready == params.issue:
            match_filter = (
                (probs_df["index"] == target_index)
                & (probs_df["category"] == row.category)
                & (probs_df["district"] == row.district)
            )
            matching_probs = probs_df.loc[match_filter]
            if len(matching_probs) > 0:
                prob_value = matching_probs.prob.values[0]
                triggers_merged.loc[idx, "prob_ready"] = prob_value

        elif row.issue_set == params.issue:
            match_filter = (
                (probs_df["index"] == target_index)
                & (probs_df["category"] == row.category)
                & (probs_df["district"] == row.district)
            )
            matching_probs = probs_df.loc[match_filter]
            if len(matching_probs) > 0:
                prob_value = matching_probs.prob.values[0]
                triggers_merged.loc[idx, "prob_set"] = prob_value

    return probs_df, triggers_merged


def validate_prism_dataframe(df: pd.DataFrame):
    # Expected columns and types
    expected_columns = {
        "district": str,
        "index": str,
        "category": str,
        "window": str,
        "issue_ready": float | int,
        "issue_set": float | int,
        "trigger_ready": float,
        "trigger_set": float,
        "vulnerability": str,
        "prob_ready": float,
        "prob_set": float,
        "season": str,
        "date_ready": str,
        "date_set": str,
    }

    # Check columns
    missing = set(expected_columns) - set(df.columns)
    extra = set(df.columns) - set(expected_columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if extra:
        raise ValueError(f"Unexpected columns: {extra}")

    # Check column types
    for col, expected_type in expected_columns.items():
        if not df[col].map(lambda x: isinstance(x, expected_type)).all():
            raise TypeError(
                f"Column '{col}' has incorrect type. Expected {expected_type.__name__}"
            )

    # Check index column is uppercase
    if not df["index"].map(lambda x: x.isupper()).all():
        raise ValueError("All values in 'index' column must be uppercase")

    # Check category values
    valid_categories = {"Normal", "Mild", "Moderate", "Severe"}
    if not df["category"].isin(valid_categories).all():
        raise ValueError(
            f"'category' column contains invalid values. Allowed: {valid_categories}"
        )

    # Check window values
    valid_windows = {"Window 1", "Window 2"}
    if not df["window"].isin(valid_windows).all():
        raise ValueError(
            f"'window' column contains invalid values. Allowed: {valid_windows}"
        )

    # Check vulnerability values
    valid_vulnerability = {"General Triggers", "Emergency Triggers"}
    if not df["vulnerability"].isin(valid_vulnerability).all():
        raise ValueError(
            f"'vulnerability' column contains invalid values. Allowed: {valid_vulnerability}"
        )

    # Check date formats
    for col in ["date_ready", "date_set"]:
        try:
            parsed_dates = pd.to_datetime(df[col], format="%Y-%m-%d", errors="raise")
        except Exception:
            raise ValueError(f"Column '{col}' must use format YYYY-MM-DD")

        if not all(parsed_dates.dt.day == 1):
            raise ValueError(f"All dates in '{col}' must have day = 01")

    return True


## Get SPI/probabilities of reference produced with R script from Gabriela Nobre for validation ##


def read_spi_references(path_ref, bc: bool = False, obs: bool = False):
    df_ref = pd.DataFrame()
    files_ref_index = glob.glob(f"{path_ref}*.csv")
    list_index_csv = []
    for ind in files_ref_index:
        df_ind = pd.read_csv(ind).melt(id_vars=["V1", "V2"])
        if not (bc) and not (obs):
            df_ind["year"] = [
                np.int16(v.split("_")[-1]) for v in df_ind.variable.values
            ]
            df_ind = df_ind.loc[df_ind.year == 2022]
        if obs:
            df_ind["period"] = np.repeat(ind.split(".")[0].split(" ")[1], len(df_ind))
            offset_year = (sorted(df_ind.variable.values)[0] == "1982") * 1
            df_ind["variable"] = [int(y) - offset_year for y in df_ind.variable.values]
        else:
            df_ind["period"] = np.repeat(ind.split("/")[-1].split(".")[-2], len(df_ind))
            df_ind["year"] = [
                np.int16(e.split("_")[-1]) for e in df_ind.variable.values
            ]
            df_ind["variable"] = [
                np.float64(e.split("_")[-2]) for e in df_ind.variable.values
            ]
            df_ind = df_ind.loc[df_ind.year == 2022]
        df_ind.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_ind = df_ind.dropna()
        list_index_csv.append(df_ind)
    df_ref_spi = pd.concat(list_index_csv)
    df_ref = pd.concat([df_ref, df_ref_spi])
    if bc:
        df_ref.columns = [
            "longitude",
            "latitude",
            "ensemble",
            "spi_ref",
            "period",
            "year",
        ]
    elif obs:
        df_ref.columns = ["longitude", "latitude", "year", "spi_ref", "period"]
    else:
        df_ref.columns = [
            "longitude",
            "latitude",
            "ensemble",
            "spi_ref",
            "year",
            "period",
        ]
    return df_ref


def read_probas_references(path_ref_probas, cats):
    df_ref = pd.DataFrame()
    for cat in cats:
        files_ref_index = glob.glob(f"{path_ref_probas}{cat}/*")
        list_index_csv = []
        for ind in files_ref_index:
            df_ind = pd.read_csv(ind).melt(id_vars=["V1", "V2"])
            df_ind["period"] = np.repeat(
                ind.split("/")[-1].split(".")[-2].split("_")[0], len(df_ind)
            )
            df_ind["category"] = np.repeat(PORTUGUESE_CATEGORIES[cat], len(df_ind))
            df_ind = df_ind.dropna()
            list_index_csv.append(df_ind)
        df_ref_cat = pd.concat(list_index_csv)
        df_ref = pd.concat([df_ref, df_ref_cat])
    df_ref.columns = [
        "longitude",
        "latitude",
        "year",
        "probability_ref",
        "period",
        "category",
    ]
    df_ref = df_ref[df_ref.year == "2022"].drop("year", axis=1)
    return df_ref
