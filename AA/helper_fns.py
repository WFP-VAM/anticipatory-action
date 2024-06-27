import glob
import os

import datetime
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

PORTUGUESE_CATEGORIES = dict(
    Normal="Normal", Mild="Leve", Moderate="Moderado", Severe="Severo"
)


def create_flexible_dataarray(start_season, end_season):
    # Create the start and end dates
    start_date = datetime.datetime(1990, start_season, 1)
    end_date = datetime.datetime(1991, end_season+1, 28)

    # Generate the date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')

    # Create the DataArray
    data_array = xr.DataArray(
        np.arange(1, len(date_range) + 1),  # Create a range of values for demonstration
        coords=dict(time=(["time"], date_range)),
        dims="time"
    )
    
    return data_array


def triggers_da_to_df(triggers_da, hr_da):
    triggers_df = triggers_da.to_dataframe().drop(["spatial_ref"], axis=1).dropna()
    triggers_df = triggers_df.reset_index().set_index(
        ["index", "category", "district", "issue"]
    )
    triggers_df.columns = ["trigger", "trigger_value", "lead_time"]
    hr_df = (
        hr_da.to_dataframe()
        .reset_index()
        .drop("lead_time", axis=1)
        .set_index(["district", "category", "issue", "index"])
    )
    triggers_df = triggers_df.join(hr_df)
    triggers_df = triggers_df.drop(["spatial_ref"], axis=1)
    triggers_df.columns = ["trigger", "lead_time", "trigger_value", "HR"]
    return triggers_df.reset_index().drop_duplicates()


def aggregate_by_district(ds, gdf, params):
    PROJ = "+proj=longlat +ellps=clrk66 +towgs84=-80,-100,-228,0,0,0,0 +no_defs"

    # Clip ds to districts
    list_districts = {}
    for _, row in gdf.iterrows():
        try:
            list_districts[row["Name"]] = (
                ds.rio.write_crs(PROJ)
                .rio.clip(gpd.GeoSeries(row.geometry))
                .mean(dim=["latitude", "longitude"])
            )
        except:
            continue

    ds_by_district = xr.concat(
        list_districts.values(), pd.Index(list_districts.keys(), name="district")
    )

    return ds_by_district


def merge_un_biased_probs(probs_district, probs_bc_district, params, period_name):
    # Get fbf_districts data in xarray format
    fbf_bc = params.fbf_districts_df
    fbf_bc = fbf_bc.loc[fbf_bc["Index"] == f"{params.index.upper()} {period_name}"]
    fbf_bc = fbf_bc[["district", "category", "issue", "BC"]]
    fbf_bc_da = fbf_bc.set_index(["district", "category", "issue"]).to_xarray().BC
    fbf_bc_da = fbf_bc_da.expand_dims(dim={"index": [f"{params.index} {period_name}"]})

    # Combination of both probabilities datasets
    probs_merged = (1 - fbf_bc_da) * probs_district + fbf_bc_da * probs_bc_district

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
    # triggers["season"] = f"{params.monitoring_year}-{str(params.monitoring_year+1)[-2:]}"
    # triggers['date'] = [params.monitoring_year if r.issue >= 5 else params.monitoring_year+1 for _, r in triggers.iterrows()]
    # triggers['date'] = [pd.to_datetime(f"{r.issue}-1-{r.date}") for _, r in triggers.iterrows()]

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
    for d, r in cov.iterrows():
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


def merge_probabilities_triggers_dashboard(probs, triggers, params, period):
    # Format probabilities
    probs_df = probs.to_dataframe().reset_index().drop("spatial_ref", axis=1)
    probs_df["prob"] = [np.round(p, 2) for p in probs_df.prob.values]
    probs_df["index"] = probs_df["index"].str.upper()
    probs_df["aggregation"] = np.repeat(
        f"{params.index.upper()} {len(period)}", len(probs_df)
    )

    triggers_merged = triggers.copy()

    # Create prob columns if reading empty triggers df
    if "prob_ready" not in triggers_merged.columns:
        triggers_merged["prob_ready"] = np.nan
        triggers_merged["prob_set"] = np.nan

    # Fill in probabilities columns matching with triggers
    for l, row in triggers_merged.iterrows():
        if (row.issue_ready == params.issue) and (
            row["index"] == f"{params.index.upper()} {period}"
        ):
            triggers_merged.loc[l, "prob_ready"] = probs_df.loc[
                (probs_df["index"] == row["index"])
                & (probs_df["category"] == row.category)
                & (probs_df["district"] == row.district)
            ].prob.values[0]
        elif (row.issue_set == params.issue) and (
            row["index"] == f"{params.index.upper()} {period}"
        ):
            triggers_merged.loc[l, "prob_set"] = probs_df.loc[
                (probs_df["index"] == row["index"])
                & (probs_df["category"] == row.category)
                & (probs_df["district"] == row.district)
            ].prob.values[0]

    return probs_df, triggers_merged


def read_fbf_districts(path_fbf, params):
    fbf_districts = pd.read_csv(path_fbf, sep=",")
    if params.issue:
        fbf_districts = fbf_districts.loc[fbf_districts.issue == params.issue]
    return fbf_districts


def read_forecasts(area, issue, local_path, update=False):
    if os.path.exists(local_path) and not update:
        forecasts = xr.open_zarr(local_path).tp.persist()
        forecasts = forecasts.sel(
            time=slice(
                None,
                datetime.datetime.strptime(
                    area.datetime_range.split("/")[1], "%Y-%m-%d"
                ),
            )
        )
    else:
        forecasts = area.get_dataset(
            ["ECMWF", f"RFH_FORECASTS_SEAS5_ISSUE{int(issue)}_DAILY"],
            load_config={
                "gridded_load_kwargs": {
                    "resampling": "bilinear",
                }
            },
        ).persist()
        forecasts.attrs["nodata"] = np.nan
        forecasts.chunk(dict(time=-1)).to_zarr(local_path, mode="w", consolidated=True)
    return forecasts


def read_observations(area, local_path):
    if os.path.exists(local_path):
        observations = xr.open_zarr(local_path).band.persist()
    else:
        observations = area.get_dataset(
            ["CHIRPS", "RFH_DAILY"],
            load_config={
                "gridded_load_kwargs": {
                    "resampling": "bilinear",
                }
            },
        ).persist()
        observations.to_zarr(local_path)
    return observations


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
