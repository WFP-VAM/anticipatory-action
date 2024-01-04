import glob
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd


PORTUGUESE_CATEGORIES = dict(
    Normal="Normal", Mild="Leve", Moderate="Moderado", Severe="Severo"
)


def aggregate_spi_dryspell_triggers(spi_window, dry_window):
    all_window = pd.concat([spi_window, dry_window])
    agg_window = pd.concat(
        [
            sub_df.sort_values("lead_time").sort_values("FR").sort_values("HR").head(4)
            for _, sub_df in all_window.groupby(["district", "category", "Window"])
        ]
    )
    return agg_window


def triggers_da_to_df(triggers_da, hr_da):
    triggers_df = (
        triggers_da.to_dataframe()
        .drop(["spatial_ref", "vulnerability"], axis=1)
        .dropna()
    )
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
    triggers_df.columns = ["trigger", "trigger_value", "lead_time", "HR", "type"]
    return triggers_df.reset_index().drop_duplicates()


def aggregate_by_district(ds, gdf, params):
    # As the shapefiles are not harmonized
    if params.iso == "MOZ":
        adm2_coord = "ADM2_PT"
    else:
        adm2_coord = "adm2_name"

    # Keep only provided districts
    shp = gdf.loc[gdf[adm2_coord].isin(params.districts )]

    PROJ = "+proj=longlat +ellps=clrk66 +towgs84=-80,-100,-228,0,0,0,0 +no_defs"

    # TODO add downscaling part

    # Clip ds to districts
    list_districts = [
        ds.rio.write_crs(PROJ)
        .rio.clip(gpd.GeoSeries(geo))
        .mean(dim=["latitude", "longitude"])
        for geo in shp.geometry
    ]
    ds_by_district = xr.concat(
        list_districts, pd.Index(shp[adm2_coord].values, name="district")
    )

    return ds_by_district


def merge_un_biased_probs(probs_district, probs_bc_district, params, period_name):
    # Get fbf_districts data in xarray format
    fbf_bc = params.fbf_districts_df
    fbf_bc = fbf_bc.loc[fbf_bc["Index"] == f"{params.index.upper()} {period_name}"]
    fbf_bc = fbf_bc[["district", "category", "issue", "BC"]]
    fbf_bc_da = fbf_bc.set_index(["district", "category", "issue"]).to_xarray().BC
    fbf_bc_da = fbf_bc_da.expand_dims(
        dim={"index": [f"{params.index.upper()} {period_name}"]}
    )
    
    # Combination of both probabilities datasets
    probs_merged = (
        1 - fbf_bc_da
    ) * probs_district.raw + fbf_bc_da * probs_bc_district.bc

    probs_merged = probs_merged.to_dataset(name="prob")

    return probs_merged


def merge_probabilities_triggers_dashboard(probs, triggers, params, period):
    # Format probabilities
    probs_df = probs.to_dataframe().reset_index().drop("spatial_ref", axis=1)
    probs_df["year"] = [str(params.year) for _ in probs_df.iterrows()]
    probs_df["prob"] = [np.round(p, 2) for p in probs_df.prob.values]
    probs_df["aggregation"] = np.repeat(
        f"{params.index.upper()} {len(period)}", len(probs_df)
    )

    # Filter triggers df to index
    triggers_index = triggers.copy()
    triggers_index.loc[triggers_index.trigger == 'trigger2', 'issue'] = triggers_index.loc[triggers_index.trigger == 'trigger2'].issue.values + 1
    triggers_index = triggers_index.loc[
        (triggers_index['index'] == f"{params.index.upper()} {period}")
        & (triggers_index['issue'] == params.issue)
    ]
    # Merge both dataframes
    df_merged = (
        triggers_index.set_index(["district", "index", "category", "issue"])
        .join(probs_df.set_index(["district", "index", "category", "issue"]))
        .reset_index()
    )

    df_merged = df_merged.drop("aggregation", axis=1)

    df_merged["type"] = [i.split(" ")[0] for i in df_merged['index'].values]
    df_merged["year"] = [
        f"{y}-{str(params.year+1)[-2:]}" for y in df_merged.year.values
    ]

    if params.iso in ["MOZ"]:
        WINDOWS_PORTUGUESE = {
            "Window1": "Janela 1",
            "Window2": "Janela 2",
            "Window3": "Janela 3",
        }
        df_merged["Window"] = [WINDOWS_PORTUGUESE[w] for w in df_merged.Window.values]

        df_merged["trigger_type"] = [
            "Acionadores de Crise"
            if d in ["Chibuto", "Guija"]
            else "Acionadores Gerais"
            for d in df_merged.district.values
        ]

    return probs_df, df_merged


def read_fbf_districts(path_fbf, params):
    fbf_districts = pd.read_csv(path_fbf, sep=",")
    if type(params.issue) != list:
        fbf_districts = fbf_districts.loc[fbf_districts.issue == params.issue]
    return fbf_districts


# Temporary local reading function before ingestion of ECMWF data
def read_forecasts_locally(rfh_path):
    files = glob.glob(rfh_path)
    list_years = []
    for f in files:
        rfh_year = xr.open_dataset(f, engine="netcdf4")
        list_years.append(rfh_year)
    rfh_all = xr.concat(list_years, dim="time")
    return rfh_all


# Temporary local reading function before ingestion of HDC data
def read_observations_locally(rfh_path):
    ds1 = xr.open_dataset(
        f"{rfh_path}/RemapMoz-Tot.days_p25_15072021.nc", engine="netcdf4"
    )
    ds2 = xr.open_dataset(
        f"{rfh_path}/RemapMoz-Tot.1990-Mar2022_days_p25.nc", engine="netcdf4"
    )
    ds2 = ds2.where(ds2.time > ds1.time.values[-1], drop=True)
    ds2022 = xr.open_dataset(
        f"{rfh_path}/RemapMoz-chirps-v2.0.2022.days_p25.nc", engine="netcdf4"
    )
    ds2022 = ds2022.where(ds2022.time > ds2.time.values[-1], drop=True)
    ds2023 = xr.open_dataset(
        f"{rfh_path}/RemapMoz-chirps-v2.0.2023.days_p25.nc", engine="netcdf4"
    )
    rfh_all = xr.concat([ds1, ds2, ds2022, ds2023], dim="time")
    return rfh_all


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
        df_ref.columns = ["longitude", "latitude", "ensemble", "spi_ref", "period", "year"]
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
