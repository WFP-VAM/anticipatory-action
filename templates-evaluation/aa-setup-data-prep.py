# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: 'Python (pixi: aa-env)'
#     language: python
#     name: aa-env
# ---

# %% [markdown]
# ### Imports

# %%
import duckdb
import fsspec
import glob
import s3fs
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from IPython.display import Markdown as md
from hip.analysis.analyses.drought import concat_obs_levels


# %% [markdown]
# ###  Reading functions


# %%
def read_aggregated_probs(path_to_zarr, index):
    fs, _, _ = fsspec.get_fs_token_paths(path_to_zarr)
    fs.invalidate_cache()
    list_issue_paths = sorted(fs.glob(f"{path_to_zarr}/*"))[
        :-1
    ]  # Last one is the `obs` folder.
    list_index = {}

    for iss_path in list_issue_paths:
        list_index_paths = fs.glob(f"{iss_path}/{index}_*")
        if list_index_paths == []:
            continue
        list_index_raw = [
            fs.sep.join([i, "probabilities.zarr"]) for i in sorted(list_index_paths)
        ]
        list_index_bc = [
            fs.sep.join([i, "probabilities_bc.zarr"]) for i in sorted(list_index_paths)
        ]
        index_names = [i.split(fs.sep)[-1] for i in sorted(list_index_paths)]

        # Restore full S3 paths if needed
        if isinstance(fs, s3fs.core.S3FileSystem):
            list_index_raw = [
                f"s3://{fs._strip_protocol(path)}" for path in list_index_raw
            ]
            list_index_bc = [
                f"s3://{fs._strip_protocol(path)}" for path in list_index_bc
            ]

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


# %%
def read_aggregated_obs(path_to_zarr, index, intensity_thresholds):
    fs, _, _ = fsspec.get_fs_token_paths(path_to_zarr)
    fs.invalidate_cache()
    list_index_paths = fs.glob(f"{path_to_zarr}/{index}_*")

    # Restore full S3 paths if needed
    if isinstance(fs, s3fs.core.S3FileSystem):
        list_index_paths = [
            f"s3://{fs._strip_protocol(path)}" for path in list_index_paths
        ]

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
    obs_bool = concat_obs_levels(obs_val, levels=intensity_thresholds)

    obs = xr.Dataset({"bool": obs_bool, "val": obs_val})

    # Reformat time and index coords
    obs["time"] = [pd.to_datetime(t).year for t in obs.time.values]
    obs["index"] = [val_path.split(fs.sep)[-1] for val_path in list_index_paths]
    return obs


# %% [markdown]
# ###  Define country and data path

# %%
COUNTRY = "MWI"

# Ideally we would move the aa folder that's in my bucket to a dedicated bucket like for LIA
DATA_PATH = f"s3://dev-hip-anticipatory-action/prod/{COUNTRY.lower()}"

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Read indicator analysis data

# %%
probs_spi = read_aggregated_probs(f"{DATA_PATH}/zarr", "spi")
probs_dry = read_aggregated_probs(f"{DATA_PATH}/zarr", "dryspell")

probs = xr.concat([probs_spi, probs_dry], "index")

# %%
probs_spi = read_aggregated_probs(f"{DATA_PATH}/zarr", "spi")

probs = xr.concat([probs_spi], "index")

# %%
intensity_thresholds = {"Normal": -0.44, "Mild": -0.68, "Moderate": -0.85, "Severe": -1}

chirps_spi = read_aggregated_obs(f"{DATA_PATH}/zarr/obs", "spi", intensity_thresholds)
chirps_dry = read_aggregated_obs(
    f"{DATA_PATH}/zarr/obs", "dryspell", intensity_thresholds
)

chirps_anomaly = xr.concat([chirps_spi, chirps_dry], "index")

# %%
intensity_thresholds = {"Normal": -0.44, "Mild": -0.68, "Moderate": -0.85, "Severe": -1}

chirps_spi = read_aggregated_obs(f"{DATA_PATH}/zarr/obs", "spi", intensity_thresholds)

chirps_anomaly = xr.concat([chirps_spi], "index")

# %%
roc = pd.concat(
    [
        pd.read_csv(f"{DATA_PATH}/auc/roc.spi.csv"),
        pd.read_csv(f"{DATA_PATH}/auc/roc.dryspell.csv"),
    ]
)

# %%
roc = pd.concat(
    [
        pd.read_csv(f"{DATA_PATH}/auc/roc.spi.csv"),
    ]
)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ###  Read triggers data

# %% jupyter={"source_hidden": true}
MOZ_DISTRICTS = [
    "Cahora_Bassa",
    "Caia",
    "Changara",
    "Chemba",
    "Chibabava",
    "Chibuto",
    "Chicualacuala",
    "Chigubo",
    "Chiure",
    "Chiuta",
    "Cidade_Da_Beira",
    "Doa",
    "Funhalouro",
    "Govuro",
    "Guija",
    "Guro",
    "Homoine",
    "Jangamo",
    "Mabalane",
    "Mabote",
    "Machanga",
    "Machaze",
    "Macossa",
    "Magoe",
    "Magude",
    "Mapai",
    "Marara",
    "Massangena",
    "Massinga",
    "Massingir",
    "Moamba",
    "Muanza",
    "Mutarara",
    "Namuno",
    "Panda",
    "Tambara",
]

# %%
MWI_DISTRICTS = roc.district.unique()

# %%
DISTRICTS = MOZ_DISTRICTS if COUNTRY == "MOZ" else MWI_DISTRICTS

# %% jupyter={"outputs_hidden": true}
fs, _, _ = fsspec.get_fs_token_paths(f"{DATA_PATH}/triggers/triggers_metrics_spi")
fs.invalidate_cache()

spi_triggers_paths = [
    f"{DATA_PATH}/triggers/triggers_metrics_spi/triggers_metrics_tbd_{d}.csv"
    for d in MOZ_DISTRICTS
]
dry_triggers_paths = [
    f"{DATA_PATH}/triggers/triggers_metrics_dryspell/triggers_metrics_tbd_{d}.csv"
    for d in MOZ_DISTRICTS
]

triggers = pd.concat(
    [
        *[pd.read_csv(f"s3://{f}") for f in tqdm(spi_triggers_paths)],
        *[pd.read_csv(f"s3://{f}") for f in tqdm(dry_triggers_paths)],
    ]
)

# %%
fs, _, _ = fsspec.get_fs_token_paths(f"{DATA_PATH}/triggers/triggers_metrics")
fs.invalidate_cache()

spi_triggers_paths = [
    f"{DATA_PATH}/triggers/triggers_metrics/triggers_metrics_tbd_{d}.csv"
    for d in DISTRICTS
]

triggers = pd.concat(
    [
        *[pd.read_csv(f"s3://{f}") for f in tqdm(spi_triggers_paths)],
    ]
)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Format to dataframe and save to parquet

# %%
probs_df = probs.to_dataframe().dropna().reset_index()
chirps_df = chirps_anomaly.to_dataframe().dropna().reset_index()

# %%
probs_df["index"] = probs_df["index"].str.upper()
chirps_df["index"] = chirps_df["index"].str.upper()
triggers["index"] = triggers["index"].str.upper()

# %%
# Rename time column as year column
probs_df = probs_df.rename(columns={"time": "year"})
chirps_df = chirps_df.rename(columns={"time": "year"})

# %%
# Unscale CHIRPS-based SPI
chirps_df["val"] = chirps_df.val / 1000

# %%
# Change types
probs_df["issue"] = probs_df.issue.astype(np.uint8)
probs_df["year"] = probs_df.year.astype(np.uint16)
chirps_df["year"] = chirps_df.year.astype(np.uint16)
chirps_df["bool"] = chirps_df["bool"].astype(bool)
roc["BC"] = roc.BC.astype(bool)
roc["issue"] = roc.issue.astype(np.uint8)

# %%
triggers["issue_ready"] = triggers.issue_ready.astype(np.uint8)
triggers["issue_set"] = triggers.issue_set.astype(np.uint8)
triggers["lead_time"] = triggers.lead_time.astype(np.uint8)
triggers["FN"] = triggers.FN.astype(np.uint8)
triggers["FP"] = triggers.FP.astype(np.uint8)
triggers["FPtol"] = triggers.FPtol.astype(np.uint8)
triggers["TN"] = triggers.TN.astype(np.uint8)
triggers["TP"] = triggers.TP.astype(np.uint8)
triggers["RP"] = triggers.RP.astype(np.uint8)

# %%
probs_df["index"] = probs_df["index"].str.replace("_", " ")
probs_df["district"] = probs_df["district"].str.replace("_", " ")

chirps_df["index"] = chirps_df["index"].str.replace("_", " ")
chirps_df["district"] = chirps_df["district"].str.replace("_", " ")

roc["Index"] = roc["Index"].str.replace("_", " ")
roc["district"] = roc["district"].str.replace("_", " ")

triggers["index"] = triggers["index"].str.replace("_", " ")
triggers["district"] = triggers["district"].str.replace("_", " ")

# %%
outdir = f"s3://dev-hip-anticipatory-action/prod/{COUNTRY.lower()}/setup-tool"

probs_df.to_parquet(f"{outdir}/probs.parquet", index=False)
chirps_df.to_parquet(f"{outdir}/obs.parquet", index=False)
roc.to_parquet(f"{outdir}/roc.parquet", index=False)
triggers.to_parquet(f"{outdir}/triggers.parquet", index=False)

# %% [markdown]
# ### Format and save boundaries

# %%
import tempfile, json
import duckdb, geopandas as gpd
from hip.analysis.data._read import get_admin_shapes

gdf = get_admin_shapes(COUNTRY, admin_level=2)  # or hip.analysis.data._read equivalent

# %%
from shapely.validation import make_valid

gdf["geometry"] = gdf["geometry"].apply(make_valid)

# %%
# Write GeoJSON to temp file so DuckDB can read it via ST_Read
with tempfile.NamedTemporaryFile(suffix=".geojson", mode="w", delete=False) as f:
    f.write(gdf.to_json())
    tmp_path = f.name

adm2_col = "Name"

# %%
conn = duckdb.connect()
conn.sql("INSTALL SPATIAL; LOAD spatial")
conn.sql(f"CREATE TABLE boundaries AS SELECT * FROM ST_Read('{tmp_path}')")

# %%
conn.sql("describe boundaries")

# %%
# Rename the country-specific name column to a canonical ADM2_EN
if adm2_col != "ADM2_EN":
    conn.sql(f"ALTER TABLE boundaries RENAME COLUMN {adm2_col} TO ADM2_EN")

# Cast BIGINT columns to INTEGER
query = conn.sql("""
    SELECT string_agg(
        CASE WHEN type = 'BIGINT'
            THEN 'CAST(' || name || ' AS INTEGER) AS ' || name
        ELSE name END, ', ')
    FROM pragma_table_info('boundaries')
""").fetchone()[0]
conn.sql(f"CREATE OR REPLACE TABLE boundaries AS SELECT {query} FROM boundaries")

# %%
out = f"boundaries.parquet"
conn.sql(f"COPY boundaries TO '{out}' (FORMAT 'parquet')")

s3_path = f"s3://dev-hip-anticipatory-action/prod/{COUNTRY.lower()}/setup-tool/boundaries.parquet"
os.system(f"aws s3 cp {out} {s3_path}")
os.remove(out)
os.remove(tmp_path)

# %%
