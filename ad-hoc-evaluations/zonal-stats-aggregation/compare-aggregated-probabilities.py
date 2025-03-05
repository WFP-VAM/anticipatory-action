# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: hdc
#     language: python
#     name: conda-env-hdc-py
# ---

# ## Compare district aggregations: using rio.clip and using zonal_stats

# %cd ../..

# +
import xarray as xr
from AA.analytical import calculate_forecast_probabilities
from AA.helper_fns import (aggregate_by_district, read_forecasts,
                           read_observations)
from config.params import Params
from hip.analysis.analyses.drought import get_accumulation_periods
from hip.analysis.aoi.analysis_area import AnalysisArea
from hip.analysis.compute.utils import start_dask

# -

client = start_dask(n_workers=1)
client

# #### Get probabilities at pixel level for Mozambique - October

country = "MOZ"
index = "SPI"
issue = "10"

params = Params(iso=country, index=index)

# +
area = AnalysisArea.from_admin_boundaries(
    iso3=params.iso.upper(),
    admin_level=2,
    resolution=0.25,
    datetime_range=f"1981-01-01/{params.calibration_year}-06-30",
)

gdf = area.get_dataset([area.BASE_AREA_DATASET])
# -

# Observations data reading
observations = read_observations(
    area,
    f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/obs/observations.zarr",
)
observations

# Forecasts data reading
forecasts = read_forecasts(
    area,
    issue,
    f"{params.data_path}/data/{params.iso}/zarr/{params.calibration_year}/{issue}/forecasts.zarr",
    update=False,
)
forecasts

# Get accumulation periods (DJ, JF, FM, DJF, JFM...)
accumulation_periods = get_accumulation_periods(
    forecasts,
    params.start_season,
    params.end_season,
    params.min_index_period,
    params.max_index_period,
)

# +
# Run workflow for one issue month
probabilities, probabilities_bc, anomaly_obs = {}, {}, {}

for period_name, period_months in accumulation_periods.items():

    probs, probs_bc, obs_values, _ = calculate_forecast_probabilities(
        forecasts,
        observations,
        params,
        period_months,
        issue,
    )

    probabilities[period_name] = probs
    probabilities_bc[period_name] = probs_bc
    anomaly_obs[period_name] = obs_values


# +
def concat_along_index(da_dict):
    da = xr.concat(da_dict.values(), dim="index")
    return da.assign_coords({"index": ("index", list(da_dict.keys()))})


probabilities = concat_along_index(probabilities)
probabilities_bc = concat_along_index(probabilities_bc)
anomaly_obs = concat_along_index(anomaly_obs)
# -

# ####  Aggregate using **rio.clip**
#
# Running time estimate: 1min

# %%time
probs_district_rio = aggregate_by_district(probabilities, gdf, params)

# %%time
probs_bc_district_rio = aggregate_by_district(probabilities_bc, gdf, params)

# %%time
obs_district_rio = aggregate_by_district(anomaly_obs, gdf, params)

# #### Aggregate using **zonal_stats**
#
# Running time estimate: 12s (5x faster)

districts_to_exclude = [
    "Cidade_De_Pemba",
    "Ibo",
    "Pemba",
    "Cidade_De_Xai-Xai",
    "Cidade_De_Inhambane",
    "Maxixe",
    "Cidade_De_Chimoio",
    "Cidade_Da_Matola",
    "Cidade_De_Nampula",
    "Ilha_De_Moçambique",
    "Cidade_De_Lichinga",
    "Maquival",
]

zone_ids = gdf.loc[~gdf.Name.isin(districts_to_exclude)].Name.values

zone_ids, zones = area._resolve_zones(
    probabilities.isel(category=0, index=0), zone_ids=zone_ids
)

# %%time
stacked = probabilities.rename({"year": "time"}).stack(
    multi_index=["category", "index"]
)
grouped = stacked.groupby("multi_index")
probs_district_zonal = grouped.map(
    lambda da: area.zonal_stats(
        da.squeeze(["multi_index"]), stats=["mean"], zone_ids=zone_ids, zones=None
    ).to_xarray()["mean"]
).rename({"zone": "district"})

# %%time
stacked = probabilities_bc.rename({"year": "time"}).stack(
    multi_index=["category", "index"]
)
grouped = stacked.groupby("multi_index")
probs_bc_district_zonal = grouped.map(
    lambda da: area.zonal_stats(
        da.squeeze(["multi_index"]), stats=["mean"], zone_ids=zone_ids, zones=None
    ).to_xarray()["mean"]
).rename({"zone": "district"})

# %%time
obs_district_zonal = (
    anomaly_obs.groupby("index")
    .map(
        lambda da: area.zonal_stats(
            da.squeeze("index"), stats=["mean"], zone_ids=zone_ids, zones=None
        ).to_xarray()["mean"]
    )
    .rename({"zone": "district"})
)

# %%time
# Unstack dimensions
probs_district_zonal = probs_district_zonal.unstack("multi_index")
probs_bc_district_zonal = probs_bc_district_zonal.unstack("multi_index")
obs_district_zonal = obs_district_zonal.unstack("multi_index")

# #### Compare both methods
#
# Identical results (relative tol: 1e-7, absolute tol: 0)

# Harmonize dimensions / coordinates sorting
probs_district_zonal = (
    probs_district_zonal.sel(
        district=probs_district_rio.district,
        index=probs_district_rio.index,
        category=probs_district_rio.category,
    )
    .rename({"time": "year"})
    .transpose(*probs_district_rio.dims)
)
probs_bc_district_zonal = probs_bc_district_rio.sel(
    district=probs_bc_district_rio.district,
    index=probs_bc_district_rio.index,
    category=probs_bc_district_rio.category,
).rename({"time": "year"})
obs_district_zonal = obs_district_rio.sel(
    district=obs_district_rio.district,
    index=obs_district_rio.index,
)

xr.testing.assert_allclose(probs_district_rio, probs_district_zonal)

xr.testing.assert_allclose(probs_bc_district_rio, probs_bc_district_zonal)

xr.testing.assert_allclose(obs_district_rio, obs_district_zonal)

# #### Check datasets to visualize the indices / categories / districts that we compare

probs_district_rio

probs_bc_district_rio

obs_district_rio
