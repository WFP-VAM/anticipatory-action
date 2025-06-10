# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: hdc
#     language: python
#     name: conda-env-hdc-py
# ---

# This notebook aims to calculate the forecast skill (ROC score) over Tanzania for the OND, JFM and MAM periods. It uses HDC datasets (read through hip-analysis). This notebook can be run using the HDC environment.

# +
# %cd ../../
import datetime
import os

import numpy as np
import xarray as xr
import xskillscore as xss
from hdc.colors.rainfall import rxs
from hip.analysis.aoi.analysis_area import AnalysisArea
from odc.geo.xr import write_cog

from AA.analytical import calculate_forecast_probabilities
from config.params import Params

# +
issue = "1"

params = Params(iso="TZA", index="SPI")
# -

DATES = "1981-01-01/2023-12-31"
area = AnalysisArea.from_admin_boundaries(
    iso3="TZA", admin_level=2, resolution=0.25, datetime_range=DATES
)
area.geobox

gdf = area.get_dataset([area.BASE_AREA_DATASET])

forecasts = area.get_dataset(
    ["ECMWF", f"RFH_FORECASTS_SEAS5_ISSUE{int(issue)}_DAILY"],
    load_config={
        "gridded_load_kwargs": {
            "resampling": "bilinear",
        }
    },
)
forecasts = forecasts.where(
    forecasts.time < np.datetime64(f"{params.year}-07-01T12:00:00.000000000"),
    drop=True,
).to_dataset()
forecasts.attrs["nodata"] = np.nan
forecasts.tp.attrs["nodata"] = np.nan

# + jupyter={"outputs_hidden": true}
forecasts.load()
# -

observations = (
    area.get_dataset(
        ["CHIRPS", "RFH_DAILY"],
        load_config={
            "gridded_load_kwargs": {
                "resampling": "bilinear",
            }
        },
    )
    .rename("precip")
    .to_dataset()
)
observations.attrs["nodata"] = observations.precip.nodata

area_lta = AnalysisArea.from_admin_boundaries(
    iso3="TZA", admin_level=2, resolution=0.25, datetime_range="1970-01-01/1970-12-31"
)
obs_lta = area_lta.get_dataset(
    ["CHIRPS", "R3H_DEKAD_LTA"],
    load_config={
        "gridded_load_kwargs": {
            "resampling": "bilinear",
        }
    },
)

observations.load()

# +
period_months = (3, 4, 5)  # MAM

probs, probs_bc, obs_values, obs_bool = calculate_forecast_probabilities(
    forecasts,
    observations,
    params,
    period_months,
    issue,
)


# +
def evaluate_roc_forecasts(observations, *forecasts, dim="year"):
    """
    Calculate the area under the ROC curve scores of forecasts dataset(s)

    Args:
        observations: xarray.DataArray, categorical observations at the pixel level for specific index
        *forecasts: xarray.DataArray object(s) containing forecasts as probabilities to evaluate for specific index and issue month
        dim: str or list, dimension(s) over which to compute the contingency table
    Returns:
        auc: tuple of the same length as *forecasts of xarray.DataArray, reduced by dimensions dim, containing area under ROC curve
    """

    aucs = []

    # Compute AUC for each forecasts xarray object
    for fc in forecasts:
        auc = xss.roc(
            observations.sel(year=fc.year), fc, dim=dim, return_results="area"
        )
        # set AUC as NaN where no rain in either chirps or forecasts (replicate R method)
        auc = auc.where((observations.sum(dim) != 0) & (fc.sum(dim) != 0), np.nan)
        aucs.append(auc)

    return tuple(aucs)


# -

auc, auc_bc = evaluate_roc_forecasts(
    obs_bool.precip,
    probs.tp,
    probs_bc.scen,
)

auc.to_zarr("ad-hoc-evaluations/tanzania/outputs/SPI MAM/auc_mam_issue1.zarr")

auc_bc.to_zarr("ad-hoc-evaluations/tanzania/outputs/SPI MAM/auc_bc_mam_issue1.zarr")

# ####  Visualization

period = "OND"
iss = 8

auc = xr.open_zarr(
    f"ad-hoc-evaluations/tanzania/outputs/SPI {period}/auc_{period.lower()}_issue{iss}.zarr"
).histogram_observations_forecasts
auc_bc = xr.open_zarr(
    f"ad-hoc-evaluations/tanzania/outputs/SPI {period}/auc_bc_{period.lower()}_issue{iss}.zarr"
).histogram_observations_forecasts

# +

for c in auc.category:
    if not os.path.exists(
        f"ad-hoc-evaluations/tanzania/outputs/rasters/SPI {period}/{c.values}"
    ):
        os.mkdir(f"ad-hoc-evaluations/tanzania/outputs/rasters/SPI {period}/{c.values}")

    write_cog(
        auc.sel(category=c.values),
        f"ad-hoc-evaluations/tanzania/outputs/rasters/SPI {period}/{c.values}/auc_{str(c.values).lower()}_{period.lower()}_issue{iss}.tif",
        overwrite=True,
    )
    write_cog(
        auc_bc.sel(category=c),
        f"ad-hoc-evaluations/tanzania/outputs/rasters/SPI {period}/{c.values}/auc_bc_{str(c.values).lower()}_{period.lower()}_issue{iss}.tif",
        overwrite=True,
    )

    figure = auc.sel(category=c.values).hip.viz.map(
        title=f"AUC {c.values} {period}", cmap="RdYlGn"
    )
    figure.savefig(
        f"ad-hoc-evaluations/tanzania/outputs/rasters/SPI {period}/{c.values}/auc_{str(c.values).lower()}_{period.lower()}_issue{iss}.png"
    )

    figure = auc_bc.sel(category=c.values).hip.viz.map(
        title=f"AUC BC {c.values} {period}", cmap="RdYlGn"
    )
    figure.savefig(
        f"ad-hoc-evaluations/tanzania/outputs/rasters/SPI {period}/{c.values}/auc_bc_{str(c.values).lower()}_{period.lower()}_issue{iss}.png"
    )
# -

# Without bias correction
cat = "Moderate"
auc.sel(category=cat).hip.viz.map(
    title=f"AUC {cat} {period}", cmap="RdYlGn"
)  # .plot.imshow()

# With bias correction
auc_bc.sel(category=cat).hip.viz.map(title=f"AUC BC {cat} {period}", cmap="RdYlGn")

# LTA

obs_ond = obs_lta.sel(time=datetime.datetime(1970, 12, 1))
obs_jfm = obs_lta.sel(time=datetime.datetime(1970, 3, 1))
obs_mam = obs_lta.sel(time=datetime.datetime(1970, 5, 1))

figure = obs_ond.drop_vars("time").hip.viz.map(title="RFH LTA OND", cmap=rxs.cmap)
figure.savefig("ad-hoc-evaluations/tanzania/outputs/rasters/SPI OND/rfh_lta_ond.png")

figure = obs_jfm.drop_vars("time").hip.viz.map(title="RFH LTA JFM", cmap=rxs.cmap)
figure.savefig("ad-hoc-evaluations/tanzania/outputs/rasters/SPI JFM/rfh_lta_jfm.png")

figure

figure = obs_mam.drop_vars("time").hip.viz.map(title="RFH LTA MAM", cmap=rxs.cmap)
figure.savefig("ad-hoc-evaluations/tanzania/outputs/rasters/SPI MAM/rfh_lta_mam.png")
figure
