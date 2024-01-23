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

# This notebook can be used to evaluate the forecast skill (ROC score) for a specific forecast dataset. It currently works for a specific index and a specific issue month as it has a research / exploratory / analytical purpose. In order to run a proper comparison, check the `compare_analytical` notebook. 

# +
import numpy as np

from config.params import Params

from helper_fns import (
    read_forecasts_locally,
    read_observations_locally,
)
from analytical import calculate_forecast_probabilities

from hip.analysis.ops._statistics import evaluate_roc_forecasts

# %cd ../

# +
issue = "09"

params = Params(iso='MOZ', index='SPI')

# TODO replace by downscaled forecasts (climax / bil. interp. / weighted mask) issued in 9 (1993 - 2022)
forecasts = read_forecasts_locally(
    f"data/{params.iso}/forecast/Moz_SAB_tp_ecmwf_{issue}/*.nc"
).isel(latitude=slice(0, 32), longitude=slice(0, 32))

forecasts = forecasts.where(
    forecasts.time < np.datetime64(f"{params.year}-07-01T12:00:00.000000000"),
    drop=True,
)

# TODO replace by `rfh_daily`
observations = read_observations_locally(f"data/{params.iso}/chirps").isel(latitude=slice(0, 32), longitude=slice(0, 32))
# -

forecasts

observations

# +
period_months = (12, 1, 2) # Dec Jan Feb

probs, probs_bc, obs_values, obs_bool = calculate_forecast_probabilities(
    forecasts,
    observations,
    params,
    period_months,
    issue,
)
# -

auc, auc_bc = evaluate_roc_forecasts(
    obs_bool.precip,
    probs.tp,
    probs_bc.scen,
)

# +
# Without bias correction

print(f"Average AUC score: {auc.sel(category='Severo').mean().values}")
auc.sel(category='Severo').plot.imshow()

# +
# With bias correction

print(f"Average AUC score: {auc_bc.sel(category='Severo').mean().values}")
auc_bc.sel(category='Severo').plot.imshow()
