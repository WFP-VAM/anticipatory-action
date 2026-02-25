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

# %cd ..

# +
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from AA.cli.analytical import calculate_forecast_probabilities
from AA.helpers.params import Params

# -

# +
# Skip those tests as the functions used and the expected results need to be updated
# after the dates alignment update and the zonal stats update
# See PR #29 and PR #30
# TODO update these tests in a future PR


# Test functions for ```calculate_forecast_probabilities``` and ```evaluate_forecast_probabilities```

params = Params(iso="MOZ", index="SPI")
params.issue = 1
params.year = 2022


@pytest.mark.skip(reason="Update needed after dates alignment and zonal stats update")
def get_test_input(month="01"):
    np.random.seed(42)
    test_input_fc = xr.Dataset(
        data_vars=dict(
            tp=(
                ["time", "ensemble", "longitude", "latitude"],
                np.random.rand(len(range(1993, 2022)) * 31, 5, 3, 3),
            ),
        ),
        coords=dict(
            time=pd.concat(
                [
                    pd.Series(pd.date_range(f"{yyyy}-{month}-01", f"{yyyy}-{month}-31"))
                    for yyyy in range(1993, 2022)
                ]
            ),
            ensemble=range(5),
            longitude=[-1, 0, 1],
            latitude=[-1, 0, 1],
        ),
    )
    test_input_obs = xr.Dataset(
        data_vars=dict(
            precip=(
                ["time", "longitude", "latitude"],
                np.random.rand(len(range(1993, 2022)) * 31, 3, 3),
            )
        ),
        coords=dict(
            time=pd.concat(
                [
                    pd.Series(pd.date_range(f"{yyyy}-{month}-01", f"{yyyy}-{month}-31"))
                    for yyyy in range(1993, 2022)
                ]
            ),
            longitude=[-1, 0, 1],
            latitude=[-1, 0, 1],
        ),
    )
    return test_input_fc, test_input_obs


# +
@pytest.mark.skip(reason="Update needed after dates alignment and zonal stats update")
def test_calculate_forecast_probabilities():
    np.random.seed(42)
    test_input_fc1, test_input_obs1 = get_test_input("01")
    probabilities_a, bc_a, _, levels_obs_a = calculate_forecast_probabilities(
        test_input_fc1, test_input_obs1, params, (1), 1
    )
    probabilities_b, bc_b, _, levels_obs_b = calculate_forecast_probabilities(
        test_input_fc1, test_input_obs1, params, (1), 12
    )

    ref_probas_month1 = xr.DataArray(
        np.array(
            [
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.13333333,
                0.2,
                0.2,
                0.4,
                0.2,
                0.0,
                0.06666667,
                0.0,
                0.2,
                0.33333333,
                0.0,
                0.2,
                0.2,
                0.13333333,
                0.2,
                0.6,
                0,
                0.6,
                0.33333333,
                0.2,
                0.2,
                0.26666667,
                0.13333333,
                0.0,
            ]
        ),
        coords=dict(year=range(1993, 2022)),
    ).assign_coords(dict(longitude=0, latitude=0))
    ref_bc_month1_issue1 = xr.DataArray(
        np.array(
            [
                0.2,
                0.26666667,
                0.2,
                0.33333333,
                0.2,
                0.06666667,
                0.26666667,
                0.2,
                0.4,
                0.26666667,
                0.0,
                0.06666667,
                0.0,
                0.2,
                0.26666667,
                0.06666667,
                0.2,
                0.13333333,
                0.2,
                0.2,
                0.73333333,
                0.0,
                0.6,
                0.33333333,
                0.2,
                0.2,
                0.26666667,
                0.2,
                0.0,
            ]
        ),
        coords=dict(year=range(1993, 2022)),
    ).assign_coords(dict(longitude=0, latitude=0))
    ref_bc_month1_issue12 = xr.DataArray(
        np.array(
            [
                0.13333333,
                0.26666667,
                0.2,
                0.13333333,
                0.2,
                0.2,
                0.26666667,
                0.2,
                0.4,
                0.26666667,
                0.0,
                0.06666667,
                0.0,
                0.2,
                0.4,
                0.0,
                0.2,
                0.2,
                0.26666667,
                0.2,
                0.66666667,
                0.0,
                0.6,
                0.46666667,
                0.2,
                0.2,
                0.33333333,
                0.2,
                0,
            ]
        ),
        coords=dict(year=range(1993, 2022)),
    ).assign_coords(dict(longitude=0, latitude=0))
    ref_obs_month1 = xr.DataArray(
        np.array(
            [
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0.66666667,
                0,
                0,
                1,
                0,
                1,
                0.33333333,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0.33333333,
            ]
        ),
        coords=dict(year=range(1993, 2022)),
    ).assign_coords(dict(longitude=0, latitude=0))

    xr.testing.assert_allclose(
        probabilities_a.sel(latitude=0, longitude=0).mean("category").tp,
        ref_probas_month1,
    )
    xr.testing.assert_allclose(
        probabilities_b.sel(latitude=0, longitude=0).mean("category").tp,
        ref_probas_month1,
    )

    xr.testing.assert_allclose(
        bc_a.sel(latitude=0, longitude=0).mean("category").scen, ref_bc_month1_issue1
    )
    xr.testing.assert_allclose(
        bc_b.sel(latitude=0, longitude=0).mean("category").scen, ref_bc_month1_issue12
    )

    xr.testing.assert_allclose(
        levels_obs_a.sel(latitude=0, longitude=0).mean("category").precip,
        ref_obs_month1,
    )
    xr.testing.assert_allclose(
        levels_obs_b.sel(latitude=0, longitude=0).mean("category").precip,
        ref_obs_month1,
    )

    print("\nFORECASTS PROBABILITIES TESTS PASSED")


# -
