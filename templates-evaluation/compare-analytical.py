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

# ### Check *analytical* pipeline using datasets/parameters of interest by comparing results with reference outputs at the district level

# %cd ../

# +
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import geopandas as gpd
import xskillscore as xss

from odc.geo.geobox import GeoBox
# -

from helper_fns import read_observations_locally, read_forecasts_locally, aggregate_by_district
from config.params import Params
from analytical import calculate_forecast_probabilities, evaluate_forecast_probabilities, get_verification_df
from hip.analysis.analyses.drought import get_accumulation_periods
from hip.analysis.aoi import AnalysisArea

# Prepare input data

fc = read_forecasts_locally('data/MOZ/forecast/Moz_SAB_tp_ecmwf_01/*.nc')

params = Params(iso='MOZ', index='SPI')
params.issue=1

obs = read_observations_locally(f"data/MOZ/chirps")

triggers_df = pd.read_csv(f"data/{params.iso}/outputs/Plots/triggers.aa.python.spi.dryspell.2022.csv")
gdf = gpd.read_file(f"data/{params.iso}/shapefiles/moz_admbnda_2019_SHP/moz_admbnda_adm2_2019.shp")

accumulation_periods = get_accumulation_periods(
    fc, 
    params.start_season,
    params.end_season,
    params.min_index_period, 
    params.max_index_period
)
accumulation_periods
period = accumulation_periods['AM']

fc = fc.where(fc.time < np.datetime64('2022-07-01T12:00:00.000000000'), drop=True)

# Test functions for ```calculate_forecast_probabilities``` and ```evaluate_forecast_probabilities```

params.year = 2022


def get_test_input(month="01"):
    np.random.seed(42)
    test_input_fc = xr.Dataset(
        data_vars=dict(
            tp=(["time", "ensemble", "longitude", "latitude"], np.random.rand(len(range(1993, 2022)) * 31, 5, 3, 3)),
        ),
        coords = dict(
            time=pd.concat([pd.Series(pd.date_range(f"{yyyy}-{month}-01", f"{yyyy}-{month}-31")) for yyyy in range(1993, 2022)]),
            ensemble=range(5),
            longitude=[-1, 0, 1],
            latitude=[-1, 0, 1],
        )
    )
    test_input_obs = xr.Dataset(
        data_vars=dict(precip=(["time", "longitude", "latitude"], np.random.rand(len(range(1993, 2022)) * 31, 3, 3))),
        coords = dict(
            time=pd.concat([pd.Series(pd.date_range(f"{yyyy}-{month}-01", f"{yyyy}-{month}-31")) for yyyy in range(1993, 2022)]),
            longitude=[-1, 0, 1],
            latitude=[-1, 0, 1],
        )
    )
    test_input_probas = xr.Dataset(
        data_vars=dict(tp=(["year"], np.random.rand(len(range(1993, 2022))))),
        coords = dict(year=range(1993, 2022))
    )
    np.random.seed(43)
    test_input_probas_bc = xr.Dataset(
        data_vars=dict(scen=(["year"], np.random.rand(len(range(1993, 2022))))),
        coords = dict(year=range(1993, 2022))
    )
    test_input_levels = xr.Dataset(
        data_vars=dict(precip=(["year"], np.random.randint(0, 2, len(range(1993, 2022))))),
        coords = dict(year=range(1993, 2022))
    )
    return test_input_fc, test_input_obs, test_input_probas, test_input_probas_bc, test_input_levels


def test_calculate_forecast_probabilities():
    np.random.seed(42)
    test_input_fc1, test_input_obs1, _, _, _ = get_test_input("01")
    probabilities_a, bc_a, _, levels_obs_a = calculate_forecast_probabilities(test_input_fc1, test_input_obs1, params, (1), 1) 
    probabilities_b, bc_b, _, levels_obs_b = calculate_forecast_probabilities(test_input_fc1, test_input_obs1, params, (1), 12) 
    
    ref_probas_month1 = xr.DataArray(
        np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.13333333, 0.2, 0.2, 0.4, 0.2, 0., 0.06666667, 0., 0.2, 0.33333333, 0., 0.2, 0.2, 0.13333333, 0.2, 0.6, 0, 0.6, 0.33333333, 0.2, 0.2, 0.26666667, 0.13333333, 0.]),
        coords=dict(year=range(1993, 2022))
    ).assign_coords(dict(longitude=0, latitude=0))
    ref_bc_month1_issue1 = xr.DataArray(
        np.array([0.2, 0.26666667, 0.2, 0.33333333, 0.2, 0.06666667, 0.26666667, 0.2, 0.4, 0.26666667, 0., 0.06666667, 0., 0.2, 0.26666667, 0.06666667, 0.2, 0.13333333, 0.2, 0.2, 0.73333333, 0., 0.6, 0.33333333, 0.2, 0.2, 0.26666667, 0.2, 0.]),
        coords=dict(year=range(1993, 2022))
    ).assign_coords(dict(longitude=0, latitude=0))
    ref_bc_month1_issue12 = xr.DataArray(
        np.array([0.13333333, 0.26666667, 0.2, 0.13333333, 0.2, 0.2 , 0.26666667, 0.2 , 0.4, 0.26666667, 0., 0.06666667, 0., 0.2, 0.4, 0., 0.2, 0.2, 0.26666667, 0.2, 0.66666667, 0., 0.6, 0.46666667, 0.2, 0.2, 0.33333333, 0.2, 0]),
        coords=dict(year=range(1993, 2022))
    ).assign_coords(dict(longitude=0, latitude=0))
    ref_obs_month1 = xr.DataArray(
        np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.66666667, 0, 0, 1, 0, 1, 0.33333333, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.33333333]),
        coords=dict(year=range(1993, 2022))
    ).assign_coords(dict(longitude=0, latitude=0))

    xr.testing.assert_allclose(probabilities_a.sel(latitude=0, longitude=0).mean('category').tp, ref_probas_month1)
    xr.testing.assert_allclose(probabilities_b.sel(latitude=0, longitude=0).mean('category').tp, ref_probas_month1)

    xr.testing.assert_allclose(bc_a.sel(latitude=0, longitude=0).mean('category').scen, ref_bc_month1_issue1)
    xr.testing.assert_allclose(bc_b.sel(latitude=0, longitude=0).mean('category').scen, ref_bc_month1_issue12)
    
    xr.testing.assert_allclose(levels_obs_a.sel(latitude=0, longitude=0).mean('category').precip, ref_obs_month1)
    xr.testing.assert_allclose(levels_obs_b.sel(latitude=0, longitude=0).mean('category').precip, ref_obs_month1)

    print("\nFORECASTS PROBABILITIES TESTS PASSED")
ref_probas_month1_issue1 = test_calculate_forecast_probabilities()


# +
def test_evaluate_forecast_probabilities():
    _, _, test_probas, test_probas_bc, test_obs_levels = get_test_input()
    test_auc, test_auc_bc = evaluate_forecast_probabilities(test_probas, test_probas_bc, test_obs_levels)

    xr.testing.assert_allclose(test_auc, xr.DataArray(0.45192308))
    xr.testing.assert_allclose(test_auc_bc, xr.DataArray(0.19230769))

    print("\nFORECASTS VERIFICATION TESTS PASSED")

test_evaluate_forecast_probabilities()


# -

# ### Comparison on the full output at the district level

def plot_hist(comparison, title, xlabel, xmin, xmax, s=1, mask_text=False):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize = (6,4))

    quant_5, quant_25, quant_50, quant_75, quant_95 = comparison.difference.quantile(0.05), comparison.difference.quantile(0.25), comparison.difference.quantile(0.5), comparison.difference.quantile(0.75), comparison.difference.quantile(0.95)

    # [quantile, opacity, length]
    quants = [[quant_5, 0.6, 0.16], [quant_25, 0.8, 0.26], [quant_50, 1, 0.36],  [quant_75, 0.8, 0.46], [quant_95, 0.6, 0.56]]

    comparison.difference.plot(kind='hist', density = True, alpha = 0.65, bins=200)
    comparison.difference.plot(kind='kde')

    # Plot the lines with a loop
    import matplotlib.pyplot as plt
    for i in quants: plt.axvline(i[0], alpha = i[1], ymax = i[2], linestyle = ":")

    # X
    ax.set_xlabel(xlabel)
    ax.set_xlim(xmin, xmax) 

    # Y
    ax.set_yticklabels([])
    ax.set_ylabel("")

    if not(mask_text):
        plt.text(quant_5-.01, 0.15 * s, "5th", size = 10, alpha = 0.8)
        plt.text(quant_25-.013, 0.27 * s, "25th", size = 11, alpha = 0.85)
        plt.text(quant_50-.013, 0.33 * s, "50th", size = 12, alpha = 1)
        plt.text(quant_75-.013, 0.39 * s, "75th", size = 11, alpha = 0.85)
        plt.text(quant_95-.025, 0.47 * s, "95th Percentile", size = 10, alpha =.8)

    # Overall
    ax.grid(False)
    ax.set_title(title, size = 17, pad = 10)

    # Remove ticks and spines
    ax.tick_params(left = False, bottom = False)
    for ax, spine in ax.spines.items():
        spine.set_visible(False)


fbfref = pd.read_csv('data/MOZ/outputs/Districts_FbF/spi/fbf.districts.roc.spi.2022.txt')
fbfref.columns = ['district', 'category', 'AUC_ref', 'BC_ref', 'Index', 'issue']
fbfref

# +
data_of_interest = "blended" # change with output name of dataset of interest

fbfP = pd.read_csv('data/MOZ/outputs/Districts_FbF/spi/fbf.districts.roc.spi.2022.{data_of_interest}.txt')
fbfP
# -

# Ratio of bias-corrected values in the final output

# Ratio of cases using Bias Correction
fbfP.BC.sum() / len(fbfP)

# Histogram of difference between full outputs (auc_to_compare - python_reference)

comparison = fbfP.set_index(['district', 'category', 'Index', 'issue']).join(fbfref.set_index(['district', 'category', 'Index', 'issue'])).reset_index()

comparison['difference'] = comparison.AUC_best - comparison.AUC_ref

plot_hist(comparison, title="R/Python AUC (BC and not) difference at the district level", xlabel="AUC difference", xmin=-.5, xmax=.5, s=10)

# ### Difference (auc_to_compare - python_reference) on the full output by category / district / index / issue

# +
x_axis = [5, 6, 7, 8, 9, 10, 11, 12, 1, 2]
y_axis = ['SPI ON', 'SPI ND', 'SPI DJ', 'SPI JF', 'SPI FM', 'SPI MA', 'SPI AM', 'SPI OND', 'SPI NDJ', 'SPI DJF', 'SPI JFM', 'SPI FMA', 'SPI MAM']

def draw_heatmap(**kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index="Index", columns="issue", values="difference")
    d = d.reindex(index=y_axis, columns=x_axis)
    sns.heatmap(d, **kwargs)

fg = sns.FacetGrid(comparison.loc[(comparison.category=='Severo')], col='district', col_wrap=4)
fg.map_dataframe(draw_heatmap, annot=True, annot_kws={"size":4}, cbar=True, cmap='RdYlGn', center=0)
fg.fig.subplots_adjust(top=0.9)
fg.fig.suptitle('SEVERE CATEGORY')
# -

fg = sns.FacetGrid(comparison.loc[(comparison.category=='Moderado')], col='district', col_wrap=4)
fg.map_dataframe(draw_heatmap, annot=True, annot_kws={"size":4}, cbar=True, cmap='RdYlGn', center=0)
fg.fig.subplots_adjust(top=0.9)
fg.fig.suptitle('MODERATE CATEGORY')

fg = sns.FacetGrid(comparison.loc[(comparison.category=='Leve')], col='district', col_wrap=4)
fg.map_dataframe(draw_heatmap, annot=True, annot_kws={"size":4}, cbar=True, cmap='RdYlGn', center=0)
fg.fig.subplots_adjust(top=0.9)
fg.fig.suptitle('MILD CATEGORY')

# ### Visualisation of differences for each pair of variables to highlight any systematic difference

# **Average difference**

# +
x_axis = [5, 6, 7, 8, 9, 10, 11, 12, 1, 2]
y_axis = ['SPI ON', 'SPI ND', 'SPI DJ', 'SPI JF', 'SPI FM', 'SPI MA', 'SPI AM', 'SPI OND', 'SPI NDJ', 'SPI DJF', 'SPI JFM', 'SPI FMA', 'SPI MAM']

def draw_heatmap(**kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index="Index", columns="issue", values="difference")
    d = d.reindex(index=y_axis, columns=x_axis)
    sns.heatmap(d, **kwargs)

fg = sns.FacetGrid(
    comparison.groupby(['issue', 'Index', 'category']).mean('district').reset_index(), 
    col='category', 
    col_wrap=3
)
fg.map_dataframe(draw_heatmap, annot=True, annot_kws={"size":4}, cbar=True, cmap='RdYlGn', center=0)

# +
x_axis = [5, 6, 7, 8, 9, 10, 11, 12, 1, 2]
y_axis = ["Chiure", "Chibuto", "Chicualacuala", "Guija", "Mabalane", "Mapai", "Massingir", "Caia", "Chemba", "Changara", "Marara"]

def draw_heatmap(**kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index="district", columns="issue", values="difference")
    d = d.reindex(index=y_axis, columns=x_axis)
    sns.heatmap(d, **kwargs)

fg = sns.FacetGrid(
    comparison.groupby(['issue', 'district', 'category']).mean('Index').reset_index(), 
    col='category', 
    col_wrap=3
)
fg.map_dataframe(draw_heatmap, annot=True, annot_kws={"size":4}, cbar=True, cmap='RdYlGn', center=0)

# +
x_axis = ['SPI ON', 'SPI ND', 'SPI DJ', 'SPI JF', 'SPI FM', 'SPI MA', 'SPI AM', 'SPI OND', 'SPI NDJ', 'SPI DJF', 'SPI JFM', 'SPI FMA', 'SPI MAM']
y_axis = ["Chiure", "Chibuto", "Chicualacuala", "Guija", "Mabalane", "Mapai", "Massingir", "Caia", "Chemba", "Changara", "Marara"]

def draw_heatmap(**kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index="district", columns="Index", values="difference")
    d = d.reindex(index=y_axis, columns=x_axis)
    sns.heatmap(d, **kwargs)

fg = sns.FacetGrid(
    comparison.groupby(['Index', 'district', 'category']).mean('issue').reset_index(), 
    col='category', 
    col_wrap=3
)
fg.map_dataframe(draw_heatmap, annot=True, annot_kws={"size":4}, cbar=True, cmap='RdYlGn', center=0)
