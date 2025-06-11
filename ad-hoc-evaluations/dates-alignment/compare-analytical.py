# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# ### Check *analytical* pipeline differences after dates alignment by comparing results with reference outputs at the district level

# %cd ../../

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.params import Params

# Prepare input data

country = "MOZ"

params = Params(iso=country, index="DRYSPELL")


# ### Comparison on the full output at the district level


def plot_hist(comparison, title, xlabel, xmin, xmax, s=1, mask_text=False):
    fig, ax = plt.subplots(figsize=(6, 4))

    quant_5, quant_25, quant_50, quant_75, quant_95 = (
        comparison.difference.quantile(0.05),
        comparison.difference.quantile(0.25),
        comparison.difference.quantile(0.5),
        comparison.difference.quantile(0.75),
        comparison.difference.quantile(0.95),
    )

    # [quantile, opacity, length]
    quants = [
        [quant_5, 0.6, 0.16],
        [quant_25, 0.8, 0.26],
        [quant_50, 1, 0.36],
        [quant_75, 0.8, 0.46],
        [quant_95, 0.6, 0.56],
    ]

    comparison.difference.plot(kind="hist", density=True, alpha=0.65, bins=200)
    comparison.difference.plot(kind="kde")

    # Plot the lines with a loop
    for i in quants:
        plt.axvline(i[0], alpha=i[1], ymax=i[2], linestyle=":")

    # X
    ax.set_xlabel(xlabel)
    ax.set_xlim(xmin, xmax)

    # Y
    ax.set_yticklabels([])
    ax.set_ylabel("")

    if not (mask_text):
        plt.text(quant_5 - 0.01, 0.15 * s, "5th", size=10, alpha=0.8)
        plt.text(quant_25 - 0.013, 0.27 * s, "25th", size=11, alpha=0.85)
        plt.text(quant_50 - 0.013, 0.33 * s, "50th", size=12, alpha=1)
        plt.text(quant_75 - 0.013, 0.39 * s, "75th", size=11, alpha=0.85)
        plt.text(quant_95 - 0.025, 0.47 * s, "95th Percentile", size=10, alpha=0.8)

    # Overall
    ax.grid(False)
    ax.set_title(title, size=17, pad=10)

    # Remove ticks and spines
    ax.tick_params(left=False, bottom=False)
    for ax, spine in ax.spines.items():
        spine.set_visible(False)


fbfref = pd.read_csv(
    f"{params.data_path}/data/{params.iso}/auc/fbf.districts.roc.{params.index}.{params.calibration_year}.csv"
)
fbfref = fbfref.rename(columns={"AUC_best": "AUC_ref", "BC": "BC_ref"})
fbfref

fbf_aligned = pd.read_csv(
    f"{params.data_path}/data/{params.iso}_align_dates/auc/fbf.districts.roc.{params.index}.{params.calibration_year}.csv"
)
fbf_aligned

# Ratio of bias-corrected values in the final output

# Ratio of cases using Bias Correction
fbf_aligned.BC.sum() / len(fbf_aligned)

# Histogram of difference between full outputs (auc_to_compare - python_reference)

comparison = (
    fbf_aligned.set_index(["district", "category", "Index", "issue"])
    .join(fbfref.set_index(["district", "category", "Index", "issue"]))
    .reset_index()
)

comparison["difference"] = comparison.AUC_best - comparison.AUC_ref

# Filter comparison on pilot districts
comparison = comparison.loc[comparison.district.isin(params.districts)]

# Histogram of non-null differences
plot_hist(
    comparison.loc[comparison.difference != 0],
    title="Before/After dates alignment AUC (BC and not) difference at the district level",
    xlabel="AUC difference (New - Old)",
    xmin=-0.05,
    xmax=0.05,
    s=100,
)


# ### Difference (auc_to_compare - python_reference) on the full output by category / district / index / issue

# +
x_axis = [5, 6, 7, 8, 9, 10, 11, 12, 1, 2]
y_axis = [
    "ON",
    "ND",
    "DJ",
    "JF",
    "FM",
    "MA",
    "AM",
    "MJ",
    "OND",
    "NDJ",
    "DJF",
    "JFM",
    "FMA",
    "MAM",
    "AMJ",
]
y_axis = [params.index.upper() + " " + ind for ind in y_axis]


def draw_heatmap(**kwargs):
    data = kwargs.pop("data")
    d = data.pivot(index="Index", columns="issue", values="difference")
    d = d.reindex(index=y_axis, columns=x_axis)
    sns.heatmap(d, **kwargs)


fg = sns.FacetGrid(
    comparison.loc[(comparison.category == "Severe")], col="district", col_wrap=5
)
fg.map_dataframe(
    draw_heatmap, annot=True, annot_kws={"size": 4}, cbar=True, cmap="RdYlGn", center=0
)
fg.fig.subplots_adjust(top=0.9)
fg.fig.suptitle("SEVERE CATEGORY")
# -

fg = sns.FacetGrid(
    comparison.loc[(comparison.category == "Moderate")], col="district", col_wrap=5
)
fg.map_dataframe(
    draw_heatmap, annot=True, annot_kws={"size": 4}, cbar=True, cmap="RdYlGn", center=0
)
fg.fig.subplots_adjust(top=0.9)
fg.fig.suptitle("MODERATE CATEGORY")

fg = sns.FacetGrid(
    comparison.loc[(comparison.category == "Mild")], col="district", col_wrap=5
)
fg.map_dataframe(
    draw_heatmap, annot=True, annot_kws={"size": 4}, cbar=True, cmap="RdYlGn", center=0
)
fg.fig.subplots_adjust(top=0.9)
fg.fig.suptitle("MILD CATEGORY")

# ### Visualisation of differences for each pair of variables to highlight any systematic difference

# **Average difference**

# +
x_axis = [5, 6, 7, 8, 9, 10, 11, 12, 1, 2]
y_axis = [
    "ON",
    "ND",
    "DJ",
    "JF",
    "FM",
    "MA",
    "AM",
    "MJ",
    "OND",
    "NDJ",
    "DJF",
    "JFM",
    "FMA",
    "MAM",
    "AMJ",
]
y_axis = [params.index.upper() + " " + ind for ind in y_axis]


def draw_heatmap(**kwargs):
    data = kwargs.pop("data")
    d = data.pivot(index="Index", columns="issue", values="difference")
    d = d.reindex(index=y_axis, columns=x_axis)
    sns.heatmap(d, **kwargs)


fg = sns.FacetGrid(
    comparison.groupby(["issue", "Index", "category"]).mean("district").reset_index(),
    col="category",
    col_wrap=3,
)
fg.map_dataframe(
    draw_heatmap, annot=True, annot_kws={"size": 4}, cbar=True, cmap="RdYlGn", center=0
)

# +
x_axis = [5, 6, 7, 8, 9, 10, 11, 12, 1, 2]
y_axis = [
    "Chiure",
    "Chibuto",
    "Chicualacuala",
    "Guija",
    "Mabalane",
    "Mapai",
    "Massingir",
    "Caia",
    "Chemba",
    "Changara",
    "Marara",
]


def draw_heatmap(**kwargs):
    data = kwargs.pop("data")
    d = data.pivot(index="district", columns="issue", values="difference")
    d = d.reindex(index=y_axis, columns=x_axis)
    sns.heatmap(d, **kwargs)


fg = sns.FacetGrid(
    comparison.groupby(["issue", "district", "category"]).mean("Index").reset_index(),
    col="category",
    col_wrap=3,
)
fg.map_dataframe(
    draw_heatmap, annot=True, annot_kws={"size": 4}, cbar=True, cmap="RdYlGn", center=0
)

# +
x_axis = [
    "ON",
    "ND",
    "DJ",
    "JF",
    "FM",
    "MA",
    "AM",
    "MJ",
    "OND",
    "NDJ",
    "DJF",
    "JFM",
    "FMA",
    "MAM",
    "AMJ",
]
x_axis = [params.index.upper() + " " + ind for ind in x_axis]
y_axis = [
    "Chiure",
    "Chibuto",
    "Chicualacuala",
    "Guija",
    "Mabalane",
    "Mapai",
    "Massingir",
    "Caia",
    "Chemba",
    "Changara",
    "Marara",
]


def draw_heatmap(**kwargs):
    data = kwargs.pop("data")
    d = data.pivot(index="district", columns="Index", values="difference")
    d = d.reindex(index=y_axis, columns=x_axis)
    sns.heatmap(d, **kwargs)


fg = sns.FacetGrid(
    comparison.groupby(["Index", "district", "category"]).mean("issue").reset_index(),
    col="category",
    col_wrap=3,
)
fg.map_dataframe(
    draw_heatmap, annot=True, annot_kws={"size": 4}, cbar=True, cmap="RdYlGn", center=0
)
# -
