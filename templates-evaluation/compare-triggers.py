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

# ### Check *triggers* pipeline with data and parameters of interest by computing coverage and comparing metrics with reference outputs

# %cd ../

# +
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
# -

# Get triggers

# +
refGT = pd.read_csv('data/MOZ/outputs/Plots/triggers.aa.python.spi.dryspell.2022.GT.csv')
refNRT = pd.read_csv('data/MOZ/outputs/Plots/triggers.aa.python.spi.dryspell.2022.NRT.csv')

ref = pd.read_csv('data/MOZ/outputs/Plots/triggers.aa.python.spi.dryspell.2022.csv')

# +
data_of_interest = "blended" # change with output names of dataset of interest

trigsGT = pd.read_csv("data/MOZ/outputs/Plots/triggers.aa.python.spi.dryspell.2022.{data_of_interest}.GT.csv")
trigsNRT = pd.read_csv("data/MOZ/outputs/Plots/triggers.aa.python.spi.dryspell.2022.{data_of_interest}.NRT.csv")

trigs = pd.read_csv("data/MOZ/outputs/Plots/triggers.aa.python.spi.dryspell.2022.{data_of_interest}.csv")


# -

# Compare coverages of Python GT and NRT triggers

def get_coverage(triggers_df, script: str, districts: list):
    if script == 'python':
        dis = 'district'; win = 'Window'; cat = 'category'
    else:
        dis = 'District'; win = 'windows'; cat = 'Category'
    cov = pd.DataFrame(columns=["W1-Leve", "W1-Moderado", "W1-Severo", "W2-Leve", "W2-Moderado", "W2-Severo"], index = districts)
    for d, r in cov.iterrows():
        val = []
        for w in triggers_df[win].unique():
            for c in triggers_df[cat].unique():
                if script == 'python':
                    val.append(len(triggers_df[(triggers_df[win]==w) & (triggers_df[cat]==c) & (triggers_df[dis]==d)]) // 2)
                else:
                    val.append(len(triggers_df[(triggers_df[win]==w) & (triggers_df[cat]==c) & (triggers_df[dis]==d)]))              
        cov.loc[d] = val

    print(f'The coverage using the {script} script is {round(100 * np.sum(cov.values > 0) / np.size(cov.values), 1)} %')
    return cov


# GENERAL TRIGGERS
get_coverage(trigsGT, 'python', trigsGT['district'].sort_values().unique())

# NON-REGRET TRIGGERS
get_coverage(trigsNRT, 'python', trigsNRT['district'].sort_values().unique())

# Get coverage of final output

get_coverage(trigs, 'python', trigs['district'].sort_values().unique())

# Difference of scores (triggers - reference) on the full output by category / district / index / issue

trigs.HR = - trigs.HR
ref.HR = - ref.HR


def compare_metric(ref, triggers, metric: str):
    dis = 'district'; win = 'Window'; cat = 'category'
    diff = pd.DataFrame(columns=["W1-Leve", "W1-Moderado", "W1-Severo", "W2-Leve", "W2-Moderado", "W2-Severo"], index = ref.district.sort_values().unique())
    for d, r in diff.iterrows():
        val = []
        for w in triggers[win].unique():
            for c in triggers[cat].unique():
                val.append(
                    triggers[(triggers[win]==w) & (triggers[cat]==c) & (triggers[dis]==d)][metric].mean() - ref[(ref[win]==w) & (ref[cat]==c) & (ref[dis]==d)][metric].mean()
                )              
        diff.loc[d] = val

    cmap = 'RdBu' * (metric == 'HR') + 'bwr' * (metric == 'FR')
    return diff.astype(float).style.background_gradient(cmap=cmap, axis=None)


# **Hit Rate (recall)**

compare_metric(ref, trigs, 'HR')

# **Failure Rate (1 - precision)**

compare_metric(ref, trigs, 'FR')
