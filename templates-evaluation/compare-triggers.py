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

from triggers import find_optimal_triggers


# Test functions for ```find_optimal_triggers```

# +
def test_find_optimal_triggers_guija():
    result, _ = find_optimal_triggers(
        np.array([0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),
        np.array([0,-0.42752841,-1.14785480,1.00835359,1.01152670,0.08140295,1.07925391,2.89334059,0.49650693,-2.10596442,-0.11560103,1.13604414,-0.61310524,0.99173242,-0.80315828,-1.33247614,-0.39426482,-0.69886303,0.70436287,-0.68371397,1.43038058,0.55627447,-0.60290152,-0.89140522,1.09524286,0.35090649,0.65150774,0.17972234,1.73618770,-0.32053682]),   
        np.array([0.19403316,0.40413737,0.03950670,0.19867207,0.36617446,0.12769049,0.09033364,0.08100221,0.28882408,0.27224869,0.30656403,0.19762351,0.14381145,0.25782165,0.13781390,0.07984945,0.29496825,0.10694549,0.18244502,0.19856039,0.22373931,0.27777267,0.51857853,0.08000000,0.19938798,0.26805690,0.36049867,0.27718082,0.19917504,0]), 
        np.array([0.14940780,0.32687554,0.10491829,0.17928207,0.28445852,0.01592254,0.10268304,0.23625106,0.21073855,0.38182610,0.11036947,0.18485942,0.07852152,0.30479109,0.14028412,0.27518070,0.33471456,0.07077074,0.31999999,0.0986679,0.16447723,0.27975520,0.15368882,0.15867205,0.22455618,0.37894413,0.37922379,0.17322889,0.15639387,0]), 
        1,
        10, 
        'Moderado', 
        'NRT',
    )
    return np.testing.assert_equal(result, np.array([0.28, 0.12]))

test_find_optimal_triggers_guija()


# +
def test_find_optimal_triggers_chibuto():
    result, _ = find_optimal_triggers(
        np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0]),
        np.array([1.02401698,-0.97512817,-0.30439997,-1.13334739,0.09295806,1.02707446,1.40903413,1.30331767,1.27853560,1.45409751,0.66932976,0.48368409,-0.03136307,-0.41989726,0.34743443,-0.23032485,-0.24260041,-0.22509143,0.13833992,0.66685081,0.14981982,-1.80140960,-0.48558316,-0.56034160,-0.89039266,-1.13516653,-0.05548601,0.09209020,0.08816186]),   
        np.array([0.124493897, 0.199551255, 0.130161867, 0.142827168, 0.203590319, 0.076381147, 0.003629984, 0.078416705, 0.128630683, 0.173713341, 0.098256983, 0.062093060, 0.182418644, 0.121198095, 0.154454961, 0.159007564, 0.166668668, 0.155030504, 0.099209271, 0.249668032, 0.270679384, 0.226243481, 0.197488889, 0.150757939, 0.234825417, 0.188320413, 0.234617561, 0.129712150, 0.141309336]), 
        np.array([0.203457773, 0.211932480, 0.015837038, 0.000000000, 0.257075459, 0.063148580, 0.035188183, 0.073143244, 0.005135113, 0.375244647, 0.007092558, 0.232922435, 0.029359102, 0.079231590, 0.006114150, 0.086469837, 0.305699944, 0.030308658, 0.002925272, 0.080881782, 0.095813885, 0.192097038, 0.343963861, 0.008978794, 0.015903212, 0.254426777, 0.133801639, 0.072654307, 0.018615846]), 
        10,
        5, 
        'Severo', 
        'NRT',
    )
    return np.testing.assert_equal(result, np.array([0.18, 0.14]))

test_find_optimal_triggers_chibuto()
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
