# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: hdc
#     language: python
#     name: conda-env-hdc-py
# ---

# ### Comparison of final outcomes 
#
# This notebook allows to get an idea of the concrete consequences of a harmonization of dates / seasons in the gamma standardization and bias correction between the different issue months and indicators. This comparison is based on the 2024-2025 season and also includes the fix of a bug in the triggers filtering (final step after brute optimization).

# %cd ../..

# +
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
# -

from config.params import Params

country = "MOZ"

params = Params(iso=country, index='SPI')

# ### New outcomes

df_merged = pd.read_csv(
    f"{params.data_path}/data/{params.iso}_align_dates/probs/aa_probabilities_triggers_pilots_helpersfixed.csv",
)

df_merged

# +
MAPPING = {0: 'NA', 1:'NA', 2: 'NA', 3: 'Set'}

for (d, c, w), case in df_merged.groupby(['district', 'category', 'window']):
    case['ready'] = case['prob_ready'] > case['trigger_ready']
    case['set'] = case['prob_set'] > case['trigger_set']
    case['state'] = 2*case['ready'] + 1*case['set']
    case['state'] = [MAPPING[s] for s in case.state]
    case.loc[case.issue_set == 2, 'state'] = 'NotSetYet'
    state = case.state.max()
    
    df_merged.loc[(df_merged.district == d) & (df_merged.window == w) & (df_merged['category'] == c), 'state'] = state
# -

outcomes_new = pd.DataFrame(columns=["W1-Leve", "W1-Moderado", "W1-Severo", "W2-Leve", "W2-Moderado", "W2-Severo"], index = df_merged['district'].sort_values().unique())
for d, r in outcomes_new.iterrows():
    val = []
    for w in sorted(df_merged['window'].unique()):
        for c in df_merged['category'].unique():
                val.append(df_merged[(df_merged['window']==w) & (df_merged['category']==c) & (df_merged['district']==d)].state.unique()[0])
    outcomes_new.loc[d] = val

outcomes_new.style.format(na_rep="missing so far")

# ### Previous outcomes

ref = pd.read_csv(
    f"{params.data_path}/data/{params.iso}/probs/aa_probabilities_triggers_pilots.csv",
)

for (d, c, w), case in ref.groupby(['district', 'category', 'window']):
    case['ready'] = case['prob_ready'] > case['trigger_ready']
    case['set'] = case['prob_set'] > case['trigger_set']
    case['state'] = 2*case['ready'] + 1*case['set']
    case['state'] = [MAPPING[s] for s in case.state]
    case.loc[case.issue_set == 2, 'state'] = 'NotSetYet'
    state = case.state.max()
    
    ref.loc[(ref.district == d) & (ref.window == w) & (ref['category'] == c), 'state'] = state

outcomes_old = pd.DataFrame(columns=["W1-Leve", "W1-Moderado", "W1-Severo", "W2-Leve", "W2-Moderado", "W2-Severo"], index = ref['district'].sort_values().unique())
for d, r in outcomes_old.iterrows():
    val = []
    for w in sorted(ref['window'].unique()):
        for c in ['Mild', 'Moderate', 'Severe']:
            val.append(ref[(ref['window']==w) & (ref['category']==c) & (ref['district']==d)].state.unique()[0])
    outcomes_old.loc[d] = val

outcomes_old.style.format(na_rep="missing so far")


# ### More detailed comparison

def compare_new_old_outcomes(old, new):
    comp = pd.DataFrame(columns=["W1-Leve", "W1-Moderado", "W1-Severo", "W2-Leve", "W2-Moderado", "W2-Severo"], index = old.index)
    for d, row in comp.iterrows():
        for col in comp.columns:
       
            if new.loc[d, col] == 'Set' and old.loc[d, col] == 'Set':
                comp.loc[d, col] = 'both set'
            elif new.loc[d, col] == 'NA' and old.loc[d, col] == 'NA':
                comp.loc[d, col] = 'both NA'
            elif new.loc[d, col] == 'Ready' and old.loc[d, col] == 'Ready':
                comp.loc[d, col] = 'both ready'
            elif new.loc[d, col] == 'Set' and old.loc[d, col] == 'NA':
                comp.loc[d, col] = 'now set'
            elif new.loc[d, col] == 'NA' and old.loc[d, col] == 'Set':
                comp.loc[d, col] = 'was set before'
            elif type(new.loc[d, col]) is not str or type(old.loc[d, col]) is not str:
                comp.loc[d, col] = 'missing so far'
            else: 
                comp.loc[d, col] = 'not complete'

    colors = {'now set': 'darkgreen', 'both set': 'mediumseagreen', 'both NA': 'mediumseagreen', 'both ready': 'mediumseagreen', 'was set before': 'coral', 'missing so far': 'burlywood', 'not complete': 'burlywood'}
    return comp.style.map(lambda val: 'background-color: {}'.format(colors.get(val,'')))


compare_new_old_outcomes(outcomes_old, outcomes_new)
