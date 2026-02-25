# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: hdc
#     language: python
#     name: conda-env-hdc-py
# ---

# %%
import pandas as pd

# %%
ISO = "ZMB"
DISTRICT = "Chirundu"
PATH = f"s3://wfp-ops-userdata/amine.barkaoui/aa/data/{ISO.lower()}/triggers/triggers_metrics/triggers_metrics_tbd_{DISTRICT}.csv"

# %%
MIN_HR=.5 #.56 
MAX_FAR=.45 #.25
MIN_SR=.65  
RP=5
CATEGORY='Moderate'

# %%
df = pd.read_csv(PATH)


# %%
def rank_triggers(df, group, sort_cols, sort_order, n_triggers):
    # Step 1: Create a composite rank score based on performance metrics
    # This allows ranking by multiple columns (e.g., HR, SR, FAR) with decreasing weight
    # The first column dominates, the second breaks ties, the third is fallback
    weights = [10 ** (3 * (len(sort_cols) - i - 1)) for i in range(len(sort_cols))]  # e.g. [1e6, 1e3, 1] for 3 cols
    df['rank_score'] = sum(
        df[sort_cols[i]].rank(ascending=sort_order[i]) * weights[i]
        for i in range(len(sort_cols))
    )

    # Step 2: Select the best-performing row per group + issue_ready
    # This ensures one candidate per issue month pair based on the composite score
    df_top = df.groupby(group, group_keys=False).apply(
        lambda x: x[x['rank_score'] == x['rank_score'].min()]
    ).reset_index(drop=True)

    # Step 3: Apply tie-breaking logic when multiple rows share the same rank
    # If all 'ready' values are the same → choose row with minimum 'set'
    # If all 'set' values are the same → choose row with minimum 'ready'
    # Otherwise → choose row with minimum absolute difference between 'ready' and 'set'
    def tie_breaker(subdf):
        all_ready_same = subdf['ready'].nunique() == 1
        all_set_same = subdf['set'].nunique() == 1
        subdf['difference'] = (subdf['ready'] - subdf['set']).abs()

        if all_ready_same:
            return subdf[subdf['set'] == subdf['set'].min()]
        elif all_set_same:
            return subdf[subdf['ready'] == subdf['ready'].min()]
        else:
            return subdf[subdf['difference'] == subdf['difference'].min()]

    df_selected = df_top.groupby(group, group_keys=False).apply(tie_breaker).reset_index(drop=True)

    # Step 4: Select top 'n_triggers' per group across all issue month pairs
    # Final sorting is done using the original sort_cols and sort_order
    best = df_selected.groupby(group, group_keys=False).apply(
        lambda x: x.sort_values(sort_cols, ascending=sort_order).head(n_triggers)
    ).reset_index(drop=True)

    # Set index name for clarity
    best.index.name = 'id'
    return best
    
def standard_trigger_selection(df, group=['index', 'category', 'issue_ready'], n_triggers=2):
    return rank_triggers(df, group, ['HR', 'SR', 'FAR'], [False, False, True], n_triggers)

def fbeta_trigger_selection(df, group=['index','category','issue_ready'], beta=1, n_triggers=2):
    # Beta values above 1 give more importance to hit rate, below 1 to FAR
    df["Fbeta"] = df.apply(lambda x: fbeta_score(x, beta=beta), axis=1)
    return rank_triggers(df, group, ['Fbeta'], [False], n_triggers)

def fbeta_score(x, beta=1):
    return (1 + beta**2)*x['TP'] / ( (1 + beta**2)*x['TP'] + x['FP'] + (beta**2)*x['FN'])  


# %% [markdown]
# ### Compliant with constraints

# %%
valid_df = df[(df["HR"]>MIN_HR) & (df["FAR"]<MAX_FAR) & (df["SR"]>MIN_SR) & (df['RP'] == RP) & (df['category'] == CATEGORY)]
valid_df.shape

# %%
standard_trigger_selection(valid_df, ['index','category','issue_ready'], n_triggers=1)

# %%
c = fbeta_trigger_selection(valid_df, ['index','category','issue_ready'], 1, n_triggers=1)

# %%
c

# %%
c[c['category']=='Moderate']

# %%
c.loc[231], c.loc[198]

# %% [markdown]
# ## Gwembe

# %%
ISO = "ZMB"
DISTRICT = "Gwembe"
PATH = f"s3://wfp-ops-userdata/amine.barkaoui/aa/data/{ISO.lower()}/triggers/triggers_metrics/triggers_metrics_tbd_{DISTRICT}.csv"
df = pd.read_csv(PATH)

# %%
df.head()

# %%
valid_df = df[(df["HR"]>MIN_HR) & (df["FAR"]<MAX_FAR) & (df["SR"]>MIN_SR) & (df['RP'] == RP) & (df['category'] == CATEGORY)]
valid_df.shape

# %%
standard_trigger_selection(valid_df, ['index','category','issue_ready'], n_triggers=1)

# %%
fbeta_trigger_selection(valid_df, ['index','category','issue_ready'], 1, n_triggers=1)

# %%
