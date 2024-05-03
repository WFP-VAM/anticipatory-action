# ---
# jupyter:
#   jupytext:
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

# ### Get final triggers file by merging:
# 1. SPI & DRYSPELL: take SPI if possible and DRYSPELL if no SPI available
# 2. GT/NRT triggers dataframes based on district vulnerability

# +
import pandas as pd

from config.params import Params

from helper_fns import (
    triggers_da_to_df,
    merge_un_biased_probs,
)
# -

# +
# Read GT and NRT
params = Params(iso='MOZ', index='DRYSPELL')
# -

# +
# Read all csvs
spigt = pd.read_csv(f"data/MOZ/outputs/Plots/triggers.aa.python.spi.{params.year}.blended.GT.csv")
drygt = pd.read_csv(f"data/MOZ/outputs/Plots/triggers.aa.python.dryspell.{params.year}.blended.GT.csv")
trigs_gt = pd.concat([spigt, drygt])

spinrt = pd.read_csv(f"data/MOZ/outputs/Plots/triggers.aa.python.spi.{params.year}.blended.NRT.csv")
drynrt = pd.read_csv(f"data/MOZ/outputs/Plots/triggers.aa.python.dryspell.{params.year}.blended.NRT.csv")
trigs_nrt = pd.concat([spinrt, drynrt])
# -

# Keep SPI by default and DRYSPELL when not available for GT
gt_merged = pd.concat([
    wcd.sort_values('index', ascending=False).head(4)
    for (w, c, d), wcd in trigs_gt.groupby(['district', 'category', 'Window'])
])

# Keep SPI by default and DRYSPELL when not available for NRT
nrt_merged = pd.concat([
    wcd.sort_values('index', ascending=False).head(4)
    for (w, c, d), wcd in trigs_nrt.groupby(['district', 'category', 'Window'])
])

# Save GT
gt_merged.to_csv(
    f"data/MOZ/outputs/Plots/triggers.aa.python.spi.dryspell.{params.year}.blended.GT.csv",
    index=False,
)

# Save NRT
nrt_merged.to_csv(
    f"data/MOZ/outputs/Plots/triggers.aa.python.spi.dryspell.{params.year}.blended.NRT.csv",
    index=False,
)

# Filter vulnerability based on district

# +
gt_merged = pd.read_csv(
    f"data/MOZ/outputs/Plots/triggers.aa.python.spi.dryspell.{params.year}.blended.GT.csv",
)
nrt_merged = pd.read_csv(
    f"data/MOZ/outputs/Plots/triggers.aa.python.spi.dryspell.{params.year}.blended.NRT.csv",
)
# -

# Merge GT and NRT
triggers_full = pd.DataFrame()
for d, v in params.districts_vulnerability.items():
    if v == 'GT':
        triggers_full = pd.concat([triggers_full, gt_merged.loc[gt_merged.district == d]])
    else:
        triggers_full = pd.concat([triggers_full, nrt_merged.loc[nrt_merged.district == d]])

# Save final triggers file
triggers_full.to_csv(
    f"data/MOZ/outputs/Plots/triggers.aa.python.spi.dryspell.{params.year}.blended.csv",
    index=False,
)
