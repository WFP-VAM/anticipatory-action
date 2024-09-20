# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: hip-workshop
#     language: python
#     name: hip-workshop
# ---

# ### Get final triggers file by merging:
# 1. SPI & DRYSPELL: take SPI if possible and DRYSPELL if no SPI available
# 2. GT/NRT triggers dataframes based on district vulnerability


# %cd ..

# +
import pandas as pd
from config.params import Params

from AA.helper_fns import format_triggers_df_for_dashboard, get_coverage, merge_un_biased_probs, triggers_da_to_df
# -

# Read GT and NRT
params = Params(iso="MOZ", index="SPI")

params.data_path

# +
# Read all csvs
spigt = pd.read_csv(
    f"{params.data_path}/data/{params.iso.lower()}/triggers/triggers.spi.{params.year}.GT.csv"
)
drygt = pd.read_csv(
    f"{params.data_path}/data/{params.iso.lower()}/triggers/triggers.dryspell.{params.year}.GT.csv"
)
trigs_gt = pd.concat([spigt, drygt])
trigs_gt["vulnerability"] = "GT"

spinrt = pd.read_csv(
    f"{params.data_path}/data/{params.iso.lower()}/triggers/triggers.spi.{params.year}.NRT.csv"
)
drynrt = pd.read_csv(
    f"{params.data_path}/data/{params.iso.lower()}/triggers/triggers.dryspell.{params.year}.NRT.csv"
)
trigs_nrt = pd.concat([spinrt, drynrt])
trigs_nrt["vulnerability"] = "NRT"
# -

# Keep SPI by default and DRYSPELL when not available for GT
gt_merged = pd.concat(
    [
        wcd.sort_values("index", ascending=False).head(8)
        for (d, c, w), wcd in trigs_gt.groupby(["district", "category", "Window"])
    ]
)

# Keep SPI by default and DRYSPELL when not available for NRT
nrt_merged = pd.concat(
    [
        wcd.sort_values("index", ascending=False).head(8)
        for (d, c, w), wcd in trigs_nrt.groupby(["district", "category", "Window"])
    ]
)

# Save GT
gt_merged.to_csv(
    f"{params.data_path}/data/{params.iso.lower()}/triggers/triggers.spi.dryspell.{params.year}.GT.csv",
    index=False,
)

# Save NRT
nrt_merged.to_csv(
    f"{params.data_path}/data/{params.iso.lower()}/triggers/triggers.spi.dryspell.{params.year}.NRT.csv",
    index=False,
)

# Filter vulnerability based on district

gt_merged = pd.read_csv(
    f"{params.data_path}/data/{params.iso.lower()}/triggers/triggers.spi.dryspell.{params.calibration_year}.GT.csv",
)
nrt_merged = pd.read_csv(
    f"{params.data_path}/data/{params.iso.lower()}/triggers/triggers.spi.dryspell.{params.calibration_year}.NRT.csv",
)

if 'Window' in gt_merged.columns:
    gt_merged = format_triggers_df_for_dashboard(gt_merged, params)
    nrt_merged = format_triggers_df_for_dashboard(nrt_merged, params)

# Filter vulnerability based on district: merge GT and NRT
triggers_pilots = pd.DataFrame()
for d, v in params.districts_vulnerability.items():
    if v == "GT":
        if params.iso == "zwe":
            tmp = gt_merged.loc[
                (gt_merged.district == d) & (gt_merged.category == "Moderate")
            ]
        else:
            tmp = gt_merged.loc[gt_merged.district == d]
        triggers_pilots = pd.concat([triggers_pilots, tmp])
    else:
        if params.iso == "zwe":
            tmp = nrt_merged.loc[
                (nrt_merged.district == d) & (nrt_merged.category == "Normal")
            ]
        else:
            tmp = nrt_merged.loc[nrt_merged.district == d]
        triggers_pilots = pd.concat([triggers_pilots, tmp])

# Take all NRT districts for MOZ
if params.iso == "moz":
    triggers_full = nrt_merged

# Visualize coverage

# + jupyter={"outputs_hidden": true}
# Pilot triggers
#columns = ["W1-Mild", "W1-Moderate", "W1-Severe", "W2-Mild", "W2-Moderate", "W2-Severe"]
columns = ["W1-Moderate", "W2-Moderate"]
get_coverage(triggers_pilots, triggers_pilots["district"].sort_values().unique(), columns)
# -

# Full list if MOZ
get_coverage(triggers_full, triggers_full["district"].sort_values().unique(), columns)

# Ratio of dryspell triggers
len(triggers_pilots.loc[triggers_pilots["index"].str[0] == "D"]) / len(triggers_pilots)

# Save final triggers files
triggers_pilots.to_csv(
    f"{params.data_path}/data/{params.iso.lower()}/triggers/triggers.spi.dryspell.{params.calibration_year}.GT.pilots.csv",
    index=False,
)

# If triggers for all districts exist
triggers_full.to_csv(
    f"{params.data_path}/data/{params.iso.lower()}/triggers/triggers.spi.dryspell.{params.year}.all.districts.csv",
    index=False,
)
