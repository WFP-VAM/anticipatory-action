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
import os

import pandas as pd

from config.params import Params

# -

# Read GT and NRT
params = Params(iso="MOZ", index="SPI")

print(params.iso)

# +
# ### Get and save final dataframe

# Now, you are done with the processing of one index (SPI or DRYSPELL). So you can rerun everything from the beginning with the other index. If you've already done it, you can run the next cell so it will merge all the different outputs to provide you with the very final dataframe that will be used operationally.

# The next cells merge SPI and DRYSPELL for each vulnerability level. So SPI is taken first, and if no SPI is available DRYSPELL is included.

# Read all GT csvs
if os.path.exists(
    f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.{params.calibration_year}.GT.csv"
):
    spigt = pd.read_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.{params.calibration_year}.GT.csv"
    )
    drygt = pd.read_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.dryspell.{params.calibration_year}.GT.csv"
    )
    trigs_gt = pd.concat([spigt, drygt])
    trigs_gt["vulnerability"] = "GT"

    # Keep SPI by default and DRYSPELL when not available for GT
    gt_merged = pd.concat(
        [
            wcd.sort_values("index", ascending=False).head(4)
            for (d, c, w), wcd in trigs_gt.groupby(["district", "category", "window"])
        ]
    )

    # Save GT
    gt_merged.to_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.GT.csv",
        index=False,
    )
# -

gt_merged.head()

# Read all NRT csvs
if os.path.exists(
    f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.{params.calibration_year}.NRT.csv"
):
    spinrt = pd.read_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.{params.calibration_year}.NRT.csv"
    )
    drynrt = pd.read_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.dryspell.{params.calibration_year}.NRT.csv"
    )
    trigs_nrt = pd.concat([spinrt, drynrt])
    trigs_nrt["vulnerability"] = "NRT"

    # Keep SPI by default and DRYSPELL when not available for NRT
    nrt_merged = pd.concat(
        [
            wcd.sort_values("index", ascending=False).head(4)
            for (d, c, w), wcd in trigs_nrt.groupby(["district", "category", "window"])
        ]
    )

    # Save NRT
    nrt_merged.to_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.NRT.csv",
        index=False,
    )

nrt_merged.head()

# +
# Now we read dataframes for both vulnerability levels if they exist and merge them according to the vulnerability defined for each district in the config file.

# Read GT and NRT dataframes
if os.path.exists(
    f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.GT.csv"
):
    gt_merged = pd.read_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.GT.csv",
    )
if os.path.exists(
    f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.NRT.csv"
):
    nrt_merged = pd.read_csv(
        f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.NRT.csv",
    )

# Filter vulnerability based on district: merge GT and NRT
triggers_full = pd.DataFrame()
for d, v in params.districts_vulnerability.items():
    if v == "GT":
        if params.iso == "zwe":
            tmp = gt_merged.loc[
                (gt_merged.district == d) & (gt_merged.category == "Moderate")
            ]
        else:
            tmp = gt_merged.loc[gt_merged.district == d]
        triggers_full = pd.concat([triggers_full, tmp])
    else:
        if params.iso == "zwe":
            tmp = nrt_merged.loc[
                (nrt_merged.district == d) & (nrt_merged.category == "Normal")
            ]
        else:
            tmp = nrt_merged.loc[nrt_merged.district == d]
        triggers_full = pd.concat([triggers_full, tmp])
# -

triggers_full.head()

# Save final triggers file
triggers_full.to_csv(
    f"{params.data_path}/data/{params.iso}/triggers/triggers.spi.dryspell.{params.calibration_year}.pilots.csv",
    index=False,
)
