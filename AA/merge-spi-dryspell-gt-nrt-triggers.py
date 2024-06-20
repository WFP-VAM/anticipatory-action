# ---
# jupyter:
#   jupytext:
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

# ### Get final triggers file by merging:
# 1. SPI & DRYSPELL: take SPI if possible and DRYSPELL if no SPI available
# 2. GT/NRT triggers dataframes based on district vulnerability


# +
import pandas as pd
from config.params import Params

from helper_fns import merge_un_biased_probs, triggers_da_to_df
# -

# Read GT and NRT
params = Params(iso="MOZ", index="SPI")

# +
# Read all csvs
spigt = pd.read_csv(
    f"/s3/scratch/amine.barkaoui/aa/data/{params.iso.lower()}/triggers/triggers.spi.{params.year}.GT.csv"
)
drygt = pd.read_csv(
    f"/s3/scratch/amine.barkaoui/aa/data/{params.iso.lower()}/triggers/triggers.dryspell.{params.year}.GT.csv"
)
trigs_gt = pd.concat([spigt, drygt])
trigs_gt["vulnerability"] = "GT"

spinrt = pd.read_csv(
    f"/s3/scratch/amine.barkaoui/aa/data/{params.iso.lower()}/triggers/triggers.spi.{params.year}.NRT.csv"
)
drynrt = pd.read_csv(
    f"/s3/scratch/amine.barkaoui/aa/data/{params.iso.lower()}/triggers/triggers.dryspell.{params.year}.NRT.csv"
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
    f"/s3/scratch/amine.barkaoui/aa/data/{params.iso.lower()}/triggers/triggers.spi.dryspell.{params.year}.GT.csv",
    index=False,
)

# Save NRT
nrt_merged.to_csv(
    f"/s3/scratch/amine.barkaoui/aa/data/{params.iso.lower()}/triggers/triggers.spi.dryspell.{params.year}.NRT.csv",
    index=False,
)

# Filter vulnerability based on district

gt_merged = pd.read_csv(
    f"/s3/scratch/amine.barkaoui/aa/data/{params.iso.lower()}/triggers/triggers.spi.dryspell.{params.year}.GT.csv",
)
nrt_merged = pd.read_csv(
    f"/s3/scratch/amine.barkaoui/aa/data/{params.iso.lower()}/triggers/triggers.spi.dryspell.{params.year}.NRT.csv",
)

# Merge GT and NRT
triggers_full = pd.DataFrame()
for d, v in params.districts_vulnerability.items():
    if v == "GT":
        triggers_full = pd.concat(
            [triggers_full, gt_merged.loc[gt_merged.district == d]]
        )
    else:
        triggers_full = pd.concat(
            [triggers_full, nrt_merged.loc[nrt_merged.district == d]]
        )

# Filter vulnerability by taking ET first

gt_merged = pd.read_csv(
    f"/s3/scratch/amine.barkaoui/aa/data/{params.iso.lower()}/triggers/triggers.spi.dryspell.{params.year}.GT.csv",
)
nrt_merged = pd.read_csv(
    f"/s3/scratch/amine.barkaoui/aa/data/{params.iso.lower()}/triggers/triggers.spi.dryspell.{params.year}.NRT.csv",
)

# +
# Take NRT
triggers_full = pd.DataFrame()
for d, v in params.districts_vulnerability.items():
    triggers_full = pd.concat([triggers_full, nrt_merged.loc[nrt_merged.district == d]])
    
if params.iso == 'ZWE':
    triggers_full = triggers_full.loc[triggers_full.category == 'Normal']
# -

# Visualize coverage

import numpy as np
def get_coverage(triggers_df, districts: list, columns: list):
    cov = pd.DataFrame(
        columns=columns,
        index=districts,
    )
    for d, r in cov.iterrows():
        val = []
        for w in triggers_df["Window"].unique():
            for c in triggers_df["category"].unique():
                if script == "python":
                    val.append(
                        len(
                            triggers_df[
                                (triggers_df["Window"] == w)
                                & (triggers_df["category"] == c)
                                & (triggers_df["district"] == d)
                            ]
                        )
                        // 2
                    )
                else:
                    val.append(
                        len(
                            triggers_df[
                                (triggers_df["Window"] == w)
                                & (triggers_df["category"] == c)
                                & (triggers_df["district"] == d)
                            ]
                        )
                    )
        cov.loc[d] = val

    print(
        f"The coverage using the {script} script is {round(100 * np.sum(cov.values > 0) / np.size(cov.values), 1)} %"
    )
    return cov


columns = ["W1-Mild", "W1-Moderate", "W1-Severe", "W2-Mild", "W2-Moderate", "W2-Severe"]
#columns = ["W1-Normal", "W2-Normal"]
get_coverage(triggers_full, triggers_full["district"].sort_values().unique(), columns)

# Ratio of dryspell triggers
len(triggers_full.loc[triggers_full["index"].str[0] == "d"]) / len(triggers_full)

triggers_full.loc[triggers_full.Window == 'Window 2'].head(10)

# Save final triggers file
triggers_full.to_csv(
    f"/s3/scratch/amine.barkaoui/aa/data/{params.iso.lower()}/triggers/triggers.spi.dryspell.{params.year}.pilots.csv",
    index=False,
)


