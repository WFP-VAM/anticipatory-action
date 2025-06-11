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

# ### Check *triggers* differences after dates alignment by comparing coverage with reference outputs - Mozambique

# %cd ../..

import warnings

# +
from AA.helper_fns import get_coverage, load_trigger_with_reference

# +
from config.params import Params

warnings.filterwarnings("ignore")

# -

# Get parameters

country = "MOZ"

params = Params(iso=country, index="SPI")

# Get triggers

triggers = load_trigger_with_reference(params, "moz_align_dates")

# Compare coverages of Python GT and NRT triggers

# GENERAL TRIGGERS - NEW
columns = ["W1-Mild", "W1-Moderate", "W1-Severe", "W2-Mild", "W2-Moderate", "W2-Severe"]
get_coverage(triggers["triggers_GT"], params.districts, columns)

# GENERAL TRIGGERS - OLD
get_coverage(triggers["reference_GT"], params.districts, columns)

# NON-REGRET TRIGGERS - NEW
get_coverage(triggers["triggers_NRT"], params.districts, columns)

# NON-REGRET TRIGGERS - OLD
get_coverage(triggers["reference_NRT"], params.districts, columns)

# Get coverage of final output

get_coverage(triggers["triggers_pilots"], params.districts, columns)

get_coverage(triggers["reference_pilots"], params.districts, columns)
