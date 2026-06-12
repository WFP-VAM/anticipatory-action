# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: 'Python (pixi: aa-env)'
#     language: python
#     name: aa-env
# ---

# %% [markdown]
# # Trigger Selection
#
# Selects the best-performing triggers per district, indicator, and issue month
# using configurable performance thresholds and ranking logic.
#
# **Outputs** a CSV in the operational format used by the PRISM dashboard,
# with one row per (district × indicator × category × issue_ready) combination.

# %%
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import HTML, display

warnings.filterwarnings("ignore")

# %cd ..

# %% [markdown]
# ## Country configurations
#
# One dict per country. Add new countries here; no other cell needs to change.

# %%
CONFIGS = {
    "TZA": dict(
        districts=["Longido", "Monduli", "Same", "Simanjiro", "Micheweni"],
        # Performance thresholds
        min_hr=0.50,
        max_far=1,
        min_sr=0.5,
        min_rp=5,
        category="Moderate",
        # Keys = indicator name (must match triggers_metrics files),
        # values = latest allowed issue_set month.
        indicators={
            "spi_ON": 10,
            "spi_OND": 10,
            "spi_ND": 10,
            "spi_NDJ": 10,
            "spi_DJ": 10,
        },
        n_triggers=1,
        season="2025-26",
        window="ONDJ",
        vulnerability="TBD",
    ),
    "ZMB": dict(
        # Set to None to load all districts from params
        districts=None,
        min_hr=0.55,
        max_far=0.45,
        min_sr=0.65,
        min_rp=5,
        category="Moderate",
        indicators={
            "spi_DJF": 11,
            "spi_DJ": 11,
            "spi_JF": 12,
        },
        n_triggers=1,
        season="2025-26",
        window="DJF",
        vulnerability="TBD",
    ),
}

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Active configuration
#
# Change `ISO` here to switch countries. Everything else is read from `CONFIGS`.

# %%
ISO = "TZA"

cfg = CONFIGS[ISO]
DISTRICTS = cfg["districts"]  # None → loaded from params below
MIN_HR = cfg["min_hr"]
MAX_FAR = cfg["max_far"]
MIN_SR = cfg["min_sr"]
MIN_RP = cfg["min_rp"]
CATEGORY = cfg["category"]
INDICATORS = cfg["indicators"]
N_TRIGGERS = cfg["n_triggers"]
SEASON = cfg["season"]
WINDOW = cfg["window"]
VULNERABILITY = cfg["vulnerability"]

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Trigger ranking helpers


# %%
def rank_triggers(df, group, sort_cols, sort_order, n_triggers):
    """
    Select the top `n_triggers` rows per `group` based on a weighted composite
    rank over `sort_cols` (each in the direction given by `sort_order`).

    Tie-breaking within a group:
      - All 'ready' values identical → keep minimum 'set'
      - All 'set'   values identical → keep minimum 'ready'
      - Otherwise → keep minimum |ready − set|
    """
    weights = [10 ** (3 * (len(sort_cols) - i - 1)) for i in range(len(sort_cols))]
    df = df.copy()
    df["rank_score"] = sum(
        df[sort_cols[i]].rank(ascending=sort_order[i]) * weights[i]
        for i in range(len(sort_cols))
    )

    # One best row per group
    df_top = (
        df.groupby(group, group_keys=False)
        .apply(lambda x: x[x["rank_score"] == x["rank_score"].min()])
        .reset_index(drop=True)
    )

    def _tie_break(sub):
        sub = sub.copy()
        sub["difference"] = (sub["ready"] - sub["set"]).abs()
        if sub["ready"].nunique() == 1:
            return sub[sub["set"] == sub["set"].min()]
        elif sub["set"].nunique() == 1:
            return sub[sub["ready"] == sub["ready"].min()]
        else:
            return sub[sub["difference"] == sub["difference"].min()]

    df_selected = (
        df_top.groupby(group, group_keys=False).apply(_tie_break).reset_index(drop=True)
    )

    best = (
        df_selected.groupby(group, group_keys=False)
        .apply(
            lambda x: x.sort_values(sort_cols, ascending=sort_order).head(n_triggers)
        )
        .reset_index(drop=True)
    )
    best.index.name = "id"
    return best


def standard_trigger_selection(df, group=None, n_triggers=1):
    """Rank by HR (↑), SR (↑), FAR (↓)."""
    if group is None:
        group = ["index", "category", "issue_ready"]
    return rank_triggers(
        df, group, ["HR", "SR", "FAR"], [False, False, True], n_triggers
    )


def fbeta_score(row, beta=1):
    return (
        (1 + beta**2)
        * row["TP"]
        / ((1 + beta**2) * row["TP"] + row["FP"] + beta**2 * row["FN"])
    )


def fbeta_trigger_selection(df, group=None, beta=1, n_triggers=1):
    """Rank by F-beta score; beta > 1 weights HR, beta < 1 weights precision."""
    if group is None:
        group = ["index", "category", "issue_ready"]
    df = df.copy()
    df["Fbeta"] = df.apply(fbeta_score, axis=1, beta=beta)
    return rank_triggers(df, group, ["Fbeta"], [False], n_triggers)


# %% [markdown]
# ## Load and filter trigger metrics

# %%
from AA.helpers.params import Params

params = Params(iso=ISO, index="SPI")

if DISTRICTS is None:
    DISTRICTS = params.districts

# Output path — matches the primary location that read_triggers() checks:
#   {output_path}/data/{iso}/triggers/triggers.final.{params.monitoring_year}.pilots.csv
OUTPUT_PATH = f"{params.output_path}/data/{ISO.lower()}/triggers/triggers.final.{params.monitoring_year}.pilots.csv"

indicator_list = list(INDICATORS.keys())
issue_max_map = INDICATORS

records = []
missing_districts = []

for district in tqdm(DISTRICTS):
    path = (
        f"s3://dev-hip-anticipatory-action/prod/{ISO.lower()}/triggers/"
        f"triggers_metrics/triggers_metrics_tbd_{district}.csv"
    )
    try:
        df_raw = pd.read_csv(path)
    except Exception:
        missing_districts.append(district)
        continue

    # Apply performance thresholds
    mask = (
        df_raw["index"].isin(indicator_list)
        & (df_raw["HR"] >= MIN_HR)
        & (df_raw["FAR"] <= MAX_FAR)
        & (df_raw["SR"] >= MIN_SR)
        & (df_raw["RP"] >= MIN_RP)
        & (df_raw["category"] == CATEGORY)
    )
    valid = df_raw.loc[mask].copy()

    # Apply per-indicator issue cap
    chunks = [
        valid.loc[(valid["index"] == idx) & (valid["issue_set"] <= issue_max_map[idx])]
        for idx in indicator_list
    ]
    filtered = pd.concat(chunks, ignore_index=True)

    if filtered.empty:
        missing_districts.append(district)
        continue

    selected = standard_trigger_selection(
        filtered,
        group=["index", "category", "issue_ready"],
        n_triggers=N_TRIGGERS,
    )
    records.append(selected)

if missing_districts:
    print(f"⚠️  No valid triggers found for: {missing_districts}")

final = pd.concat(records, ignore_index=True)
print(f"Selected {len(final)} triggers across {final['district'].nunique()} districts.")

# %% [markdown]
# ## Coverage overview
#
# Number of triggers per district and indicator.

# %%
from AA.helpers.utils import get_coverage

final["window"] = WINDOW
coverage = get_coverage(final, sorted(DISTRICTS), columns=[f"{WINDOW} - {CATEGORY}"])

# %% [markdown]
# ### Coverage table (compact multi-column layout)


# %%
def display_coverage_table(df, n_cols=6, title=None):
    """Render a 2-column DataFrame as n_cols side-by-side blocks."""
    assert df.shape[1] == 2, "Expects exactly 2 columns."
    n = len(df)
    rows_per_col = math.ceil(n / n_cols)
    blocks = []
    for i in range(n_cols):
        chunk = df.iloc[i * rows_per_col : (i + 1) * rows_per_col]
        if chunk.empty:
            continue
        blocks.append(
            f'<div style="flex:1;min-width:120px;overflow:auto;padding-right:8px;">'
            f"{chunk.to_html(index=False)}</div>"
        )
    header = (
        f'<h3 style="font-family:system-ui,sans-serif;margin:0 0 6px 0;">{title}</h3>'
        if title
        else ""
    )
    display(
        HTML(
            f'{header}<div style="display:flex;gap:8px;align-items:flex-start;">'
            f"{''.join(blocks)}</div>"
        )
    )


display_coverage_table(
    coverage.reset_index().rename(columns={"index": "district"}),
    n_cols=6,
    title=f"{ISO} – Trigger coverage ({WINDOW} / {CATEGORY})",
)

# %% [markdown]
# ### Coverage heatmap per indicator


# %%
def coverage_by_indicator(df_triggers, districts, indicators, category):
    """
    Return a DataFrame with districts as index and indicators as columns,
    values = number of triggers found.
    """
    rows = []
    for d in sorted(districts):
        row = {"district": d}
        for idx in indicators:
            count = len(
                df_triggers.loc[
                    (df_triggers["district"] == d)
                    & (df_triggers["index"] == idx)
                    & (df_triggers["category"] == category)
                ]
            )
            row[idx] = count
        rows.append(row)
    return pd.DataFrame(rows).set_index("district")


cov_by_idx = coverage_by_indicator(final, DISTRICTS, indicator_list, CATEGORY)

# Styled table: green = covered, white = 0
styled = (
    cov_by_idx.style.background_gradient(
        cmap="YlGn", vmin=0, vmax=N_TRIGGERS, axis=None
    )
    .set_caption(f"Triggers per district × indicator ({CATEGORY})")
    .format("{:.0f}")
)
display(styled)

print(
    f"\nDistricts with ≥1 trigger: {(cov_by_idx.sum(axis=1) > 0).sum()} / {len(cov_by_idx)}"
)


# %% [markdown]
# ## Format output for operational use
#
# Produces the schema expected by the PRISM dashboard:
# `district · index · category · window · issue_ready · issue_set ·
# trigger_ready · trigger_set · vulnerability · prob_ready · prob_set ·
# season · date_ready · date_set`


# %%
def format_operational_output(df, season, window, vulnerability):
    """
    Convert the raw trigger selection output to the operational CSV schema.
    Trigger thresholds stay as decimals (0–1). Dates are derived from issue months.
    """
    # Infer the activation year from the season string (e.g. "2025-26" → 2025)
    activation_year = int(season.split("-")[0])

    out = df[
        [
            "district",
            "index",
            "category",
            "issue_ready",
            "issue_set",
            "ready",
            "set",
            "HR",
            "SR",
            "FAR",
            "RP",
        ]
    ].copy()

    out = out.rename(columns={"ready": "trigger_ready", "set": "trigger_set"})
    out["index"] = out["index"].str.upper().str.replace("_", " ")
    out["window"] = window
    out["season"] = season
    out["vulnerability"] = vulnerability
    out["prob_ready"] = np.nan
    out["prob_set"] = np.nan
    out["date_ready"] = out["issue_ready"].apply(
        lambda m: datetime(activation_year, m, 1).strftime("%Y-%m-%d")
    )
    out["date_set"] = out["issue_set"].apply(
        lambda m: datetime(activation_year, m, 1).strftime("%Y-%m-%d")
    )

    # Column order matching the reference CSV
    cols = [
        "district",
        "index",
        "category",
        "window",
        "issue_ready",
        "issue_set",
        "trigger_ready",
        "trigger_set",
        "vulnerability",
        "prob_ready",
        "prob_set",
        "season",
        "date_ready",
        "date_set",
        # Kept for traceability; drop if not needed downstream
        "HR",
        "SR",
        "FAR",
        "RP",
    ]
    return (
        out[cols]
        .sort_values(["district", "index", "category", "issue_ready"])
        .reset_index(drop=True)
    )


output = format_operational_output(final, SEASON, WINDOW, VULNERABILITY)
output

# %% [markdown]
# ## Optional: validate with PRISM schema

# %%
from AA.helpers.utils import validate_prism_dataframe

validate_prism_dataframe(output)

# %% [markdown]
# ## Save

# %%
output.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(output)} rows to:\n  {OUTPUT_PATH}")
