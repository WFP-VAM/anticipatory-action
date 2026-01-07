import logging
import re

import click
import fsspec
import pandas as pd

from AA.src.params import S3_OPS_DATA_PATH

logging.basicConfig(level="INFO", force=True)


def pivot_triggers_pilots_data(df):
    """
    Transform triggers pilots DataFrame by keeping specific columns and pivoting
    ready/set data into a long format with a phase column.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame from aa_probabilities_triggers_pilots.csv

    Returns:
    --------
    pd.DataFrame
        Transformed DataFrame with columns: district, category, issue, index, prob, trigger, phase
        where phase takes values 'ready' or 'set'
    """
    # Select required columns
    cols_to_keep = [
        "district",
        "category",
        "index",
        "issue_ready",
        "issue_set",
        "prob_ready",
        "prob_set",
        "trigger_ready",
        "trigger_set",
    ]
    df_subset = df[cols_to_keep].copy()

    # Pivot the data to create long format
    df_ready = df_subset[
        ["district", "category", "index", "issue_ready", "prob_ready", "trigger_ready"]
    ].copy()
    df_ready["phase"] = "ready"
    df_ready = df_ready.rename(
        columns={
            "issue_ready": "issue",
            "prob_ready": "prob",
            "trigger_ready": "trigger",
        }
    )

    df_set = df_subset[
        ["district", "category", "index", "issue_set", "prob_set", "trigger_set"]
    ].copy()
    df_set["phase"] = "set"
    df_set = df_set.rename(
        columns={"issue_set": "issue", "prob_set": "prob", "trigger_set": "trigger"}
    )

    # Concatenate the two DataFrames
    result_df = pd.concat([df_ready, df_set], ignore_index=True)

    # Reorder columns
    result_df = result_df[
        ["district", "category", "issue", "index", "prob", "trigger", "phase"]
    ]

    return result_df


def validate_probabilities_match(prob_df, trigger_df):
    """
    Validate that probabilities match between SPI dataset and pivoted triggers pilots dataset.
    Only checks matches for rows that exist in the trigger_df.
    Excludes expected missing cases where trigger issue > max issue in prob_df.

    Parameters:
    -----------
    prob_df : pd.DataFrame
        DataFrame from aa_probabilities_spi_6.csv with columns: district, category, issue, index, prob
    trigger_df : pd.DataFrame
        Pivoted DataFrame from pivot_triggers_pilots_data() function

    Returns:
    --------
    dict
        Dictionary with validation results containing:
        - 'matches': bool - True if all probabilities match for trigger_df rows
        - 'mismatches': pd.DataFrame - Rows where probabilities don't match
        - 'missing_in_prob': pd.DataFrame - Unexpected rows in trigger_df not found in prob_df
        - 'expected_missing': pd.DataFrame - Expected missing rows (issue > max available)
    """
    # Find global maximum issue value in prob_df
    max_issue = prob_df["issue"].max()
    logging.info(f"Maximum issue value in probability data: {max_issue}")

    # Separate expected missing (issue > max_issue) from cases that should have matches
    expected_missing = trigger_df[trigger_df["issue"] > max_issue].copy()
    should_match = trigger_df[trigger_df["issue"] <= max_issue].copy()

    logging.info(
        f"Expected missing rows (issue > {max_issue}): {len(expected_missing)}"
    )
    logging.info(f"Rows that should have matches: {len(should_match)}")

    # Merge the datasets on the key columns, keeping only should_match rows
    merge_cols = ["district", "category", "issue", "index"]

    merged = pd.merge(
        should_match,
        prob_df,
        on=merge_cols,
        how="left",
        suffixes=("_trigger", "_prob"),
        indicator=True,
    )

    # Find rows that exist in both datasets
    both_datasets = merged[merged["_merge"] == "both"].copy()

    # Find mismatches in probability values
    if len(both_datasets) > 0:
        both_datasets["prob_match"] = (
            abs(both_datasets["prob_trigger"] - both_datasets["prob_prob"]) < 1e-10
        )
        mismatches = both_datasets[~both_datasets["prob_match"]]
        mismatches["impacts_programme"] = (
            mismatches["trigger"] > mismatches["prob_trigger"]
        ) != (mismatches["trigger"] > mismatches["prob_prob"])
    else:
        mismatches = pd.DataFrame()

    # Find unexpected missing rows (should have matches but don't)
    unexpected_missing = merged[merged["_merge"] == "left_only"][
        merge_cols + ["prob_trigger", "phase"]
    ]

    # Check if all probabilities match (excluding expected missing cases)
    matches = len(mismatches) == 0 and len(unexpected_missing) == 0

    return {
        "matches": matches,
        "mismatches": (
            mismatches[
                merge_cols
                + ["trigger", "prob_trigger", "prob_prob", "phase", "impacts_programme"]
            ]
            if len(mismatches) > 0
            else pd.DataFrame()
        ),
        "missing_in_prob": unexpected_missing,
        "expected_missing": (
            expected_missing[merge_cols + ["prob", "phase"]]
            if len(expected_missing) > 0
            else pd.DataFrame()
        ),
    }


@click.command()
@click.argument("iso", required=True, type=str)
def main(iso):
    """
    Main function to validate probability matching between SPI and triggers pilots datasets.

    ISO: Country ISO code for validation
    """
    logging.info(f"Validating probabilities for country: {iso}")

    # Convert ISO to lowercase to match directory structure
    iso_lower = iso.lower()

    # Construct file paths
    data_dir = f"{S3_OPS_DATA_PATH}/data/{iso_lower}/probs/"
    triggers_file = f"{data_dir}aa_probabilities_triggers_pilots.csv"

    # Find all probability files using fsspec
    logging.info("Finding all probability files...")
    logging.info(f"Reading from: {data_dir}")

    fs = fsspec.filesystem("s3")
    all_files = fs.ls(data_dir.replace("s3://", ""))

    # Filter files matching the pattern aa_probabilities_{spi|dryspell}_{number}.csv
    prob_files = []
    pattern = re.compile(r"aa_probabilities_(spi|dryspell)_\d+\.csv$")

    for file_path in all_files:
        file_name = file_path.split("/")[-1]
        if pattern.match(file_name):
            prob_files.append(f"s3://{file_path}")

    logging.info(
        f"Found {len(prob_files)} probability files: {[f.split('/')[-1] for f in prob_files]}"
    )

    # Read and append all probability files
    prob_dfs = []
    for file_path in prob_files:
        df = pd.read_csv(file_path)
        prob_dfs.append(df)
        logging.info(f"Loaded {len(df)} rows from {file_path.split('/')[-1]}")

    # Concatenate all probability DataFrames
    if prob_dfs:
        prob_df = pd.concat(prob_dfs, ignore_index=True)
        logging.info(f"Combined probability data: {len(prob_df)} total rows")
    else:
        raise FileNotFoundError("No probability files found matching the pattern")

    # Read the triggers file
    triggers_raw_df = pd.read_csv(triggers_file)
    logging.info(
        f"Loaded {len(triggers_raw_df)} rows from {triggers_file.split('/')[-1]}"
    )

    # Pivot the triggers data
    logging.info("Pivoting triggers data...")
    trigger_df = pivot_triggers_pilots_data(triggers_raw_df)
    logging.info(f"Pivoted data contains {len(trigger_df)} rows")

    # Validate probabilities match
    logging.info("Validating probability matches...")
    validation_results = validate_probabilities_match(prob_df, trigger_df)

    # Log results
    if validation_results["matches"]:
        logging.info("✅ All probabilities match!")
    else:
        logging.warning("❌ Found mismatches or missing data:")

        if len(validation_results["mismatches"]) > 0:
            logging.error(
                f"🔍 {len(validation_results['mismatches'])} probability mismatches found:"
            )
            logging.error(f"\n{validation_results['mismatches']}")

        if len(validation_results["missing_in_prob"]) > 0:
            logging.error(
                f"📊 {len(validation_results['missing_in_prob'])} unexpected missing rows in trigger data:"
            )
            logging.error(f"\n{validation_results['missing_in_prob']}")

    # Report expected missing cases (informational)
    if len(validation_results["expected_missing"]) > 0:
        logging.info(
            f"💡 {len(validation_results['expected_missing'])} rows expected to be missing (issue > max available):"
        )
        logging.info(
            f"📊 These have issue values {validation_results['expected_missing']['issue'].unique()}"
        )


if __name__ == "__main__":
    main()
