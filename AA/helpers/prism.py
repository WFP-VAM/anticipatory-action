# AA/helpers/prism.py
import pandas as pd
from AA.helpers.utils import validate_prism_dataframe

PRISM_BUCKET = "hip-workshop-sharing-public-eu-central-1-485262375119"

DATE_COLUMNS = ["date_ready", "date_set"]  # adjust to match your actual PRISM schema


def _normalize_dates(df: pd.DataFrame, columns) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col], format="mixed", dayfirst=False
            ).dt.strftime("%Y-%m-%d")
    return df


def update_prism_dashboard(
    iso3: str,
    pilot_df: pd.DataFrame,
    seasons_to_keep=("2025-26", "2024-25", "2023-24"),
    read_path: str | None = None,
    write_path: str | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Merge freshly computed pilot trigger probabilities (`pilot_df`) into the
    historical PRISM dashboard data for `iso3`, apply label corrections, and
    write the result either to the shared PRISM S3 path (default) or to
    `write_path` if provided (e.g. a local path for notebook/dev use).

    read_path: where to read the existing PRISM history from. Defaults to the
        shared S3 PRISM path. Pass a local path here to avoid any S3 access
        (e.g. a copy you've already downloaded once for local dev/testing).
    write_path: where to write the merged result. Defaults to the same S3
        PRISM path as read_path (i.e. updates the live dashboard in place).
        Pass a local path here to write only locally.
    dry_run: if True, skip writing entirely (merge + validate only).
    """
    iso3 = iso3.lower()
    default_prism_path = f"s3://{PRISM_BUCKET}/anticipatory-action/{iso3}/prism/aa_probabilities_triggers_{iso3}.csv"

    source_path = read_path or default_prism_path
    target_path = write_path or default_prism_path

    print(f"📥 Reading PRISM data from: {source_path}")
    prism_df = pd.read_csv(source_path)
    if "Unnamed: 0" in prism_df.columns:
        prism_df = prism_df.drop("Unnamed: 0", axis=1)

    print("🔗 Concatenating filtered PRISM and pilot data...")
    df_concat = pd.concat(
        [prism_df.loc[prism_df.season.isin(seasons_to_keep)], pilot_df]
    ).reset_index(drop=True)

    if iso3 == "moz":
        print("🗺️ Applying district name corrections for Mozambique...")
        districts_mapping = {
            "Cahora_Bassa": "Cahora Bassa",
            "Cidade_Da_Beira": "Cidade Da Beira",
        }
        df_concat["district"] = [
            districts_mapping.get(v, v) for v in df_concat.district.values
        ]

    print("🧭 Mapping vulnerability labels...")
    vulnerability_mapping = {"GT": "General Triggers", "NRT": "Emergency Triggers"}
    df_concat["vulnerability"] = [
        vulnerability_mapping.get(v, v) for v in df_concat.vulnerability.values
    ]

    print("🪟 Mapping window labels...")
    window_mapping = {"Window1": "Window 1", "Window2": "Window 2"}
    df_concat["window"] = [window_mapping.get(v, v) for v in df_concat.window.values]

    print("Validating dataframe...")
    df_concat = _normalize_dates(df_concat, DATE_COLUMNS)
    validate_prism_dataframe(df_concat)

    if dry_run:
        print("🧪 dry_run=True — skipping write.")
    else:
        print(f"💾 Saving processed data to: {target_path}")
        df_concat.to_csv(target_path, index=False)
        print("✅ PRISM update complete.")

    return df_concat
