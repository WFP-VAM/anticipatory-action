import argparse
import pandas as pd
from AA.helpers.utils import validate_prism_dataframe
from AA.helpers.params import S3_OPS_DATA_PATH


def main(iso3: str):
    # Normalize inputs
    iso3 = iso3.lower()

    # Define paths (PRISM PATH TO BE UPDATED ONCE CONFIRMED)
    prism_path = f"s3://wfp-ops-userdata/public-share/aa/staging/aa_probabilities_triggers_{iso3}.csv"
    pilot_path = f"{S3_OPS_DATA_PATH}/{iso3}/probs/aa_probabilities_triggers_pilots.csv"

    print(f"📥 Reading PRISM data from: {prism_path}")
    prism_df = pd.read_csv(prism_path)
    print(f"prism_df columns: {prism_df.columns.tolist()}")
    if "Unnamed: 0" in prism_df.columns:
        prism_df = prism_df.drop("Unnamed: 0", axis=1)

    print(f"📥 Reading probs pilot data from: {pilot_path}")
    df = pd.read_csv(pilot_path)
    print(f"probs pilot data columns: {df.columns.tolist()}")

    print("🔗 Concatenating filtered PRISM and pilot data...")
    df_concat = (
        pd.concat(
            [
                prism_df.loc[prism_df.season.isin(["2024-25", "2023-24"])],
                df,
            ]
        )
        .reset_index()
        .drop("level_0", axis=1)
    )
    print(df_concat.head())

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
    validate_prism_dataframe(df_concat)  # Fixed: was validating `df` instead of `df_concat`

    print(f"💾 Saving processed data to: {prism_path}")
    df_concat.to_csv(prism_path, index=False)

    print("✅ Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update PRISM with latest AA probabilities")
    parser.add_argument("iso3", type=str, help="ISO3 country code (e.g., moz)")
    args = parser.parse_args()
    main(args.iso3)