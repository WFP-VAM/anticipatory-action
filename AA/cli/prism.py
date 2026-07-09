import argparse
import pandas as pd
from AA.helpers.params import S3_OPS_DATA_PATH
from AA.helpers.prism import update_prism_dashboard


def main(iso3: str):
    iso3 = iso3.lower()
    pilot_path = f"{S3_OPS_DATA_PATH}/{iso3}/probs/aa_probabilities_triggers_pilots.csv"
    print(f"📥 Reading probs pilot data from: {pilot_path}")
    pilot_df = pd.read_csv(pilot_path)
    update_prism_dashboard(iso3, pilot_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update PRISM with latest AA probabilities"
    )
    parser.add_argument("iso3", type=str, help="ISO3 country code (e.g., moz)")
    args = parser.parse_args()
    main(args.iso3)
