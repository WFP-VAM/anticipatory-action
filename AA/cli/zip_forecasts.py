import argparse
import os
import shutil
import subprocess

from AA.helpers.params import S3_OPS_DATA_PATH

S3_PUBLIC_PATH = "s3://hip-workshop-sharing-public-eu-central-1-485262375119/anticipatory-action"

ZIP_NAME = "forecasts.zip"
ZARR_NAME = "forecasts.zarr"


def main(iso3: str, issue_month: int):
    # Normalize inputs
    iso3 = iso3.lower()
    month_padded = str(issue_month).zfill(2)

    # Define paths
    zarr_path = f"{S3_OPS_DATA_PATH}/{iso3}/zarr/{month_padded}/forecasts.zarr"
    tmp_path = f"/tmp/{iso3}/{month_padded}"
    zip_output_path = f"{S3_PUBLIC_PATH}/{iso3}/forecasts/{month_padded}/{ZIP_NAME}"

    # Check if zip already exists
    print(f"🔍 Checking if {ZIP_NAME} already exists at {zip_output_path}...")
    result = subprocess.run(
        ["aws", "s3", "ls", zip_output_path], capture_output=True
    )
    if result.returncode == 0:
        print(f"⚠️  WARNING: {ZIP_NAME} already exists at {zip_output_path}. It will be overwritten.")

    # Create local tmp directory
    print(f"📁 Creating local directory: {tmp_path}")
    os.makedirs(tmp_path, exist_ok=True)

    # Copy Zarr from S3 to local
    print(f"📦 Copying Zarr file from {zarr_path} to {tmp_path}")
    subprocess.run(
        ["aws", "s3", "cp", "--recursive", zarr_path, f"{tmp_path}/{ZARR_NAME}"],
        check=True,
    )

    # Zip the Zarr directory
    print(f"🗜️ Zipping {ZARR_NAME} into {ZIP_NAME}")
    shutil.make_archive(
        base_name=f"{tmp_path}/forecasts",  # output path without .zip
        format="zip",
        root_dir=tmp_path,
        base_dir=ZARR_NAME,
    )

    # Upload zip to S3
    print(f"🚀 Uploading {ZIP_NAME} to {zip_output_path}")
    subprocess.run(
        ["aws", "s3", "cp", f"{tmp_path}/{ZIP_NAME}", zip_output_path],
        check=True,
    )

    # Cleanup
    print("🧹 Cleaning up local files")
    shutil.rmtree(tmp_path)

    print(f"✅ Done: {ZIP_NAME} uploaded to {zip_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zip and upload forecasts Zarr to S3")
    parser.add_argument("iso3", type=str, help="ISO3 country code (e.g., moz)")
    parser.add_argument("issue_month", type=int, help="Issue month as integer (e.g., 3 for March)")
    args = parser.parse_args()
    main(args.iso3, args.issue_month)