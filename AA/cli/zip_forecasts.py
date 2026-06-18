import argparse
import os
import shutil
import boto3
from botocore.exceptions import ClientError
from AA.helpers.params import S3_OPS_DATA_PATH

S3_PUBLIC_BUCKET = "hip-workshop-sharing-public-eu-central-1-485262375119"
S3_PUBLIC_PREFIX = "anticipatory-action"
ZIP_NAME = "forecasts.zip"
ZARR_NAME = "forecasts.zarr"


def s3_key_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def s3_download_recursive(s3_client, bucket: str, prefix: str, local_dir: str):
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = os.path.relpath(key, prefix)
            local_path = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3_client.download_file(bucket, key, local_path)


def s3_upload_file(s3_client, local_path: str, bucket: str, key: str):
    s3_client.upload_file(local_path, bucket, key)


def main(iso3: str, issue_month: int):
    iso3 = iso3.lower()
    month_padded = str(issue_month).zfill(2)

    # Parse ops bucket/prefix from S3_OPS_DATA_PATH (s3://bucket/prefix)
    ops_parts = S3_OPS_DATA_PATH.removeprefix("s3://").split("/", 1)
    ops_bucket, ops_prefix = ops_parts[0], ops_parts[1]

    zarr_prefix = f"{ops_prefix}/{iso3}/zarr/{month_padded}/forecasts.zarr"
    zip_key = f"{S3_PUBLIC_PREFIX}/{iso3}/forecasts/{month_padded}/{ZIP_NAME}"
    tmp_path = f"/tmp/{iso3}/{month_padded}"

    s3 = boto3.client("s3")

    print(f"🔍 Checking if {ZIP_NAME} already exists at s3://{S3_PUBLIC_BUCKET}/{zip_key}...")
    if s3_key_exists(s3, S3_PUBLIC_BUCKET, zip_key):
        print(f"⚠️  WARNING: {ZIP_NAME} already exists. It will be overwritten.")

    print(f"📁 Creating local directory: {tmp_path}")
    os.makedirs(tmp_path, exist_ok=True)

    print(f"📦 Downloading zarr from s3://{ops_bucket}/{zarr_prefix}")
    s3_download_recursive(s3, ops_bucket, zarr_prefix, f"{tmp_path}/{ZARR_NAME}")

    print(f"🗜️ Zipping {ZARR_NAME} into {ZIP_NAME}")
    shutil.make_archive(
        base_name=f"{tmp_path}/forecasts",
        format="zip",
        root_dir=tmp_path,
        base_dir=ZARR_NAME,
    )

    print(f"🚀 Uploading {ZIP_NAME} to s3://{S3_PUBLIC_BUCKET}/{zip_key}")
    s3_upload_file(s3, f"{tmp_path}/{ZIP_NAME}", S3_PUBLIC_BUCKET, zip_key)

    print("🧹 Cleaning up local files")
    shutil.rmtree(tmp_path)
    print(f"✅ Done: {ZIP_NAME} uploaded to s3://{S3_PUBLIC_BUCKET}/{zip_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zip and upload forecasts Zarr to S3")
    parser.add_argument("iso3", type=str)
    parser.add_argument("issue_month", type=int)
    args = parser.parse_args()
    main(args.iso3, args.issue_month)