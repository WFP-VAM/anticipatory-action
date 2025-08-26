#!/bin/bash

# Check for 2 arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <iso3> <issue_month>"
  exit 1
fi

# Input parameters
ISO3_RAW="$1"
MONTH_RAW="$2"

# Normalize inputs
ISO3=$(echo "$ISO3_RAW" | tr '[:upper:]' '[:lower:]')
MONTH_PADDED=$(printf "%02d" "$MONTH_RAW")

# Define filenames
ZIP_NAME="forecasts.zip"
ZARR_NAME="forecasts.zarr"

# Define paths
PATH1="s3://wfp-ops-userdata/amine.barkaoui/aa/data/${ISO3}/zarr/2022/${MONTH_PADDED}/forecasts.zarr"
PATH2="/tmp/${ISO3}/${MONTH_PADDED}"
PATH3="s3://wfp-ops-userdata/public-share/aa/forecasts/${ISO3}/${MONTH_PADDED}"

echo "🔍 Checking if $ZIP_NAME already exists in $PATH3..."
if aws s3 ls "$PATH3/$ZIP_NAME" > /dev/null 2>&1; then
  echo "⚠️  WARNING: $ZIP_NAME already exists in $PATH3. It will be overwritten."
fi

echo "📁 Creating local directory: $PATH2"
mkdir -p "$PATH2"

echo "📦 Copying Zarr file from $PATH1 to $PATH2"
aws s3 cp --recursive "$PATH1" "$PATH2/$ZARR_NAME"

echo "🗜️ Zipping $ZARR_NAME into $ZIP_NAME"
cd "$PATH2"
zip -r "$ZIP_NAME" "$ZARR_NAME"

echo "🚀 Uploading $ZIP_NAME to $PATH3"
aws s3 cp "$ZIP_NAME" "$PATH3/$ZIP_NAME"

echo "🧹 Cleaning up local files"
rm -rf "$PATH2/$ZARR_NAME" "$PATH2/$ZIP_NAME"

echo "✅ Done: $ZIP_NAME uploaded to $PATH3"
echo "🌐  Accessible at: https://data.earthobservation.vam.wfp.org/public-share/aa/forecasts/${ISO3}/${MONTH_PADDED}/forecasts.zip"
