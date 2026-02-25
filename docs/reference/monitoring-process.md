# Monitoring Process Workflow

This document outlines the step-by-step process for generating, packaging, and sharing seasonal forecast outputs with national meteorological services (Mozambique, Zimbabwe, and Malawi). It includes operational script execution, data merging, packaging, sharing, and dashboard updates.

---

## 1️⃣ Run Operational Script

Generate forecast outputs using the operational script.

```bash
pixi run python -m AA.cli.operational <ISO> <ISSUE_MONTH> <SPI/DRYSPELL>
```

***Notes:***

- <ISO> should be MOZ, ZWE, or MWI

- <ISSUE_MONTH> is in integer format (e.g. 9 or 11)

- MOZ requires both SPI and DRYSPELL runs

- ZWE and MWI require only SPI

This step automatically merges the current issue month probabilities with the triggers. The output is saved in the data folder in `S3_OPS_DATA_PATH` which is the workspace dedicated to the drought AA work.

---
## 2️⃣ Zip Forecasts for Distribution

Compress the forecast files to prepare them for sharing.

```bash
bash AA/helpers/zip-forecasts.sh <ISO> <ISSUE_MONTH>
```

***Notes:***

- This script zips the forecast outputs and stores them in the public-share folder of the S3 bucket

- Ensure the script has access to the correct paths and permissions

---
## 3️⃣ Share Forecasts via Email

Send the zipped forecasts to the national meteorological services.

***Notes:***

- Use the standard email template

- Include direct links to the zipped files in the public-share folder

- Be aware of caching issues in the forecasts folder. Overwriting existing files may not reflect immediately due to caching. Coordinate with the Infrastructure team to invalidate cache if necessary.

---
## 4️⃣ Await Feedback from Met Services

Wait for confirmation that each Met Service has successfully run the script and generated outputs.

***Notes:***

- This feedback is essential before updating the Prism dashboard

- If issues arise, follow up with the respective country teams. In general, problems arise when some files are moved or renamed (especially the probs/aa_probabilities_triggers_pilots.csv file that the script automatically reads to add new probabilities). One thing to confirm first is the time coordinate of the forecasts they are using to make sure they use the dataset that has just been shared.

---
## 5️⃣ Run Prism Update Script

Once confirmation is received, update the dashboard using the Prism script.

```bash
pixi run python -m AA.helpers.prism <ISO> <ISSUE_MONTH>
```

***Notes:***
- This script generates the dashboard data for the specified country and issue month

- Ensure the operational outputs are finalized before running this step

- Only the staging version of the dashboard is being updated here.

---
## 6️⃣ Verify Staging Dashboard

Check the staging version of the Prism dashboard to confirm everything looks correct.

***Notes:***
- Open the operational dashboard URL and append `?staging=true`

- Confirm the current issue month appears on the timeline

- Status indicators should be visible for each pilot district after selecting the current issue month


---
## 7️⃣ Promote Staging File to Operational

Once the staging dashboard is verified, promote the final CSV file to the operational folder.

```bash
aws s3 cp s3://wfp-ops-userdata/public-share/aa/staging/aa_probabilities_triggers_<iso>.csv \
s3://wfp-ops-userdata/public-share/aa/aa_probabilities_triggers_<iso>.csv
```

***Notes:***
- Replace <iso> with the appropriate country code

- This makes the updated probabilities and triggers publicly accessible