# Monitoring Process: Running Operational Forecasts

This guide explains how national meteorological services can generate seasonal forecast outputs using the **Anticipatory Action pipeline**.

---

## 1️⃣ Run Operational Script

Generate forecast outputs using the operational script:

```bash
pixi run python -m AA.cli.operational <ISO> <ISSUE_MONTH> <SPI/DRYSPELL>
```

### Notes:
- `<ISO>` should be one of: `MOZ`, `ZWE`, or `MWI`.
- `<ISSUE_MONTH>` is the month number (e.g., `9` for September, `11` for November).
- **MOZ** requires both `SPI` and `DRYSPELL` runs.
- **ZWE** and **MWI** require only `SPI`.

This step automatically merges the current issue month probabilities with the triggers. The output is saved in the configured data folder (`S3_OPS_DATA_PATH`).

---

## After Running

Once the script completes:
- The outputs will be available in your local environment or configured S3 path.
- These outputs can be used for your internal analysis and dashboard updates.

> For dashboard integration or advanced packaging, contact WFP support. Those steps are handled internally and are **not required for running the forecasts**.

