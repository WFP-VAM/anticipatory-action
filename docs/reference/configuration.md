# Configuration: `config/{iso}_config.yaml`

Each country has a YAML configuration file that defines workflow parameters for **analytical**, **trigger**, and **operational** scripts. These files are read by the `Params` class in `AA/params.py`.

---

## How It Works

- The configuration file is loaded by:
  ```python
  config = load_config(iso)  # reads ./config/{iso}_config.yaml
  ```
- All keys in the YAML are mapped to attributes in `Params`.
- Defaults (e.g., monitoring year, calibration year, paths) are applied if keys are missing.
- Additional logic:
  - `aggregate` is set based on `index` (`spi` or `dryspell`).
  - `districts` are inferred from `districts_vulnerability`.
  - `tolerance` and `requirements` are converted to typed dictionaries for performance.
  - `windows` define indicator periods for each index family.

---

## Main Sections in YAML

- **Identification**
  - `iso`: Country code (e.g., `moz`, `zwe`).
  - `index`: Indicator family (`spi` or `dryspell`).

- **Season & Monitoring**
  - `issue_months`: Months for trigger selection.
  - `monitoring_year`, `calibration_year`, `start_monitoring`.
  - `start_season`, `end_season`.

- **Vulnerability & Skill**
  - `vulnerability`: `GT`, `NRT`, or `TBD`.
  - `districts_vulnerability`: Map of district → vulnerability class.
  - `general_t` and `non_regret_t`: Skill requirements for GT/NRT.
  - `tolerance`: Thresholds for false-alarm tolerance.

- **Windows**
  - `windows`: Nested structure defining indicator periods for each index.
    Example:
    ```yaml
    windows:
      spi:
        window1: ["ON", "NDJ"]
        window2: ["DJF", "MAM"]
      dryspell:
        window1: ["NDJ"]
        window2: ["DJF"]
    ```

- **Paths**
  - `data_path` and `output_path`: Where to read/write data.
  - Defaults to `s3://wfp-ops-userdata/amine.barkaoui/aa` if not provided.

---

## Example Structure

```yaml
iso: moz
index: spi
issue_months: [9, 10, 11]
monitoring_year: 2024
calibration_year: 2022
start_monitoring: 5

districts_vulnerability:
  Chokwe: GT
  Guija: GT
  Mandlakazi: NRT

general_t:
  hit_rate: 0.6
  success_rate: 0.55

non_regret_t:
  hit_rate: 0.5
  success_rate: 0.5

tolerance:
  mild: 0.05
  moderate: 0.10

windows:
  spi:
    window1: ["ON", "NDJ"]
    window2: ["DJF", "MAM"]
  dryspell:
    window1: ["NDJ"]
    window2: ["DJF"]

data_path: s3://wfp-ops-userdata/amine.barkaoui/aa
output_path: s3://wfp-ops-userdata/amine.barkaoui/aa
```

---

## Key Points

- Keep YAML minimal but complete for required sections.
- Avoid hardcoding defaults in YAML unless overriding.
- Validate changes against `Params` logic in `params.py`.
- Document any country-specific deviations inline in the YAML.
