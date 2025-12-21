# Module: `AA.operational`

Generates operational outputs for a given country, **issue month**, and indicator.

## Usage

### Pixi
```bash
pixi run python -m AA.operational <ISO> <ISSUE_MONTH> <SPI/DRYSPELL>
```

### Docker
```bash
docker run --rm \
  -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
  -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
  -e AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN} \
  aa-runner:latest \
  python -m AA.operational <ISO> <ISSUE_MONTH> <SPI/DRYSPELL> \
  --data-path <DATA_PATH> --output-path <OUTPUT_PATH>
```

## Arguments

- `<ISO>`: 3-letter ISO code.
- `<ISSUE_MONTH>`: Issue month (e.g., `2025-02`).
- `<SPI/DRYSPELL>`: Indicator family.

## Inputs & Outputs

- Uses outputs from **Analytical** and/or **Triggers** stages.
- Writes operational products to configured output directory.

See [Configuration](configuration.md) and [Environments](environment.md) for paths and credentials.


:::AA.operational