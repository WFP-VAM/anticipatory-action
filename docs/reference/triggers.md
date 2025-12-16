# Module: `AA.triggers`

Computes triggers for SPI or Dryspell, for a given country and **vulnerability** mode.

## Usage

### Pixi
```bash
pixi run python -m AA.triggers <ISO> <SPI/DRYSPELL> <VULNERABILITY>
```

### Docker
```bash
docker run --rm \
  -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
  -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
  -e AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN} \
  aa-runner:latest \
  python -m AA.triggers <ISO> <SPI/DRYSPELL> <VULNERABILITY> \
  --data-path <DATA_PATH> --output-path <OUTPUT_PATH>
```

## Arguments

- `<ISO>`: 3-letter ISO code.
- `<SPI/DRYSPELL>`: Indicator family.
- `<VULNERABILITY>`:
  - `GT` — General Triggers
  - `NRT` — Non-Regret Triggers
  - `TBD` — Save full list without filtering

## Post-processing

After running both SPI and Dryspell triggers, use:
- `merge-spi-dryspell-gt-nrt-triggers.py` (Jupytext notebook) to **filter and merge** results.

> For core data models and common utilities used in trigger computation, see **HIP Analysis** docs:  
> https://wfp-vam.github.io/hip-analysis/


:::AA._triggers_ready_set