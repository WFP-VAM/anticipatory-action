# Module: `AA.analytical`

Runs the analytical stage for a given country (`<ISO>`) and indicator (`SPI` or `DRYSPELL`).

## Usage

### Pixi
```bash
pixi run python -m AA.analytical <ISO> <SPI/DRYSPELL>
```

### Docker
```bash
docker run --rm \
  -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
  -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
  -e AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN} \
  aa-runner:latest \
  python -m AA.analytical <ISO> <SPI/DRYSPELL> \
  --data-path <DATA_PATH> --output-path <OUTPUT_PATH>
```

## Arguments

- `<ISO>`: 3-letter ISO code (e.g., `KEN`, `ZMB`).
- `<SPI/DRYSPELL>`: Which analytical track to run.

## Configuration

Reads from `config/{iso}_config.yml`. See [Configuration](configuration.md).

## Notes

- Make sure HDC credentials are configured. See [HDC Credentials](../how-to-guides/hdc-credentials.md).
- Data paths can be overridden via `--data-path` and `--output-path` in Docker.


:::AA.analytical