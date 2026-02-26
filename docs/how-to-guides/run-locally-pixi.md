# How-To: Run Locally with Pixi

Install Pixi:
```bash
pixi install --locked
```

Run workflows:
```bash
pixi run python -m AA.cli.analytical <ISO> <SPI/DRYSPELL>
pixi run python -m AA.cli.triggers <ISO> <SPI/DRYSPELL> <VULNERABILITY>
pixi run python -m AA.cli.operational <ISO> <ISSUE_MONTH> <SPI/DRYSPELL>
```
