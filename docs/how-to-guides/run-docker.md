# How-To: Run in Docker

Build image:
```bash
docker build -t aa-runner .
```

Export AWS credentials:
```bash
export AWS_ACCESS_KEY_ID="XXX"
export AWS_SECRET_ACCESS_KEY="XXX"
export AWS_SESSION_TOKEN="XXX"
```

Run workflows:
```bash
docker run --rm -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} -e AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN} aa-runner:latest python -m AA.analytical <ISO> <SPI/DRYSPELL> --data-path <DATA_PATH> --output-path <OUTPUT_PATH>
```
