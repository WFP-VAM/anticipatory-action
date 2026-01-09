# Anticipatory Action


This repository provides scripts and Jupytext notebooks to run **analytical**, **trigger**, and **operational** workflows for anticipatory action. You can run them:

- In a **local Pixi-managed environment**, or  
- Inside a **Docker container**, using environment variables to pass parameters.

> 🔗 For core objects used here, inherited datasets, and key imported functions, see the **HIP Analysis** [documentation](https://wfp-vam.github.io/hip-analysis/).

---

## ✅ Quick Start

### 1. Pull the latest code
```bash
git pull origin main
```

### 2. Install Pixi
Follow [Pixi installation guide](https://pixi.sh/latest/getting_started/), then:
```bash
pixi install --locked
```

### 3. Run workflows locally
- **Analytical**
  ```bash
  pixi run python -m AA.analytical <ISO> <SPI/DRYSPELL>
  ```
- **Triggers**
  ```bash
  pixi run python -m AA.triggers <ISO> <SPI/DRYSPELL> <VULNERABILITY>
  ```
- **Operational**
  ```bash
  pixi run python -m AA.operational <ISO> <ISSUE_MONTH> <SPI/DRYSPELL>
  ```

> Make sure your parameters are defined in `config/{iso}_config.yml`.

---

## 🐳 Run with Docker (short version)

Build the image:
```bash
docker build -t aa-runner .
```

Export AWS credentials:
```bash
export AWS_ACCESS_KEY_ID="XXX"
export AWS_SECRET_ACCESS_KEY="XXX"
export AWS_SESSION_TOKEN="XXX"
```

Run a module:
```bash
docker run --rm \
  -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
  -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
  -e AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN} \
  aa-runner:latest \
  python -m AA.analytical <ISO> <SPI/DRYSPELL> \
  --data-path <DATA_PATH> --output-path <OUTPUT_PATH>
```

For full details, see [Run with Docker](how-to-guides/run-docker.md).

---

## 📚 Documentation Sections

- **[How-To Guides](how-to-guides/run-locally-pixi.md)** — Pixi, Docker, SSH/GitHub, credentials, dependencies, tests, Jupytext.
- **[Explanation](explanation/overview.md)** — What the pipeline does and how it’s validated.

---

## 🔗 See Also

- **[HIP Analysis Docs](https://wfp-vam.github.io/hip-analysis/) (core objects, functions & datasets)**  
