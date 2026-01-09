# anticipatory-action

[![Tests](https://github.com/WFP-VAM/anticipatory-action/actions/workflows/ci.yml/badge.svg)](https://github.com/WFP-VAM/anticipatory-action/actions/workflows/ci.yml)
[![Coverage](.github/badges/coverage.svg)](https://github.com/WFP-VAM/anticipatory-action/actions/workflows/ci.yml)

This repository provides a set of scripts and Jupytext notebooks to run analytical, trigger, and operational workflows for anticipatory action. You can run them either:

- in a local Pixi-managed environment, or

- inside a Docker container, using environment variables to pass parameters.


## 📚 Documentation

Full usage instructions, tutorials, and reference details are available in the documentation site:

[👉 View the Docs](https://wfp-vam.github.io/anticipatory-action/)


## Running

- If new features have been released on GitHub, you will have to pull the changes, run: `git pull origin main` in a shell or use the GitHub Desktop.

- Make sure your parameters are well defined in `config/{iso}_config.yml`.

- Make sure you are in the anticipatory-action folder: use the `cd` command to move to the anticipatory-action folder.


### ⚙️ Running in a local Pixi environment

To install [Pixi](https://pixi.sh/latest/getting_started/), follow the official instructions. Once installed, you can create the environment:

```bash
pixi install --locked
```

You can now run any of the scripts:

#### 1. Analytical script

```bash
pixi run python -m AA.analytical <ISO> <SPI/DRYSPELL>
```

#### 2. Triggers script

```bash
pixi run python -m AA.triggers <ISO> <SPI/DRYSPELL> <VULNERABILITY>
```

Where `<VULNERABILITY>` is one of:

- GT – General Triggers

- NRT – Non-Regret Triggers

- TBD – To Be Defined (for saving full list of triggers without filtering)

After running both SPI and Dryspell triggers, use the notebook `merge-spi-dryspell-gt-nrt-triggers.py` to filter and merge results.

#### 3. Operational script

```bash
pixi run python -m AA.operational <ISO> <ISSUE_MONTH> <SPI/DRYSPELL>
```

### 📓 Working with Jupytext Notebooks

This project uses [Jupytext](https://jupytext.readthedocs.io/en/latest/) to version notebooks as .py files. To open and run these:

1. Launch Jupyter Lab:

```bash
pixi run jupyter lab
```

2. Right-click on any .py notebook file and select "Open with → Notebook".

In the different folders of this repository, you will find different notebooks with the extension `.py`. These are jupytext files that can be run as notebooks. They facilitate version control and have a much smaller memory size.

3. Outputs will be saved in a paired .ipynb file. Be careful not to edit both simultaneously to avoid merge issues.

### 🛠️ Development Tips

#### Dependencies

To add a dependency:

```bash
pixi add package_name
```

To remove a dependency:

```bash
pixi remove package_name
```

#### Running

To run a script:

```bash
pixi run python -m path/to/script.py
```

#### Linting and formatting

Linting and formatting checks can be run by:

```pixi run lint-check```

To run checks and automatically fix issues (if they can be fixed automatically):

```pixi run lint```


#### Tests

Run the tests:

```pixi run test```

Run tests with coverage report and update the coverage badge:

```pixi run test-coverage```

This will:
- Run all tests
- Generate a coverage report in the terminal
- Update the coverage badge in `.github/badges/coverage.svg`


### 🐳 Running with Docker (locally)

Docker can be used to run workflows in a fully reproducible containerized environment.

#### 1. Build the Docker image

```bash
docker build -t aa-runner .
```
#### 2. Set up your AWS credentials

Before running the container, export the necessary environment variables:
```bash
export AWS_ACCESS_KEY_ID="XXX"
export AWS_SECRET_ACCESS_KEY="XXX"
export AWS_SESSION_TOKEN="XXX"
```

If you're using [Granted](https://docs.commonfate.io/granted/getting-started), you can assume your AWS profile using:
```bash
assume <your-profile>
```
This will export the necessary credentials as environment variables:

- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_SESSION_TOKEN

#### 3. Export the config parameters environment variable

You can provide the configuration via an environment variable instead of mounting a file. Convert your YAML config to JSON using `yq`:

```bash
export AA_CONFIG_JSON="$(pixi run yq '.' config/{iso}_config.yaml)"
```

Replace `{iso}` with the ISO3 country code (e.g., tza).
If `AA_CONFIG_JSON` is not set, the workflow will automatically fall back to reading *./config/{iso}_config.yaml* inside the container.


#### 4. Run the workflow in Docker

You can run any module via `docker run`, passing command-line arguments:

**Analytical**
```bash
docker run --rm \
  -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
  -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
  -e AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN} \
  -e AA_CONFIG_JSON="${AA_CONFIG_JSON}" \
  aa-runner:latest \
  python -m AA.analytical <ISO> <SPI/DRYSPELL> --data-path <DATA_PATH> --output-path <OUTPUT_PATH>
```

**Triggers**
```bash
docker run --rm \
  -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
  -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
  -e AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN} \
  -e AA_CONFIG_JSON="${AA_CONFIG_JSON}" \
  aa-runner:latest \
  python -m AA.triggers <ISO> <SPI/DRYSPELL> <VULNERABILITY> --data-path <DATA_PATH> --output-path <OUTPUT_PATH>
```

**Operational**
```bash
docker run --rm \
  -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
  -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
  -e AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN} \
  -e AA_CONFIG_JSON="${AA_CONFIG_JSON}" \
  aa-runner:latest \
  python -m AA.analytical <ISO> <ISSUE_MONTH> <SPI/DRYSPELL> --data-path <DATA_PATH> --output-path <OUTPUT_PATH>
```

## Set-up

**These steps must only be done the first time.**

You can do the set-up in JupyterHub and/or locally on your machine.

1. Set up SSH for your GitHub account

    - Follow the set up steps [here](https://github.com/WFP-VAM/ram-data-science-tools-docs/blob/main/docs/how-to/get-set-up-ssh-for-github.md)
    
2. Configuring GitHub with your name and email

    - Copy and paste the text below in the terminal

        `git config --global user.name "Your GitHub user name"`

    - Copy and paste the text below in the terminal

        `git config --global user.email YourGitAdress@example.com`

    And now you are good to go with GitHub as you like!
    
3. Clone the repository in your folder system

To start using the AA pipeline, you will have to import the repository from GitHub (you are lucky, you just set up SSH to access GitHub from this distant server!!).

Get back in the Terminal and run the commands below:

- `cd ~` to make sure you are in your home directory (or `cd YourWantedDirectory`)

- `git clone git@github.com:WFP-VAM/anticipatory-action.git` to clone anticipatory-action

Once it's done, you should see the anticipatory-action folder and all its files in in your system. 

4. Add your HDC credentials

You will need to set-up credentials for the WFP HDC STAC API.

To obtain the HDC STAC API token, go to [HDC token](https://auth.earthobservation.vam.wfp.org/stac-token.html) to generate a key. Then copy the key and get back in your files to create a `hdc_stac.tk` file in your home folder:
   - Create a Text File ".txt"
   - Paste your token as a first line
   - Save the file
   - Right click on this new file > click on **Rename** > rename the file to `hdc_stac.tk`

You are now done with setting up credentials for the HDC data.

**You are now good to run the AA workflow**


## Validation and ad-hoc evaluations

The `validation-nbs` folder contains the notebooks used to validate this pipeline in Python, based on reference results produced in R. In this folder you'll find several analyses for each stage of the workflow, showing the reliability of this new implementation, and comments explaining the sources of any discrepancies.

The purpose of the `template-eval-nbs` folder is to host notebook templates that will enable system performance to be re-evaluated in terms of ROC scores, coverage, and Hit Rate / Failure Rate; after running the AA scripts with different parameters or different datasets. 

Finally, we'll store in the `ad-hoc-evaluations` folder the results of specific evaluations obtained using the notebooks found in the `template-eval-nbs` folder. For instance, results related to the performance of the AA system using blended chirps or another type of forecast will be stored in this folder. 