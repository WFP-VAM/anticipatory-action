# anticipatory-action


## Validation and ad-hoc evaluations

The `validation-nbs` folder contains the notebooks used to validate this pipeline in Python, based on reference results produced in R. In this folder you'll find several analyses for each stage of the workflow, showing the reliability of this new implementation, and comments explaining the sources of any discrepancies.

The purpose of the `template-eval-nbs` folder is to host notebook templates that will enable system performance to be re-evaluated in terms of ROC scores, coverage, and Hit Rate / Failure Rate; after running the AA scripts with different parameters or different datasets. 

Finally, we'll store in the `ad-hoc-evaluations` folder the results of specific evaluations obtained using the notebooks found in the `template-eval-nbs` folder. For instance, results related to the performance of the AA system using blended chirps or another type of forecast will be stored in this folder. 


## Running

- If new features have been released on GitHub, you will have to pull the changes, run: `git pull origin main` in a shell or use the GitHub Desktop.

- Make sure your environment is active:

```commandline
$ conda activate aa-env
```

- Make sure your parameters are well defined in `config/params.py`.

- Make sure your forecasts and chirps folders are up to date in `data/<country-iso>/forecast` and `data/<country-iso>/chirps` (tmp)

- Make sure your country shapefile is present in the folder `data/<country-iso>/shapefiles/`.

- Make sure you are in the anticipatory-action folder: use the `cd` command to move to the anticipatory-action folder.


### How to run jupytext files 

In the different folders of this repository, you will find different notebooks with the extension `.py`. These are jupytext files that can be run as notebooks. They facilitate version control and have a much smaller memory size. They are simple to use, after activating the environment: 

1. Open jupyter lab by executing the command `jupyter lab` in the terminal. 
2. Find the file you need to work on in the file explorer.
3. Right-click on it and select `Open with -> Notebook`. You can now execute your notebook cell by cell.

An `.ipynb` file will immediately be created, storing the cell outputs. As these two files are paired, any changes you make to one will immediately be reflected in the other, and vice versa. You can also work directly on the `.ipynb` file when you return to your workspace. Be careful, however, not to modify both files at the same time, as this may create conflicts and errors when you save or reopen them. 


### Full workflow through the script

You can now run the workflow for a given country.

**Analytical script**

```commandline
$ python analytical.py <ISO> <SPI/DRYSPELL>
```

**Triggers script**

```commandline
$ python triggers.py <ISO> <SPI/DRYSPELL>
```

After running this script for SPI / DRYSPELL and General / Non-Regret Triggers you can use the `merge-spi-dryspell-gt-nrt-triggers.py` notebook to filter the triggers for each district regarding the selected vulnerability and merge spi and dryspell. It actually provides the very final output. 

**Operational script**

```commandline
$ python operational.py <ISO> <issue-month> <SPI/DRYSPELL>
```


### Check outputs

In `data/outputs/FbF_Pilot_MockUp/`, the final outputs that will serve as input for the Tableau dashboard are stored in a csv format. It is possible to open these in excel to inspect the results.


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

4. Create conda environment specific to the workflows

Before using `anticipatory-action`, you have to create the anaconda environment specific to the workflow containing all the python libraries needed. To do so, run the commands below:
- use the `cd` command to go in the anticipatory-action folder (ex: `cd anticipatory-action`)
- `conda env create -f aa_env.yml` to create the AA environment (`conda env create -p /envs/user/aa-env -f aa-env.yml` if you are in JupyterHub)
- `conda activate aa-env`
- `python -m ipykernel install --user --name aa-env`

You can now activate your environment: `conda activate aa-env`

Make sure it is active before running any workflow.

5. Add your HDC credentials

You will need to set-up credentials for the WFP HDC STAC API.

To obtain the HDC STAC API token, go to [HDC token](https://auth.earthobservation.vam.wfp.org/stac-token.html) to generate a key. Then copy the key and get back in your files to create a `hdc_stac.tk` file in your home folder:
   - Create a Text File ".txt"
   - Paste your token as a first line
   - Save the file
   - Right click on this new file > click on **Rename** > rename the file to `hdc_stac.tk`

You are now done with setting up credentials for the HDC data.

**You are now good to run the AA workflow**