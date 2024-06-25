# Anticipatory Action Python set-up


### Table of contents

* [1. Clone the repository from GitHub](#chapter1)
* [2. Create an environment with Anaconda](#chapter2)
* [3. Download the data](#chapter3)
* [4. Run the Anticipatory Action workflow](#chapter4)


## 1. Clone the repository from GitHub <a class="anchor" id="chapter1"></a>

Start by installing **GitHub desktop** on your machine using this [link](https://desktop.github.com/).

Then open GitHub desktop and follow this [tutorial](https://docs.github.com/en/desktop/overview/getting-started-with-github-desktop#part-1-installing-and-authenticating) to authenticate. This requires having a GitHub account, so please create an account if you don't have one yet. 

Once this is done, you can follow these steps [here](https://docs.github.com/en/desktop/adding-and-cloning-repositories/cloning-a-repository-from-github-to-github-desktop) to clone the [anticipatory action repository](https://github.com/WFP-VAM/anticipatory-action/).


## 2. Create an environment with Anaconda <a class="anchor" id="chapter2"></a>

You need to install the required Python libraries in an environment to run this package. To do this, we will use Anaconda, which can be downloaded freely [here](https://www.anaconda.com/download/success).

Once you have it installed, open the **Anaconda PowerShell** prompt and use `cd` (change directory) and `ls` (list files and directories) commands to go to the folder where you have downloaded the **aa-env** repository.

For example, run the following command if the anticipatory-action repo has been cloned in the Documents folder: 

`cd Documents/anticipatory-action` 

Once you are in the anticipatory-action folder and are able to see the `aa-env.yml` file when doing `ls`, please create a new environment named `aa-env` that will contain all the required packages:

`conda env create -f aa-env.yml`

Now, activate the environment:

`conda activate aa-env`

To confirm that you have successfully activated the environment, your command line prompt should begin with *aa-env*.

Make sure it is active before running any workflow.


## 3. Download the data <a class="anchor" id="chapter3"></a>

Please start downloading the data from [that link](https://data.earthobservation.vam.wfp.org/public-share/aa/zwe.zip) and save it within the **data** folder of the *anticipatory-action* repo. 

Once that is done, make sure you have the following structure within your filesystem:

``` 
anticipatory-action
├── data
│   ├── iso
│   │   ├── auc
│   │   │   ├── split_by_issue
│   │   ├── probs
│   │   ├── triggers
│   │   ├── zarr
│   │   │   ├── 2022
│   │   │   │   ├── 01
│   │   │   │   ├── 02
│   │   │   │   ├── 05
│   │   │   │   ├── 06
│   │   │   │   ├── 07
│   │   │   │   ├── 08
│   │   │   │   ├── 09
│   │   │   │   ├── 10
│   │   │   │   ├── 11
│   │   │   │   ├── 12
│   │   │   │   ├── obs
```


## 4. Run the Anticipatory Action workflow

The training contains two main scripts:

* `run-full-verification.py` contains the verification step with the computation of the roc scores and the triggers step with the selection of the optimal triggers. This one should be run only once, and in general before the beginning of the monitoring. 

* `run-operational-monitoring.py` contains the operational steps needed to process the forecasts received each month. This should be run each month to derive the probabilities, and these probabilities are then merged with the triggers to check the alerts. 

If you want to work on these notebooks, please open the **Anaconda PowerShell** prompt and run the following commands:

`cd <path_to_AA_folder>` 

`conda activate aa-env`

`jupyter lab`

Once the jupyter lab window is open, please click-right on the notebook you want to open, select *Open with* > *Jupyter Notebook*. 

