# anticipatory-action/validation-nbs

The purpose of this folder is to host the notebooks involved in the certification of this implementation of the AA system which has been carried out using the Mozambique use case. As part of a transition from R to Python, various analyses have been carried out to ensure the validity and accuracy of this code. These can be found in the following notebooks, a brief description of which follows:

- `check-operational.ipynb`: shows differences included at each step of the operational workflow (anomaly, bias correction, probability) at the pixel level for different accumulation periods and forecasts issued in October​

- `check-analytical.ipynb`: compares R & Python ROC scores at the pixel level (for different indexes / issue months)  and at the district level and details the sources of differences for the different examples

- `check-coverage.ipynb`: compares triggers coverage (by district / category / window)​

- `validation-outcomes.ipynb`: compares actual outcomes of a system transition based on May-Oct 2023 forecasts, hence it gives a clear idea of the concrete consequences of running the AA workflow in Python​

​