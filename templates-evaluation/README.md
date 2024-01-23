# anticipatory-action/template-eval-nbs

The purpose of this folder is to host jupytext notebook templates that will enable system performance to be re-evaluated in terms of ROC scores, coverage, and Hit Rate / Failure Rate, after running the AA scripts with different parameters or different datasets. Once the full analytical / triggers pipeline has been rerun on a specific dataset or with specific parameters that are different to the original ones, these notebooks can be copied and pasted within a dedicated folder in `ad-hoc-evaluations/`.

Given the quite distinct purposes of these different notebooks, here is a brief description of each of them:

- `compare_analytical.py` compares ROC scores at the pixel level (for different indexes / issue months)  and at the district level after running the analytical script with a different forecast / observation dataset or with different parameters. Of course, it requires having run `analytical.py` in advance. 

- `compare_triggers.py` compares triggers coverage (by district / category / window)​ and actual outcomes of a system that relies on a different observations / forecasts dataset or different parameters. It gives a clear idea of the associated consequences in terms of Hit Rate and Failure Rate which can be interpreted as Recall and Precision. Of course, it requires having run `analytical.py` and `triggers.py` in advance. 

The following ones are more specific and thus can also be used for exploratory / analytical purposes. They are not mandatory when conducting a full evaluation of a new system. 

- `evaluate-roc-scores.py` allows to compute the forecast skill (roc scores) with given forecasts and observations for a specific issue month and a specific index. It can then be used to compare parameters or datasets at a more granular level, or for instance to try a different downscaling / bias correction method.

- `inspect-operational-steps.py` allows to inspect each step of the operational workflow and play with the data or parameters. It can be used as a tutorial for the core methodology of the system but also to check probability values for a specific index and issue month. 