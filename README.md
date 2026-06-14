# EMEL-NB: Evolutionary Multi-objective Ensemble Learning using Naive Bayes

This repository contains a clean, GitHub-ready copy of EMEL-NB, the Evolutionary Multi-objective Ensemble Learning approach using Naive Bayes from the research codebase. It contains a small demo wrapper for the example dataset `datasets/Musk1.csv`.

The method builds an ensemble classifier through iterative multi-objective feature selection. NSGA-II searches for feature subsets, base classifiers are trained on selected features, and a sparse logistic regression model integrates the selected classifier outputs.

## Repository Contents

```text
.
|-- README.md
|-- run_demo.py
|-- requirements.txt
|-- datasets
|   `-- Musk1.csv
|-- experiments
|   `-- experiments.py
|-- ensemble
|   `-- ensemble.py
|-- optimizer
|   |-- __init__.py
|   |-- ga.py
|   |-- nsga2.py
|   |-- sort.py
|   |-- sortc.py
|   `-- sort.dll
|-- eval
|   |-- eval_perform.py
|   |-- fit_evaluator.py
|   `-- fitsarray.py
`-- util
    |-- preprocessingdata.py
    `-- util.py
```

`run_demo.py` is the GitHub-facing entry point for running the proposed method on `Musk1.csv` with either cross-validation or hold-out validation. It does not depend on `run_exp_new.py`.

## Environment

The original experiments used Python 3.11. A minimal package set is provided in `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you use Conda, create or activate an environment for the project before installing dependencies:

```bash
conda create -n emel-nb python=3.11
conda activate emel-nb
pip install -r requirements.txt
```

Then run the demo with the active environment's Python interpreter:

```bash
python run_demo.py --help
```

`optimizer/sort.dll` is included for the Windows fast non-dominated sorting path. If the DLL cannot be loaded, the code falls back to the Python sorting implementation.

## Dataset Location

Place the example dataset here:

```text
datasets/Musk1.csv
```

The included demo commands use the dataset stem `Musk1`; the script resolves it to `datasets/Musk1.csv`.

## Manuscript Defaults

`run_demo.py` uses the proposed-method defaults inferred from `run_experiment_0_2b.bat` and the original project runner:

```text
ensem_data_type = train_cv_prob
boost_update_type = New
rep_times = 30
save_info_option = 0134
step = 0.5
n_iter = 5
ga_pop_size = 30
ga_n_iter = 100
ga_c_rate = 0.9
ga_m_rate = -1  # uses 1 / num_features
ifbamu = True
random_state = 1123
base classifier = GaussianNB
```

These settings can take a long time because they reproduce the manuscript-scale configuration. For a quick smoke test, reduce `--rep-times`, `--n-iter`, and `--ga-n-iter` from the command line.

## Run 10-Fold Cross-Validation

From the repository root:

```bash
python run_demo.py --validation-mode cv
```

This uses 10 folds by default. To change the number of folds:

```bash
python run_demo.py --validation-mode cv --folds 5
```

## Run Hold-Out Validation

```bash
python run_demo.py --validation-mode holdout
```

The default hold-out test fraction is `0.2`. To change it:

```bash
python run_demo.py --validation-mode holdout --test-size 0.3
```

## Optional Lightweight Smoke Run

This command keeps the same pipeline but reduces repetition and optimizer settings so imports, data loading, splitting, and result writing can be checked quickly:

```bash
python run_demo.py --validation-mode holdout --rep-times 1 --n-iter 1 --ga-pop-size 4 --ga-n-iter 1 --save-info-option none
```

## Outputs

By default, outputs are written under:

```text
results/
```

Expected output includes:

- Console progress messages showing the dataset, validation mode, key parameter settings, and optimizer iterations.
- An aggregated CSV file named like `<timestamp>_<save_info>_Musk1.csv`.
- When `--save-info-option` is not `none`, detailed run folders under `<timestamp>_<save_info>_Musk1/s<seed>/r<rep>_c<fold>/` containing files such as `params.csv`, `iter_info.csv`, `iter_performance.csv`, and `ensemble_data.csv` according to the selected save option.

## Notes on Code Preservation

The core implementation scripts were copied without modifying the proposed method logic. New or edited files are limited to release-facing files such as `run_demo.py`, `README.md`, `requirements.txt`, and `.gitignore`.

The original project directory was treated as read-only while preparing this GitHub-ready version. No original project files were modified, deleted, renamed, reformatted, or overwritten.
