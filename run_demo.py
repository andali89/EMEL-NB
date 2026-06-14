"""Clean demo entry point for the proposed ensemble method.

This wrapper keeps the copied core implementation unchanged and provides a
small command-line interface for running the example dataset with either
10-fold cross-validation or hold-out validation.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from ctypes import cdll
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from ensemble.ensemble import EnsembleClassifier
from experiments.experiments import Experiment
from optimizer.nsga2 import Nsga2
from optimizer.sortc import NonDomiSort
from util.preprocessingdata import read_data


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = "Musk1"


# Defaults inferred from run_experiment_0_2b.bat and run_exp_new.py.
MANUSCRIPT_DEFAULTS = {
    "n_iter": 5,
    "percentage": 1,
    "step": 0.5,
    "ensem_re_type": "Best",
    "ensem_data_type": "train_cv_prob",
    "ini_weight_type": None,
    "balance_train_data": True,
    "boost_update_type": "New",
    "save_info_option": "0134",
    "ga_pop_size": 30,
    "ga_n_iter": 100,
    "ga_c_rate": 0.9,
    "ga_m_rate": -1,
    "ifbamu": True,
    "n_fold": 5,
    "if_weight_test": True,
    "solver": "liblinear",
    "penalty": "l1",
    "random_state": 1123,
    "rep_times": 30,
    "method_name": "EnsembleEA",
    "base_classifier": GaussianNB,
    "ini_method": None,
    "mi_lam": 0.8,
    "mi_sigma": 10,
}


def str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def build_parameters(args: argparse.Namespace) -> dict:
    parameters = dict(MANUSCRIPT_DEFAULTS)

    parameters.update(
        {
            "exp_type": "cross_validation" if args.validation_mode == "cv" else "holdout",
            "test_p_splits": args.folds if args.validation_mode == "cv" else args.test_size,
            "rep_times": args.rep_times,
            "n_iter": args.n_iter,
            "step": args.step,
            "ga_pop_size": args.ga_pop_size,
            "ga_n_iter": args.ga_n_iter,
            "ga_c_rate": args.ga_c_rate,
            "ga_m_rate": args.ga_m_rate,
            "ifbamu": args.ifbamu,
            "if_weight_test": args.if_weight_test,
            "save_info_option": None if args.save_info_option.lower() == "none" else args.save_info_option,
            "random_state": args.random_state,
        }
    )

    dll_path = REPO_ROOT / "optimizer" / "sort.dll"
    parameters["dll"] = str(dll_path) if dll_path.exists() else None
    return parameters


def load_and_preprocess_data(data_name: str):
    datasets_dir = REPO_ROOT / "datasets"
    if not datasets_dir.is_dir():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")

    matches = [
        path for path in datasets_dir.iterdir()
        if path.is_file() and path.name.lower().startswith(data_name.lower())
    ]
    if not matches:
        raise FileNotFoundError(f"No dataset files found for {data_name} in {datasets_dir}")

    preferred = None
    for ext in (".csv", ".mat", ".arff"):
        for match in matches:
            if match.suffix.lower() == ext:
                preferred = match
                break
        if preferred is not None:
            break
    if preferred is None:
        preferred = sorted(matches)[0]

    try:
        if preferred.suffix.lower() == ".csv":
            data = pd.read_csv(preferred)
        elif preferred.suffix.lower() in {".mat", ".arff"}:
            data = read_data(str(preferred))
        else:
            data = pd.read_csv(preferred)
    except Exception as exc:
        raise RuntimeError(f"Failed to read dataset '{preferred}': {exc}") from exc

    feat_cols = list(data.columns[:-1])
    label_col = data.columns[-1]

    if np.any(data.isnull()):
        imputer = KNNImputer()
        data = pd.DataFrame(
            data=imputer.fit_transform(data),
            index=data.index,
            columns=data.columns,
        )

    num_attri = data.shape[1]
    y_labels = np.unique(data.iloc[:, num_attri - 1].to_numpy())
    if len(y_labels) != 2:
        raise ValueError(
            f"Target variable in {data_name} must have exactly 2 unique values, "
            f"found {len(y_labels)}: {y_labels}"
        )

    idx_zero = data.iloc[:, num_attri - 1] == y_labels[0]
    idx_one = data.iloc[:, num_attri - 1] == y_labels[1]
    label_name = data.columns[num_attri - 1]
    if np.sum(idx_zero) < np.sum(idx_one):
        data.loc[idx_zero, label_name] = 1
        data.loc[idx_one, label_name] = -1
    else:
        data.loc[idx_one, label_name] = 1
        data.loc[idx_zero, label_name] = -1

    print(f"num of negative instances {np.sum(data.iloc[:, num_attri - 1] == -1)}")
    print(f"num of positive instances {np.sum(data.iloc[:, num_attri - 1] == 1)}")

    scaler = StandardScaler()
    data_X = scaler.fit_transform(data.iloc[:, :num_attri - 1])
    data_Y = data.iloc[:, num_attri - 1].to_numpy(dtype=int)
    return data_X, data_Y, feat_cols, label_col


def create_ensemble_classifier(num_features: int, parameters: dict, random_state: int):
    if parameters["dll"] is None:
        nonsort = None
    else:
        dllpath = parameters["dll"]
        if not os.path.exists(dllpath):
            raise FileNotFoundError(f"Sort DLL not found at {dllpath}")
        try:
            mydll = cdll.LoadLibrary(dllpath)
            nonsort = NonDomiSort(mydll=mydll)
        except Exception:
            nonsort = None
            print("Warning: Could not load sort.dll")

    clf = parameters["base_classifier"]()
    optimizer = Nsga2(
        var_length=num_features,
        pop_size=parameters["ga_pop_size"],
        c_rate=parameters["ga_c_rate"],
        m_rate=(1 / num_features) if (
            "ga_m_rate" not in parameters
            or parameters["ga_m_rate"] is None
            or parameters["ga_m_rate"] <= 0
        ) else parameters["ga_m_rate"],
        n_iters=parameters["ga_n_iter"],
        sort_operator=nonsort.non_domi_sort if nonsort else None,
        random_state=random_state,
        no_prd=False,
        if_ba_mu=parameters["ifbamu"],
    )

    en_clf = EnsembleClassifier(
        clf,
        optimizer,
        iter_num=parameters["n_iter"],
        percentage=parameters["percentage"],
        random_state=parameters["random_state"],
        iter_save_path=None,
        step=parameters["step"],
        solver=parameters["solver"],
        penalty=parameters["penalty"],
        n_fold=parameters["n_fold"],
        obj_type="double",
        ensem_data_type=parameters["ensem_data_type"],
        ensem_re_type=parameters["ensem_re_type"],
        ini_weight_type=parameters["ini_weight_type"],
        balance_train_data=parameters["balance_train_data"],
        boost_update_type=parameters["boost_update_type"],
        save_info_option=parameters["save_info_option"],
        ini_method=parameters.get("ini_method"),
        mi_lam=parameters.get("mi_lam", 0.8),
        mi_sigma=parameters.get("mi_sigma", 10),
    )
    en_clf.evaluator.set_weight_test(parameters["if_weight_test"])
    return en_clf


def run_proposed_experiment(
    data_name: str,
    split_seed: int,
    parameters: dict,
    save_folder: str,
    save_id: str,
) -> None:
    print(f"Running experiment: {data_name}, seed: {split_seed}")
    data_X, data_Y, _, _ = load_and_preprocess_data(data_name)
    num_features = data_X.shape[1]
    r_state = parameters["random_state"]

    experiment = Experiment(
        data_X=data_X,
        data_y=data_Y,
        classifier=create_ensemble_classifier(num_features, parameters, r_state),
        exp_type=parameters["exp_type"],
        parameters=parameters["test_p_splits"],
        split_seed=split_seed,
        method_name=parameters["method_name"],
        data_name=data_name,
        save_folder=save_folder,
        save_id=save_id,
        repetitions=parameters["rep_times"],
        rep_update_func=lambda rep: create_ensemble_classifier(
            num_features, parameters, r_state + rep
        ),
    )
    experiment.run_experiment()
    print(f"Completed experiment: {data_name}, seed: {split_seed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the proposed ensemble method on datasets/Musk1.csv."
    )
    parser.add_argument(
        "--validation-mode",
        choices=["cv", "holdout"],
        default="cv",
        help="Use 10-fold cross-validation ('cv') or hold-out validation ('holdout').",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset file stem under datasets/.")
    parser.add_argument("--seed", type=int, default=40, help="Data split seed.")
    parser.add_argument("--folds", type=int, default=10, help="Number of CV folds when --validation-mode cv.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out test fraction when --validation-mode holdout.")
    parser.add_argument("--save-dir", default=str(REPO_ROOT / "results"), help="Directory for result CSV files.")
    parser.add_argument("--save-info", default=None, help="Optional suffix for output file names.")

    parser.add_argument("--rep-times", type=int, default=MANUSCRIPT_DEFAULTS["rep_times"])
    parser.add_argument("--n-iter", type=int, default=MANUSCRIPT_DEFAULTS["n_iter"])
    parser.add_argument("--step", type=float, default=MANUSCRIPT_DEFAULTS["step"])
    parser.add_argument("--ga-pop-size", type=int, default=MANUSCRIPT_DEFAULTS["ga_pop_size"])
    parser.add_argument("--ga-n-iter", type=int, default=MANUSCRIPT_DEFAULTS["ga_n_iter"])
    parser.add_argument("--ga-c-rate", type=float, default=MANUSCRIPT_DEFAULTS["ga_c_rate"])
    parser.add_argument("--ga-m-rate", type=float, default=MANUSCRIPT_DEFAULTS["ga_m_rate"])
    parser.add_argument("--ifbamu", type=str_to_bool, default=MANUSCRIPT_DEFAULTS["ifbamu"])
    parser.add_argument("--if-weight-test", type=str_to_bool, default=MANUSCRIPT_DEFAULTS["if_weight_test"])
    parser.add_argument("--save-info-option", default=MANUSCRIPT_DEFAULTS["save_info_option"])
    parser.add_argument("--random-state", type=int, default=MANUSCRIPT_DEFAULTS["random_state"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.chdir(REPO_ROOT)
    os.environ.setdefault("COMPUTERNAME", platform.node() or "unknown")

    parameters = build_parameters(args)
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "10fold_cv" if args.validation_mode == "cv" else "holdout"
    save_info = args.save_info or (
        f"{parameters['ensem_data_type']}_step05_IT{parameters['n_iter']}_"
        f"SD_{args.seed}_{mode_tag}"
    )
    save_id = f"{int(time.time())}_{save_info}"

    print("Running proposed ensemble method")
    print(f"Dataset: datasets/{args.dataset}.csv")
    print(f"Validation mode: {args.validation_mode}")
    print(f"Output directory: {save_dir}")
    print("Key settings:")
    for key in (
        "ensem_data_type",
        "boost_update_type",
        "rep_times",
        "n_iter",
        "step",
        "ga_pop_size",
        "ga_n_iter",
        "save_info_option",
    ):
        print(f"  {key}: {parameters[key]}")

    run_proposed_experiment(
        data_name=args.dataset,
        split_seed=args.seed,
        parameters=parameters,
        save_folder=str(save_dir),
        save_id=save_id,
    )


if __name__ == "__main__":
    main()
