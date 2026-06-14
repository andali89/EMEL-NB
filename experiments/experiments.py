import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold


class Experiment:
    def __init__(
            self,
            data_X: np.ndarray,
            data_y: np.ndarray,
            classifier: str,
            fit_params: dict = None,
            exp_type: str = "holdout",
            # proportion of data for holdout or number of folds for cross-validation
            parameters: float = 0.2,
            split_seed: int = 42,
            method_name: str = "Classifier",
            data_name: str = "Dataset",
            save_folder: str = "Experiment",
            save_id: str = "experiment_1",
            repetitions: int = 1,
            rep_update_func: callable = None,
            extra_meta: dict = None,
    ):
        """
            Initializes the experiment with the provided dataset, classifier, and configuration parameters.
            
            Args:
                data_X (np.ndarray): Feature matrix of the dataset.
                data_y (np.ndarray): Target labels of the dataset.
                classifier (str): Name of the classifier to be used.
                fit_params (dict, optional): Parameters for fitting the classifier. Defaults to None.
                exp_type (str, optional): Type of experiment ('holdout' or 'cross_validation'). Defaults to "holdout".
                parameters (float, optional): Proportion of data for holdout or number of folds for cross-validation. Defaults to 0.2.
                split_seed (int, optional): Random seed for data splitting. Defaults to 42.
                method_name (str, optional): Name of the classification method. Defaults to "Classifier".
                data_name (str, optional): Name of the dataset. Defaults to "Dataset".
                save_folder (str, optional): Folder to save experiment results. Defaults to "Experiment".
                save_id (str, optional): Identifier for saving experiment results. Defaults to "experiment_1".
                repetitions (int, optional): Number of experiment repetitions. Defaults to 1.
                rep_update_func (callable, optional): Function to update parameters between repetitions. Defaults to None.
                extra_meta (dict, optional): Extra columns to append to every result row. Each key becomes
                    a column name; each value should be a single-element list. Used for scale experiment
                    metadata (scale_V, scale_H, scale_seed). Defaults to None.
        """

        self.data_X = data_X
        self.data_y = data_y
        self.classifier = classifier
        self.fit_params = fit_params if fit_params is not None else {}
        self.exp_type = exp_type
        self.parameters = parameters
        self.split_seed = split_seed
        self.method_name = method_name
        self.save_folder = save_folder
        self.repetitions = repetitions
        self.data_name = data_name
        self.save_id = save_id
        self.rep_update_func = rep_update_func
        self.extra_meta = extra_meta
        

    def run_experiment(self):
        if self.exp_type == "holdout":
            return self.holdout_experiment()
        elif self.exp_type == "cross_validation":
            return self.cross_validation_experiment()
        else:
            raise ValueError(
                "Invalid experiment type. Choose 'holdout' or 'cross_validation'.")

    def get_meta_save_info(self, cv_fold: int, rep: int):
        """Set additional parameters information."""
        addition_columns = [
            "Folder",
            "Write_time",
            "Data",
            "Computer_name",
            "num_training_samples",
            "num_testing_samples",
            "num_features",
            "Experiment_type",
            "Experiment_parameters",
            "Seed",
            "CV_fold",  # current fold
            "Repetition",
            "Method",            
        ]
        self.folder_detail = os.path.join(
            self.save_folder,
            f"{self.save_id}_{self.data_name}",
            f"s{self.split_seed}",
            f"r{rep}_c{cv_fold}"
        )

        # do not include save_folder in sub_folder_detail
        # to avoid redundancy in the path
        sub_folder_detail = os.path.join(
            f"{self.save_id}_{self.data_name}",
            f"s{self.split_seed}",
            f"r{rep}_c{cv_fold}"
        )

        addition_info = {
            "Method": [self.method_name],
            "Folder": [sub_folder_detail],
            "Data": [self.data_name],            
            "Seed": [self.split_seed],
            "Experiment_type": [self.exp_type],
            "Experiment_parameters": [self.parameters],
            "CV_fold": [cv_fold],  # current fold
            "Repetition": [rep],
            "num_training_samples": [self.X_train.shape[0]],
            "num_testing_samples": [self.X_test.shape[0]],
            "num_features": [self.X_train.shape[1]],
            "Write_time": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Computer_name": [os.environ['COMPUTERNAME']]
        }
        meta_info = pd.DataFrame(addition_info, columns=addition_columns)

        # Append any extra metadata columns (e.g. scale_V, scale_H, scale_seed)
        if self.extra_meta is not None:
            for col, val in self.extra_meta.items():
                meta_info[col] = val

        return meta_info

    def _update_classifier_if_needed(self, rep: int):
        """Update classifier if repetition > 0 and update function is provided."""
        if rep > 0 and self.rep_update_func:
            self.classifier = self.rep_update_func(rep)

    def _train_and_predict(self):
        """Train the classifier and make predictions."""
        # Train the model
        self.classifier.fit(self.X_train, self.y_train, **self.fit_params)

        # Make predictions
        predict_results = self.classifier.predict_save(
            self.X_test, self.y_test, path_name=None)

        predict_results.reset_index(drop=True, inplace=True)
        return predict_results

    def _get_save_path(self):
        """Get the CSV save path for results."""
        return os.path.join(self.save_folder, f"{self.save_id}_{self.data_name}.csv")

    def _save_results_to_csv(self, run_results: pd.DataFrame):
        """Save results to CSV file with appropriate header handling."""
        save_path = self._get_save_path()

        # Check if file exists and set header accordingly
        if not os.path.exists(save_path):
            os.makedirs(self.save_folder, exist_ok=True)
            c_header = True
        else:            
            c_header = False

        # Save results
        run_results.to_csv(save_path, mode="a+", header=c_header, index=False)

    def _save_classifier_params(self):
        """Save classifier parameters and data to detailed folder structure."""
        

        self.classifier.save_params_info(
            save_path=self.folder_detail,
            data=[self.X_train, self.y_train, self.X_test, self.y_test]
        )

    def _process_single_fold(self, cv_fold, rep: int):
        """Process a single fold: train, predict, and save results."""
        # Train and predict
        predict_results = self._train_and_predict()

        # Get meta info and combine with predictions
        # For holdout, use -1 in meta_info but "x" in folder structure
        meta_cv_fold = -1 if cv_fold == "x" else cv_fold
        meta_info = self.get_meta_save_info(cv_fold=meta_cv_fold, rep=rep)
        run_results = pd.concat((meta_info, predict_results), axis=1)

        # Save results
        self._save_results_to_csv(run_results)
        self._save_classifier_params()

    def holdout_experiment(self):
        """Run holdout validation experiment."""
        for rep in range(self.repetitions):
            self._update_classifier_if_needed(rep)

            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.data_X, self.data_y,
                test_size=self.parameters,
                random_state=self.split_seed,
                stratify=self.data_y
            )

            # Process this fold (using "x" as cv_fold for holdout, -1 for meta_info)
            self._process_single_fold(cv_fold="x", rep=rep)

    def cross_validation_experiment(self):
        """Run cross-validation experiment."""
        # Initialize cross-validation
        cv = StratifiedKFold(
            n_splits=int(self.parameters),
            random_state=self.split_seed,
            shuffle=True
        )

        for rep in range(self.repetitions):
            self._update_classifier_if_needed(rep)

            for fold, (train_index, test_index) in enumerate(cv.split(self.data_X, self.data_y)):
                # Split the data for current fold
                self.X_train, self.X_test = self.data_X[train_index], self.data_X[test_index]
                self.y_train, self.y_test = self.data_y[train_index], self.data_y[test_index]

                # Process this fold
                self._process_single_fold(cv_fold=fold, rep=rep)
    
    
