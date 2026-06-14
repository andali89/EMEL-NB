import numpy as np
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import eval.fit_evaluator as fit_evaluator
import optimizer.nsga2 as nsga2
import eval.fitsarray as fitsarray
from eval.eval_perform import FsPerformance
import util.util as util
from scipy import stats
import time
import pandas as pd
from eval.fit_evaluator import EvaluatorWeight
import os


class EnsembleClassifier(object):
    def __init__(
        self,
        # Base classifier (e.g., RandomForest, SVM)
        clf,
        # NSGA-II optimizer for multi-objective feature selection
        optimizer: nsga2.Nsga2,
        # Evaluator that computes fitness values with sample weights
        evaluator: fit_evaluator.EvaluatorWeight = None,
        # Number of boosting-like iterations to build ensemble
        iter_num=1,
        # Fraction of Pareto-optimal solutions to retain (1=all, 0.5=half)
        percentage=1,
        random_state=None,                      # Random seed for reproducible results
        # Directory path to save iteration results as CSV
        iter_save_path=None,
        # Regularization type for meta-classifier (l1=sparse, l2=ridge)
        penalty="l1",
        solver="liblinear",                     # Optimization algorithm for meta-classifier
        # Number of cross-validation folds for feature evaluation
        n_fold=5,
        # Objective function type (e.g., "double" for 2 objectives)
        obj_type="double",
        # Data type for meta-classifier (e.g., "train", "train_cv", "train_prob", "train_cv_prob")
        ensem_data_type="train_cv_prob",
        # Type of ensemble feature selection results to use: "All" (All solutions used to build meta-classifier) or "Best" (Non-dominated solutions used to build meta-classifier)
        ensem_re_type="All",
        # Step size for sample weight updates (higher = more focus on misclassified samples)
        step=0.5,
        # Initial sample weight type: "None" (decided by weight  input in fit) or "balanced"
        ini_weight_type=None,
        # Whether to balance the training data when fitting the ensemble classifier
        balance_train_data: bool = False,
        # type of boost update weight
        boost_update_type="New",  # "Old" or "New"                 
        save_info_option="all",  # "iter_optimizer", "iter_info", "iter_performance"
        #  parameters_save: save the parameters of the ensemble classifier
        parameters_save=None,
        # MI-guided population initialization (optional)
        ini_method=None,        # None (default random) or "mi" (MI-guided)
        mi_lam=0.8,             # λ: upper bound for initialization probability
        mi_sigma=10,            # σ: sharpness of the sigmoid scaling
    ) -> None:
        """
        Initialize the EnsembleClassifier.

        Args:
            clf: Base classifier (must support fit, predict, predict_proba).
            optimizer: NSGA-II optimizer for multi-objective feature selection.
            evaluator: Evaluator for fitness computation with sample weights.
            iter_num: Number of boosting-like iterations for ensemble diversity.
            percentage: Fraction of Pareto-optimal solutions to retain (1=all, <1=subset).
            random_state: Random seed for building logistic regression model.
            iter_save_path: Directory to save iteration results as CSV files.
            penalty: Regularization penalty for meta-classifier ("l1", "l2", "elasticnet").
            solver: Solver for meta-classifier ("liblinear", "saga", "lbfgs").
            n_fold: Number of cross-validation folds for feature evaluation.
            obj_type: Objective function type (e.g., "double" for accuracy+size).
            ensem_data_type: Data type for meta-classifier:
                - "train": Use training predictions.
                - "train_cv": Use cross-validation predictions.
                - "train_prob": Use prediction probabilities.
                - "train_cv_prob": Use cross-validation prediction probabilities.
            ensem_re_type: Type of ensemble feature selection results ("All" or "Best").
                - "All": Use all solutions for meta-classifier.
                - "Best": Use only non-dominated solutions for meta-classifier.
            step: Step size for sample weight updates (higher = more focus on misclassified samples).
            ini_weight_type: Initial sample weight type ("balanced" or None for uniform weights).
            balance_train_data: Whether to balance training data using up-sampling.
            boost_update_type: Sample weight update method ("Old" or "New").
            save_info_option: Information to save during training. (default:"All")For example: "12" for saving iter_info and iter_optimizer. 
                - 0-ensemble_data,
                - 1-iter_info,
                - 2-iter_optimizer,
                - 3-iter_performance,
                - 4-params
                - "all" for all components.
                - None for no saving.
            parameters_save: Dictionary to save ensemble classifier parameters.
            ini_method: Population initialization method. None (default) uses uniform
                random initialization. "mi" enables MI-guided initialization where
                each feature's initialization probability is derived from its mutual
                information with the class label.
            mi_lam: Upper bound λ for the MI-guided initialization probability (default 0.8).
            mi_sigma: Sharpness σ of the sigmoid scaling in the probability formula (default 10).
        """

        self.clf = clf
        self.optimizer = optimizer

        self.iter_num = iter_num
        self.percentage = percentage
        self.random_state = random_state
        self.save_folder = iter_save_path
        self.sort_times = []
        self.step = step
        self.penalty = penalty
        self.solver = solver
        self.ifprob = None
        self.ensem_data_type = ensem_data_type
        self.set_data_type(self.ensem_data_type)
        self.ensem_re_type = ensem_re_type
        self.ini_weight_type = ini_weight_type
        
        self.balance_train_data = balance_train_data
        self.balance_train_data = balance_train_data
        self.boost_update_type = boost_update_type
        self.save_info_option = save_info_option
        if evaluator is not None:
            self.evaluator = evaluator
        else:
            self.evaluator = EvaluatorWeight(clf,
                                             n_fold=n_fold, obj_type=obj_type)

        if iter_save_path is None:
            self.save_folder = ""
        elif not (iter_save_path[-1] == "/" or iter_save_path[-1] == "\\"):
            self.save_folder = iter_save_path + "/"
        else:
            self.save_folder = iter_save_path

        if self.percentage < 1 and self.percentage > 0:
            num = np.round(optimizer.pop_size * self.percentage)
            self.keep_idx = np.arange(num, dtype=int)

        self.parameters_save = parameters_save
        self.ini_method = ini_method
        self.mi_lam = mi_lam
        self.mi_sigma = mi_sigma

    def set_data_type(self, data_type="train"):
        """set the data type for building ensemble classifier


        Args:
            data_type (str, optional): ["train", "train_cv", "train_prob","train_cv_prob]. Defaults to "train".

        """
        self.data_type = data_type
        if data_type == "train_cv":
            self.get_data = self.get_data_train_cv
            self.ifprob = False
        elif data_type == "train_prob":
            self.get_data = self.get_data_train_prob
            self.ifprob = True
        elif data_type == "train_cv_prob":
            self.get_data = self.get_data_train_cv_prob
            self.ifprob = True
        else:
            self.get_data = self.get_data_train
            self.ifprob = False

    def get_data_type(self):
        return self.data_type

    def _single_run(self, sample_weight, iter):
        """One single run with given sample weight with GA.
        """
        self.sample_weights.append(sample_weight)
        self.evaluator.set_sample_weight(sample_weight)
        self.optimizer.set_evaluator(self.evaluator)
        self.optimizer.run()
        best_re = self.optimizer.get_best_results()
        final_re = self.optimizer.get_final_results()
        self.final_results.append(final_re)
        self.iter_results.append(self.optimizer.save_iter_results())
        self.optimizer_times.append(self.optimizer.get_run_time())

        if self.percentage >= 1:
            final_re_ = final_re
        else:
            if self.percentage == -1:
                keep_idx = (final_re[2][:, 0] == 1)
            else:
                keep_idx = self.keep_idx
            final_re_ = []
            final_re_.append(final_re[0][keep_idx, :])
            final_re_.append(final_re[1].sort(keep_idx))
            final_re_.append(final_re[2][keep_idx, :])

        return best_re, final_re_

    def fit(self, train_X, train_y, initial_w=None, regression_w=None):
        """build the ensamble classifer
        """
        
        # MI-guided initialization: compute once, reuse across all iterations
        if self.ini_method and self.ini_method.lower() == "mi":
            print("Computing MI-guided initialization probabilities...")
            from sklearn.feature_selection import mutual_info_classif
            mi = mutual_info_classif(
                train_X, train_y, random_state=self.random_state)
            ini_probs = self._compute_mi_ini_probs(
                mi, lam=self.mi_lam, sigma=self.mi_sigma)
            self.optimizer.set_ini_probs(ini_probs)
            print(f"MI-guided initialization enabled (lam={self.mi_lam}, sigma={self.mi_sigma}).")
        else:
            ini_probs = None

        if self.balance_train_data:
            train_X, train_y = up_sampling(train_X, train_y)
            print("Training data is balanced by up-sampling.")

        if initial_w is None:
            if self.ini_weight_type == "balanced":
                initial_w = np.ones((train_X.shape[0],))
                num_pos = np.sum(train_y == 1)
                num_neg = np.sum(train_y == -1)
                initial_w[train_y == 1] = num_neg / num_pos
                print("Initial sample weight is set to balanced weights.")
            if self.ini_weight_type is None:
                initial_w = np.ones((train_X.shape[0],))
                print("Initial sample weight is set to ones.")

            else:
                raise ValueError(
                    "Invalid initial weight type, choose 'balanced' or None.")
        else:
            print("Initial sample weight is set to user input.")

        self.evaluator.set_data(data_X=train_X, data_y=train_y)
        self.run_time = time.perf_counter()
        self.optimizer_times = []
        self.final_results = []
        self.iter_results = []
        self.sample_weights = []
        self._run_fs(train_X, train_y, initial_w)
        self.get_data()
        self._build_ensemble(self.prd_data, train_y, regression_w)
        self.run_time = time.perf_counter() - self.run_time
        return self

    def _build_ensemble(self, data, train_y, regression_w=None):
        self.rg = build_ensemble(data, train_y,
                                 regression_w, self.penalty, self.solver, self.random_state)

    @staticmethod
    def _compute_mi_ini_probs(w, lam=0.8, sigma=10):
        """Compute per-feature initialization probabilities from MI values.

        Parameters
        ----------
        w : array-like of shape (n_features,)
            Mutual information values for each feature.
        lam : float
            Upper bound λ for the initialization probability (default 0.8).
        sigma : float
            Sharpness σ of the sigmoid scaling (default 10).

        Returns
        -------
        p : ndarray of shape (n_features,)
            Probability that each feature bit is initialized to 1.
        """
        w = np.asarray(w, dtype=float)
        w_min, w_max = w.min(), w.max()
        if w_max == w_min:
            return np.full(len(w), lam / 2.0)
        z = 2.0 * sigma * (w - w_min) / (w_max - w_min) - sigma
        return lam / (1.0 + np.exp(-z))

    def _run_fs(self, train_X, train_y, initial_w):
        self.clfs = []
        self._train_pred_set = []
        self._train_probs_set = []
        self._final_re_set = []
        w = initial_w
        sols = []

        

        for i in range(self.iter_num):
            best_re, final_re = self._single_run(w, i)
            ensem_re = self._select_ensem_re(final_re, best_re)
            clfs_, train_pred_, train_probs_ = self._train_clfs(
                self.clf,
                train_X,
                train_y,
                ensem_re[0],
                sample_weight=w,
                ifprob=self.ifprob
            )

            # self._final_re_set.append(final_re)
            # self.clfs.extend(clfs_)
            # self._train_pred_set.append(train_pred_)
            # self._train_probs_set.append(train_probs_)
            # w = self._update_weight(w, final_re[1])
            # sols.append(final_re[0])

            self._final_re_set.append(ensem_re)
            self.clfs.extend(clfs_)
            self._train_pred_set.append(train_pred_)
            self._train_probs_set.append(train_probs_)
            w = self._update_weight(w, ensem_re[1])
            sols.append(ensem_re[0])

        self.sols = np.vstack(sols)

    def _select_ensem_re(self, fs_results, best_results):
        """Select the features selection results used to build the final ensamble
           based on the specified criteria.

        Parameters
        ----------
            fs_results : list
                The feature selection results from the optimizer.

        Returns
        -------
            en_results: list
                the selected feature selection results used to build the final 
                ensamble classifier.
        """
        if self.ensem_re_type == "All":
            return fs_results
        elif self.ensem_re_type == "Best":
            return best_results

    def get_data_train(self):
        """get the data for building ensemble model

        Returns
        -------
            prd_data: matix
                the data records the predicted results of each training sample 
                on the classifier built by the train set.
        """
        prd_data = []
        for train_pred_ in self._train_pred_set:
            prd_data_ = np.column_stack(train_pred_)
            prd_data.append(prd_data_)
        self.prd_data = np.hstack(prd_data)
        self.ifprob = False
        return self.prd_data

    def get_data_train_prob(self):
        """get the data for building ensemble model

        Returns
        -------
            prd_data: matix
                the data records the predicted PROBABILITY results of each 
                training sample on the classifier built by the train set.
        """
        prd_data = []
        for train_probs_ in self._train_probs_set:
            prob_data = self.prob_to_data(train_probs_)
            prd_data_ = np.column_stack(prob_data)
            prd_data.append(prd_data_)

        self.prd_data = np.hstack(prd_data)
        self.ifprob = True
        return self.prd_data

    def get_data_train_cv(self):
        """get the data for building ensemble model

        Returns
        -------
            prd_data: matix
                the data records the predicted results of each 
                training sample during the inner K-fold CV in FS process
        """
        prd_data = []
        for final_re in self._final_re_set:
            prd_data_ = self._get_data2(final_re[1])
            prd_data.append(prd_data_)
        self.prd_data = np.hstack(prd_data)
        self.ifprob = False
        return self.prd_data

    def get_data_train_cv_prob(self):
        """get the data for building ensemble model

        Returns
        -------
            prd_data: matix
                the data records the predicted PROBABILITY results of each 
                training sample during the inner K-fold CV in FS process
        """
        prd_data = []
        for final_re in self._final_re_set:
            prd_data_ = self._get_data3(final_re[1])
            prd_data.append(prd_data_)
        self.prd_data = np.hstack(prd_data)
        self.ifprob = True
        return self.prd_data

    def _get_data(self, train_X, train_y, initial_w):
        """previous codes for generating the data to build ensemble model
        do not suggested to used.

        Args:
            train_X (_type_): _description_
            train_y (_type_): _description_
            initial_w (_type_): _description_
        """
        clfs = []
        prd_data = []
        prd_data2 = []
        prd_data3 = []
        prd_data4 = []
        sols = []
        w = initial_w
        for i in range(self.iter_num):
            _, final_re = self._single_run(w, i)
            clfs_, train_pred_, train_probs_ = self._train_clfs(
                self.clf,
                train_X,
                train_y,
                final_re[0],
                sample_weight=w
            )
            sols.append(final_re[0])
            clfs.extend(clfs_)
            prd_data_ = np.column_stack(train_pred_)
            prd_data.append(prd_data_)
            prob_data = self.prob_to_data(train_probs_)
            prd_data4_ = np.column_stack(prob_data)
            prd_data4.append(prd_data4_)
            prd_data2_ = self._get_data2(final_re[1])
            prd_data3_ = self._get_data3(final_re[1])
            prd_data2.append(prd_data2_)
            prd_data3.append(prd_data3_)
            w = self._update_weight(w, final_re[1])

        self.prd_data = np.hstack(prd_data)
        self.prd_data4 = np.hstack(prd_data4)
        self.prd_data2 = np.hstack(prd_data2)
        self.prd_data3 = np.hstack(prd_data3)
        self.sols = np.vstack(sols)
        self.clfs = clfs

    def predict(self, test_X, ifprob=None):
        if ifprob is None:
            ifprob = self.ifprob
        return self._predict_sample(test_X, self.clfs, self.sols, self.rg, ifprob=ifprob)

    def prob_to_data(self, probs):
        prob_data = []
        for prob in probs:
            prob_data.append(-1 * prob[:, 0] + 1 * prob[:, 1])
        return prob_data

    def predict_save(self, test_X, test_y, path_name=None, ifprob=None):

        if ifprob is None:
            ifprob = self.ifprob
        start_time = time.perf_counter()
        predict_test_y = self.predict(test_X, ifprob=ifprob)
        test_time = time.perf_counter() - start_time
        re, columns = util.get_clf_score(test_y, predict_test_y)
        model_coeffi = self.rg.coef_
        run_time = self.get_run_time()
        # Create a DataFrame to save the data
        data_dict = {
            "run_time": [run_time],
            "test_time": [test_time],
        }

        # Add performance metrics
        for i, col in enumerate(columns):
            data_dict[col] = [re[i]]

        data_dict["sample_predict"] = [",".join(map(str, predict_test_y))]

        # Add model coefficients
        for i in range(model_coeffi.shape[1]):
            data_dict[f"rg_w_{i+1}"] = [model_coeffi[0, i]]

        df = pd.DataFrame(data_dict)

        if path_name is not None:
            df.to_csv(path_name, index=False)

        return df

    def get_params(self):
        params = {
            "ensemble_iter_num": self.iter_num,
            "ensemble_step": self.step,
            "ensemble_percentage": self.percentage,
            "ensemble_random_state": self.random_state,
            "ensemble_evaluator": self.evaluator.__class__,
            "ensemble_penalty": self.penalty,
            "ensemble_solver": self.solver,
            "ensemble_optimizer": self.optimizer.__class__,
            "ensemble_data_type": self.data_type,
            "ensemble_ensem_re_type": self.ensem_re_type,
            "ensemble_ini_weight_type": self.ini_weight_type,
            "ensemble_balance_train_data": self.balance_train_data,
            "ensemble_boost_update_type": self.boost_update_type,
            "ensemble_ini_method": self.ini_method,
            "ensemble_mi_lam": self.mi_lam,
            "ensemble_mi_sigma": self.mi_sigma,
        }
        params.update(self.optimizer.get_params())
        params.update(self.evaluator.get_params())
        return params

    def save_params_info(self, save_path=None, data=None):
        """save the parameters of the ensemble classifier
        """
        if self.save_info_option is None:
            print("No save info option provided, using default '01234'.")
            return
        if self.save_info_option.lower() == "all":
            save_info_option = "01234"        
        else:
            save_info_option = self.save_info_option
            
        if self.parameters_save is None:
            parameters_ = {}
        else:
            parameters_ = self.parameters_save
        parameters_.update(self.get_params())
        save_params = pd.Series(parameters_)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if "0" in save_info_option:
            self.save_metabuild_solutions(save_path=os.path.join(
                save_path, "ensemble_data.csv"))
        if "1" in save_info_option:
            self.save_iter_info(save_path=os.path.join(
                save_path, "iter_info.csv"))
        if "2" in save_info_option:
            self.save_iter_optimizer(save_path=os.path.join(
                save_path, "iter_optimizer.csv"))
        if "3" in save_info_option:
            self.save_iter_performance(
                train_X=data[0],
                train_y=data[1],
                test_X=data[2],
                test_y=data[3],
                save_path=os.path.join(save_path, "iter_performance.csv")
            )
        if "4" in save_info_option:
            save_params.to_csv(os.path.join(save_path, "params.csv"))

        

    def get_run_time(self):
        return self.run_time

    def save_metabuild_solutions(self, save_path=None):
        """Save the solutions used to build the ensemble classifier.

        Parameters
        ----------
        save_path : str, optional
            The path to save the solutions as a CSV file.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the solutions used to build the ensemble classifier.
        """

        # Collect data from all iterations
        all_data = []
        for idx, ensem_re in enumerate(self._final_re_set):
            idx_ = np.full((ensem_re[0].shape[0], 1), idx, dtype=int)
            # Each solution's info for this iteration
            df = pd.DataFrame(
                np.hstack([idx_, ensem_re[2], ensem_re[1][:, :], ensem_re[0]]),
                columns=["iteration", "level", "crowding_distance"] +
                [f"f{i}" for i in range(ensem_re[1][:, :].shape[1])] +
                [f"x{i}" for i in range(ensem_re[0].shape[1])]
            )
            all_data.append(df)
        tosave = pd.concat(all_data, axis=0, ignore_index=True)
        tosave.reset_index(inplace=True)
        if save_path is not None:
            tosave.to_csv(save_path, index=False)
        return tosave

    def save_iter_performance(self, train_X, train_y, test_X, test_y, save_path=None):
        eval_per = FsPerformance(
            self.clf,
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y
        )
        pd_data = []
        for i, final_re in enumerate(self.final_results):
            pd_data_ = eval_per.evaluate_save(final_re[0], final_re[1][:], final_re[2],
                                              save_solutions=True, use_multi=False)
            iter_info = pd.DataFrame(
                [i] * final_re[0].shape[0], columns=["iter"])

            pd_data.append(pd.concat((iter_info, pd_data_), axis=1))

        pd_data = pd.concat(pd_data, axis=0)
        if save_path is not None:
            pd_data.to_csv(save_path)
        return pd_data

    def save_iter_optimizer(self, save_path=None):
        pd_data = []
        for i, info in enumerate(self.iter_results):
            info.reset_index(drop=True, inplace=True)
            iter_info = pd.DataFrame([i] * info.shape[0], columns=["iter"])
            pd_data.append(pd.concat((iter_info, info), axis=1))
        pd_data = pd.concat(pd_data, axis=0)
        if save_path is not None:
            pd_data.to_csv(save_path)
        return pd_data

    def save_iter_info(self, save_path=None):
        weights = np.vstack(self.sample_weights)
        iter_times = np.array(self.optimizer_times)
        iter_times = iter_times.reshape((iter_times.shape[-1], 1))
        iter = np.arange(self.iter_num)
        iter = iter.reshape((iter.shape[-1], 1))
        return util.save_pddata(
            (iter, iter_times, weights),
            (("iter",), ("run_time",), "$w"),
            path_name=save_path
        )

    def _update_weight(self, weight, pre_re: fitsarray.FitsArray):
        """Update the weight of samples
        """
        if weight is None:
            return None
        # each element in predic_correct is a list with 3 elements [array0, array1,array2]
        # where array0 is the predicted labels, array1 is bool array denotes if predicted 
        # correctly (True) or wrong (False), array2 is a array stores probablity values (to be further verified)  
        pred_set = [x[1] for x in pre_re.predic_correct]
        pre_re_ = np.vstack(pred_set)
        p_correct = np.sum(pre_re_, axis=0) / pre_re_.shape[0]
        if self.boost_update_type == "Old":
            n_weight = self._update_sample_weight_org(
                weight, p_correct, step=self.step)
        elif self.boost_update_type == "New":
            n_weight = self._update_sample_weight_new(
                weight, p_correct, step=self.step)
        return n_weight

    def _update_sample_weight_org(
        self,
        weights,
        pre_re,
        step=0.5
    ):
        """Update the sample weight of each instance.

        Parameters
        ----------
        weitghts : array-like of shape \
                    (n_samples, )
                Vector containing the current weight of each sample.

        pre_re : array-like of shape (n_samples,)
                The prediction performance (e.g., accuracy rate) for each sample.
        step : double
                The step define the extent to change the weight. 

        Returns
        -------
        updated_w : array-like of shape (n_samples,)
                The updated weight of each sample.

        """
        updated_w = weights * np.exp((1 - pre_re) * step)
        updated_w = updated_w / np.sum(updated_w)
        return updated_w

    def _update_sample_weight_new(self, weights, pre_re, step=0.5):
        """Update the sample weight of each instance.

        Parameters
        ----------
        weitghts : array-like of shape \
                    (n_samples, )
                Vector containing the current weight of each sample.

        pre_re : array-like of shape (n_samples,)
                The prediction performance (e.g., accuracy rate) for each sample.
        step : double
                The step define the extent to change the weight. 

        Returns
        -------
        updated_w : array-like of shape (n_samples,)
                The updated weight of each sample.

        """
        # calculate the overall error of the ensemble classifier
        n_pre_re = np.sum(pre_re * weights) / np.sum(weights)
        error = 1 - np.mean(n_pre_re)

        # calculate the weight for adjustment
        if error == 0:
            return weights
        alpha = step * np.log((1 - error) / error)

        # 1 for correct predictions, -1 for incorrect predictions
        pred_sign = self._get_pred_sign(pre_re)

        # Element-wise multiplication and exponentiation
        updated_w = weights * np.exp(-alpha * pred_sign)
        updated_w = updated_w / np.sum(updated_w)
        return updated_w

    def _get_pred_sign(self, pre_re):
        """Get the sign of predictions for each sample.
        cluster the pre_re into two groups: correct and incorrect predictions.
        according to the value of pre_re, which is the prediction performance
        (e.g., accuracy rate) for each sample.

        Parameters
        ----------
        pre_re : array-like of shape (n_samples,)
            The prediction performance (e.g., accuracy rate) for each sample.

        Returns
        -------
        pred_sign : array-like of shape (n_samples,)
            The sign of predictions: 1 for correct predictions, -1 for incorrect predictions.
        """
        data = pre_re.reshape(-1, 1)
        # Use K-Means to find two clusters
        kmeans = KMeans(
            n_clusters=2, random_state=self.random_state, n_init='auto').fit(data)

        # Identify which cluster center is 'correct' (has a higher value)
        cluster_centers = kmeans.cluster_centers_.flatten()
        correct_cluster_label = np.argmax(cluster_centers)

        # Assign 1 to the 'correct' cluster and -1 to the 'incorrect' one
        pred_sign = np.where(
            kmeans.labels_ == correct_cluster_label, 1.0, -1.0)
        return pred_sign

    def _get_data2(self, pre_re: fitsarray.FitsArray):
        pred_set = [x[0] for x in pre_re.predic_correct]
        data = np.column_stack(pred_set)
        return data

    def _get_data3(self, pre_re: fitsarray.FitsArray):
        pred_set = [x[2] for x in pre_re.predic_correct]
        scores_ = []
        for prob in pred_set:
            score = -1 * prob[:, 0] + 1 * prob[:, 1]
            scores_.append(score)
        data = np.column_stack(scores_)
        return data

    def _train_clfs(self, clf, X, y, sols, sample_weight=None, ifprob=True):
        clfs = []
        clfs_re = []
        clfs_re_prob = []
        for sol in sols:
            X_ = X[:, sol == 1]
            clf_ = clone(clf)
            clf_.fit(X_, y, sample_weight=sample_weight)
            re_ = clf_.predict(X_)
            clfs_re.append(re_)
            clfs.append(clf_)
            if ifprob:
                clfs_re_prob.append(clf_.predict_proba(X_))
        return clfs, clfs_re, clfs_re_prob

    def _predict_sample(self, data, clfs, sols, rg, ifprob=False):
        """Predict samples in data with the ensemble classifier.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The input data (e.g. test set) for prediction
        clfs : tuple of base classifiers
            The base classifiers
        sols : array-like of shape (n_solutions, n_features)
            The input solutions where each solution denotes \
                    a feature subset.
        rg : the model to intergrate the base classifiers

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted y value for each input sample.
        """
        idx = np.nonzero(rg.coef_[0])[0]
        clfs_re = np.zeros((data.shape[0], sols.shape[0]))
        for i in idx:
            X_ = data[:, sols[i, :] == 1]
            if ifprob:
                prob = clfs[i].predict_proba(X_)
                clfs_re[:, i] = -1 * prob[:, 0] + 1 * prob[:, 1]
            else:
                clfs_re[:, i] = clfs[i].predict(X_)
        y = rg.predict(clfs_re)
        return y


class EnsembleClassifierCV(EnsembleClassifier):
    """A new version of ensemble classifier.

    This version of classifer is slightly different from EnsembleClassifier. 
    The trained clssifiers of inner CV in the feature selection phase is 
    directly used to build the ensemble classifier. The classification results of
    K classifers in each CV is averaged first and then input to the ensemble
    (the final regression model) to get the final preidctions.

    """

    def __init__(
            self,
            clf,
            optimizer: nsga2.Nsga2,
            evaluator: fit_evaluator.EvaluatorWeight,
            test_X=None,
            test_y=None,
            iter_num=1,
            percentage=1,
            random_state=None,
            iter_save_path=None
    ) -> None:

        super().__init__(
            clf,
            optimizer,
            evaluator,
            test_X,
            test_y,
            iter_num,
            percentage,
            random_state,
            iter_save_path
        )

    def _train_clfs(self, clf, X, y, sols, sample_weight=None):
        evaluator = self.evaluator
        clfs = []
        clfs_re = []
        clf_org = evaluator.clf
        evaluator.clf = clf
        for sol in sols:
            _, _, re = evaluator._get_score(
                sol, X=X, y=y, weights=sample_weight, getclfs=True)
            clfs.append(re[-1])
            clfs_re.append(re[0])
        evaluator.clf = clf_org
        return clfs, clfs_re

    def _predict_sample(self, data, clfs, sols, rg):
        idx = np.nonzero(rg.coef_[0])[0]
        clfs_re = np.zeros((data.shape[0], sols.shape[0]))
        for i in idx:
            X_ = data[:, sols[i, :] == 1]
            if np.sum(sols[i, :]) == 1:
                X_ = X_.reshape((X_.shape[0], 1))
            clfs_re[:, i] = self._get_avg_predict(clfs[i], X_)
        y = rg.predict(clfs_re)
        return y

    def _get_avg_predict(self, clfs, data):
        clf_re = np.zeros((data.shape[0], len(clfs)))
        for i, clf in enumerate(clfs):
            clf_re[:, i] = clf.predict(data)
        mode_, _ = stats.mode(clf_re, axis=1,  keepdims=False)
        return mode_


def build_ensemble(data, y, sample_weight, penalty, solver, random_state=None):
    if penalty == "elasticnet":
        l1_ratio = 0.5
    else:
        l1_ratio = None

    rg = LogisticRegression(
        penalty,
        solver=solver,
        random_state=random_state,
        l1_ratio=l1_ratio
    ).fit(
        data,
        y,
        sample_weight=sample_weight
    )

    return rg


def up_sampling(X, y):
    """Perform up-sampling on the dataset."""

    # Identify majority and minority classes
    majority_mask = y == -1
    minority_mask = y == 1

    X_majority = X[majority_mask]
    y_majority = y[majority_mask]
    X_minority = X[minority_mask]
    y_minority = y[minority_mask]

    majority_size = len(y_majority)
    minority_size = len(y_minority)

    # Calculate replication parameters
    repeat_factor = majority_size // minority_size
    remainder = majority_size % minority_size

    # Create upsampled minority class
    X_resampled = np.tile(X_minority, (repeat_factor, 1))
    y_resampled = np.tile(y_minority, repeat_factor)

    # Add remaining samples if needed
    if remainder > 0:
        X_resampled = np.vstack((X_resampled, X_minority[:remainder]))
        y_resampled = np.hstack((y_resampled, y_minority[:remainder]))

    # Combine majority and upsampled minority classes
    X_balanced = np.vstack((X_majority, X_resampled))
    y_balanced = np.hstack((y_majority, y_resampled))

    return X_balanced, y_balanced
