
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import get_scorer, balanced_accuracy_score
from sklearn.base import clone


class Evaluator(object):

    def __init__(
            self,
            clf,
            data_X=None,
            data_y=None,
            n_fold=None,
            measure="balanced_accuracy",
            if_gb=True,
            obj_type="single"
    ) -> None:

        self.data_X = data_X
        self.data_y = data_y
        self.n_fold = n_fold
        self.measure = measure
        self.if_gb = if_gb  # greater better
        self.set_clf(clf)
        if obj_type == "single":
            self.evaluate = self._evaluate_single
            self._num_obj = 1
        elif obj_type == "double":
            self.evaluate = self._evaluate_num_classi
            self._num_obj = 2
        else:
            print("Caution:obj_type is not correctly defined!")

    @property
    def num_obj(self):
        return self._num_obj

    def set_clf(self, clf):
        self.clf = clf

    def set_data(self, data_X, data_y):
        self.data_X = data_X
        self.data_y = data_y

    def _get_score(self, sol):
        idx = sol == 1
        X = self.data_X[:, idx]
        y = self.data_y    
        # print(self._test_get_score(X, y))    
        scores = cross_val_score(self.clf, X,
                                 y, cv=self.n_fold,
                                 scoring=self.measure)
        # print(scores)
        return np.sum(idx), np.mean(scores), None
    
    def _test_get_score(self, X, y):       
        """Test function to perform a StratifiedKFold cross-validation to get score.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The class labels.
        Returns
        -------
        tuple: (y_true, y_pred)
            y_true: array-like of shape (n_samples,) - The true class labels.
            y_pred: array-like of shape (n_samples,) - The predicted class labels.
        """
        cv = StratifiedKFold(n_splits=self.n_fold)
        fold_results = []
        all_y_true = []
        all_y_pred = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self.clf.fit(X_train, y_train)
            y_pred = self.clf.predict(X_test)

            fold_results.append({
                'fold':   fold_idx + 1,
                'y_true': y_test,
                'y_pred': y_pred,
            })
            all_y_true.append(y_test)
            all_y_pred.append(y_pred)
            print(f"Fold {fold_idx + 1} - True, Pred: {np.unique(y_test)}, {np.unique(y_pred)}")

        # Equivalent aggregate to the original cross_val_score mean
        from sklearn.metrics import get_scorer
        scorer = get_scorer(self.measure)
        scores = [
            scorer._score_func(r['y_true'], r['y_pred'])
            for r in fold_results
        ]
        return scores
    
    def _get_score_skf(self, sol):
        """
        Get the evaluation score for the input solution using StratifiedKFold.

        Parameters
        ----------
        sol : array-like of shape (n_features,)
            The input solution (feature subset).
        

        Returns
        -------
        num_features : scalar
            The number of features in the input solution.

        score : scalar
            The score on the classification measure of the input solution.
        """
        idx = sol == 1
        X = self.data_X[:, idx]
        y = self.data_y
        skf = StratifiedKFold(n_splits=self.n_fold)
        scores = np.zeros(self.n_fold)
        scorer = get_scorer(self.measure)
        for i, (train, test) in enumerate(skf.split(X, y)):
            self.clf.fit(X[train, :], y[train])
            scores[i] = scorer(self.clf, X[test, :], y[test])
        
        return np.sum(idx), np.mean(scores), None

    def _evaluate_single(self, sol):
        """
        single objective FS model, performance or 1 - performance
        """
        _, score, _ = self._get_score(sol)
        return (score if self.if_gb else 1-score)

    def _evaluate_num_classi(self, sol):
        """
        bi-objective FS model, (f1,f2)=(num, 1-performance)

        """
        num, score, _ = self._get_score(sol)
        return (num, 1-score), None
    
    def get_params(self):
        """
        Get parameters of the evaluator.
        
        Returns:
            dict: Parameters of the evaluator
        """
        params = {
            "fit_evaluator_type": self.__class__.__name__,
            "fit_measure": self.measure,
            "fit_if_gb": self.if_gb,
            "fit_num_obj": self._num_obj,
            "fit_n_fold": self.n_fold,
            "fit_if_weight_test": getattr(self, 'weight_test', False),
        }
        return params


class EvaluatorWeight(Evaluator):
    def __init__(
            self,
            clf,
            data_X=None,
            data_y=None,
            n_fold=None,
            measure="balanced_accuracy",
            if_gb=True,
            obj_type="single",
            sample_weight=None,
            n_jobs=None,
            n_class=2            
    ) -> None:

        super().__init__(
            clf,
            data_X,
            data_y,
            n_fold,
            measure,
            if_gb,
            obj_type
        )
        self.weight_test = True
        self.n_class = n_class
        self.sample_weight = sample_weight
        self.scorer = get_scorer(measure)
        if obj_type == "double":
            self.evaluate = self._evaluate_num_classi_pred
            self._num_obj = 2
        if n_jobs is not None:
            self._get_score_ = self._get_score_njobs
            self.n_jobs = n_jobs
        else:
            self._get_score_ = self._get_score
    
    def set_weight_test(self, weight_test):
        """set if the weight is considerd in calculating score
            The sample weights are considered in traing the classifered for evaluation,
            no matter weight_test is True or False.

        Args:
            weight_test: bool
                True, the sample weight is considerd in calcualting the score.
                False, the samaple weight is NOT considered in calcualting the score.
                
        """
        self.weight_test = weight_test
        

    def set_sample_weight(self, sample_weight):
        self.sample_weight = sample_weight

    def _get_score_njobs(self, sol, X=None, y=None, weights=None):
        """
        Get the evaluation score for the input solution.

        Parameters
        ----------
        sol : array-like of shape (n_features,)
            The input solution (feature subset).

        X : array-like of shape (n_samples, n_features), optional
            The input X. If None is given, the X of class is used.

        y : array-like of shape (n_samples,), optional
            The input class label y. If None is given, the y of 
            class is used.

        weights : ndarray of shape (n_samples,), optional
            The sample weights.


        Returns
        -------

        num_features : scalar
            The number of features in the input solution.

        score : scalar
            The score on the classification measure of the input solution.

        info : tuple (predictions, pred_correct, clfs)
             predictions : array-like of shape (n_samples,) 
                 The actuall predicions for each sample.
             pred_correct : array-like of shape (n_samples,) 
                 If each sample is correctly predicted (True or False).

        """

        w = self.sample_weight if weights is None else weights
        idx = sol == 1
        num_sel = np.sum(idx)
        if X is None:
            X = self.data_X[:, idx]
        else:
            X = X[:, idx]
        if y is None:
            y = self.data_y
        if num_sel == 0:
            return 10000,  -1, (y != y)
        pred_y = y.copy()
        skf = StratifiedKFold(n_splits=self.n_fold)
        if w is None:
            pred_y = cross_val_predict(
                self.clf, X, y, cv=skf, n_jobs=self.n_jobs)

        else:
            pred_y = cross_val_predict(self.clf, X, y, cv=skf, params={
                                       "sample_weight": w}, n_jobs=self.n_jobs)

        score = self.scorer._score_func(y, pred_y, sample_weight=w)
        pred_correct = (y == pred_y)
        info = [pred_y, pred_correct]

        return np.sum(idx), score, info

    def _get_score(self, sol, X=None, y=None, weights=None, getclfs=False, get_proba=False):
        """
        Get the evaluation score for the input solution.

        Parameters
        ----------
        sol : array-like of shape (n_features,)
            The input solution (feature subset).

        X : array-like of shape (n_samples, n_features), optional
            The input X. If None is given, the X of class is used.

        y : array-like of shape (n_samples,), optional
            The input class label y. If None is given, the y of 
            class is used.

        weights : ndarray of shape (n_samples,), optional
            The sample weights.

        getclfs : bool
            If the classifiers built with the given solution are returned.
        
        getproba : bool
            If the probalibity information of classifier is returned.

        Returns
        -------

        num_features : scalar
            The number of features in the input solution.

        score : scalar
            The score on the classification measure of the input solution.

        info : tuple (predictions, pred_correct, clfs)
             predictions : array-like of shape (n_samples,) 
                 The actuall predicions for each sample.
             pred_correct : array-like of shape (n_samples,) 
                 If each sample is correctly predicted (True or False).
             pred_prob : array-like of shape (n_samples,)

             clfs : tuple of classifiers, optional
                 The classifiers.        
        """

        w = self.sample_weight if weights is None else weights
        idx = sol == 1
        num_sel = np.sum(idx)
        if X is None:
            X = self.data_X[:, idx]
        else:
            X = X[:, idx]
        if y is None:
            y = self.data_y
        if num_sel == 0:
            return 10000,  -1, (y != y)
        pred_y = y.copy()
        skf = StratifiedKFold(n_splits=self.n_fold)
        if get_proba:
            pred_prob = np.zeros((X.shape[0], self.n_class))
        clfs = []
        if w is None:
            for i, (train, test) in enumerate(skf.split(X, y)):
                if getclfs:
                    clf_ = clone(self.clf)
                else:
                    clf_ = self.clf
                clf_.fit(X[train, :], y[train])
                pred_y[test] = clf_.predict(X[test, :])
                if get_proba:
                    pred_prob[test, :] = clf_.predict_proba(X[test, :])
                clfs.append(clf_)
        else:
            for i, (train, test) in enumerate(skf.split(X, y)):
                if getclfs:
                    clf_ = clone(self.clf)
                else:
                    clf_ = self.clf
                clf_.fit(X[train, :], y[train], w[train])
                pred_y[test] = clf_.predict(X[test, :])
                if get_proba:
                    try:
                        pred_prob[test, :] = clf_.predict_proba(X[test, :])
                    except:
                        # If the classifier does not support predict_proba, using predict instead
                        pred_prob[test, :] = None
                clfs.append(clf_)
        if self.weight_test:
            score = self.scorer._score_func(y, pred_y, sample_weight=w)
        else:
            score = self.scorer._score_func(y, pred_y)
        pred_correct = (y == pred_y)
        info = [pred_y, pred_correct]
        if get_proba:
            info.append(pred_prob)
        if getclfs:
            info.append(clfs)

        return np.sum(idx), score, info

    def _evaluate_num_classi_pred(self, sol, get_proba=False):
        """
        bi-objective FS model, (f1,f2)=(num, 1-performance)
        pred_correct denotes additional infomation

        """
        num, score, info = self._get_score_(sol, get_proba=get_proba)
        return (num, 1-score), info


class EvaluatorWeightHoldout(EvaluatorWeight):
    def __init__(
        self,
        clf,
        train_X,
        train_y,
        test_X,
        test_y,
        measure="balanced_accuracy",
        if_gb=True,
        obj_type="single",
        sample_weight_train=None,
        sample_weight_test=None
    ) -> None:

        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.sample_weight_train = sample_weight_train
        self.sample_weight_test = sample_weight_test
        super().__init__(
            clf,
            None,
            None,
            None,
            measure,
            if_gb,
            obj_type,
            sample_weight=None
        )
        self._get_score = self._get_score_holdout

    def _get_score_holdout(self, sol):
        """Get the prediction score on the holdout set.

        Returns
        -------
        num_attributes : int
            Return the number of features selected by solution.

        score: double
            Return the prediction score on given score metric

        pred_correct: tuple (array of correctly predicted on training set,\
            array of correctly predicted on test set)
        """

        idx = sol == 1
        train_X = self.train_X[:, idx]
        train_y = self.train_y
        test_y = self.test_y
        test_X = self.test_X[:, idx]
        self.clf.fit(train_X, train_y, self.sample_weight_train)
        pred_train_y = self.clf.predict(train_X)
        pred_test_y = self.clf.predict(test_X)
        score = self.scorer._score_func(
            test_y, pred_test_y, sample_weight=self.sample_weight_test)
        pred_correct = (train_y == pred_train_y, test_y == pred_test_y)
        return np.sum(idx), score, pred_correct
