
from sklearn.model_selection import train_test_split
import sklearn.metrics as mtrc
import numpy as np
import pandas as pd
from util.util import save_pddata
import time

class FsPerformance(object):
    def __init__(
            self,
            classifier,
            train_X=None,
            train_y=None,
            test_X=None,
            test_y=None,
            data_X=None,
            data_y=None,
    ) -> None:
        self.classifier = classifier
        self.train_X = train_X
        self.test_X = test_X
        self.train_y = train_y
        self.test_y = test_y
        self.data_X = data_X
        self.data_y = data_y
        self.pos = 1
        self.neg = -1

    def split_data(self, test_size: float, random_state: int):
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            self.data_X, self.data_y, test_size, random_state=random_state)

    def evaluate_save(
        self,
        solutions,
        fits=None,
        level_cd=None,
        save_solutions=False,
        path_name= None,
        use_multi=True
    ):
        performance = self.evaluate_solutions(solutions)
        if not save_solutions:
            solutions_ = None
        else:
            solutions_ = solutions
        return self.save_performance(
            performance,
            solutions_,
            level_cd,
            fits,
            path_name=path_name,
            use_multi=use_multi
        )

    def save_performance(self, performance, solutions=None, level_cd=None,
                         fits=None, path_name=None, use_multi=True):
        column_set = [("accuracy", "balanced_accuracy", "F1_score",
                       "F1_score_macro", "G-mean", "recall",
                       "specificity", "precision")]
        data = []
        multi_columns = []
        data.append(performance)
        multi_columns.append("Performance Metrics")
        if fits is not None:
            data.append(fits)
            column_set.append("$f")
            multi_columns.append("Objective Function Values")
        if level_cd is not None:
            data.append(level_cd)
            column_set.append(("level", "crowding_distance"))
            multi_columns.append("Sorting Results")
        if solutions is not None:
            data.append(solutions)
            column_set.append("$x")
            multi_columns.append("Solutions")
        if not use_multi:
            multi_columns = None
        return save_pddata(data, column_set, path_name, multi_columns)

    def evaluate_solutions(self, solutions):
        cl_performance = []
        for i, x in enumerate(solutions):
            cl_performance.append(self._evaluate_solution(x))
        return np.vstack(cl_performance)

    def _evaluate_solution(self, solution):
        """
        return 
        -------
        re: array
          store the results of accuracy, balanced_accuracy, F1_score, 
            F1_score_Macro, GM, recall (sensitivity), specificity, precision

        """
        idx = solution == 1
        X = self.train_X[:, idx]
        y = self.train_y
        test_X = self.test_X[:, idx]
        test_y = self.test_y
        time_start = time.perf_counter()
        clf = self.classifier.fit(X, y)
        self.train_time = time.perf_counter() - time_start
        time_start = time.perf_counter()
        pred_y = clf.predict(test_X)
        self.test_time = time.perf_counter() - time_start

        tpr = mtrc.recall_score(test_y, pred_y, pos_label=1)
        tnr = mtrc.recall_score(test_y, pred_y, pos_label=self.neg)
        gm = np.sqrt(tpr*tnr)
        re = []
        re.append(mtrc.accuracy_score(test_y, pred_y))
        re.append(mtrc.balanced_accuracy_score(test_y, pred_y))
        re.append(mtrc.f1_score(test_y, pred_y))
        re.append(mtrc.f1_score(test_y, pred_y, average="macro"))
        re.append(gm)
        re.append(tpr)
        re.append(tnr)
        re.append(mtrc.precision_score(test_y, pred_y))
        return np.array(re)
    
    def get_times(self):
        return self.train_time, self.test_time
