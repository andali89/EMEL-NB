import pandas as pd
import numpy as np
import warnings
import sklearn.metrics as mtrc

def save_pddata(
    data_sets,
    columns,
    path_name=None,
    multi_columns=None
):
    if len(data_sets) != len(columns):
        warnings.warn("the length of data and column is not equal!")
    

    column_names = []
    multi_column_names = []
    for i, column in enumerate(columns):
        num_dcolumn = data_sets[i].shape[-1]
        if not isinstance(column, (tuple, list)) \
                and column.startswith("$"):
            a = column[1:]
            column_names.extend([f"{a}{i+1}"
                                 for i in range(num_dcolumn)])
        elif len(column) != num_dcolumn:
            warnings.warn(f"the length of {i+1}th data and column"
                          "is not equal!")
        else:
            column_names.extend(column)
        if multi_columns is not None:
            multi_column_names.extend(multi_columns[i] for _ in
                                      range(num_dcolumn))
    if multi_columns is not None:
        column_names = pd.MultiIndex.from_arrays(
            [multi_column_names, column_names]
        )

    data = np.hstack(data_sets)
    save_data = pd.DataFrame(data, columns=column_names)
    if path_name is not None:
        save_data.to_csv(path_name)
    return save_data

def get_clf_score(test_y, pred_y):
        """Get classification performance results on different measures.

        Parameters
        ----------
        test_y : array-like of shape (n_samples,)
            The true class labels of samples.

        pred_y : array-like of shape (n_samples,)
            The predicted class labels of samples.

        Returns 
        -------
        re : array-like of shape (n_measures,)
          Store the results of accuracy, balanced_accuracy, F1_score, \
            F1_score_Macro, GM, recall (sensitivity), specificity, precision.
        measures : tuple
            The names of measures.

        """   
        measures = ("accuracy", "balanced_accuracy", "F1_score",
                       "F1_score_macro", "G-mean", "recall",
                       "specificity", "precision")
        tpr = mtrc.recall_score(test_y, pred_y, pos_label=1)
        tnr = mtrc.recall_score(test_y, pred_y, pos_label=-1)
        
        [tpr_, tnr_] = cal_recall(test_y, pred_y)
        print(f"Compare tpr: {tpr} vs {tpr_}, tnr: {tnr} vs {tnr_}")
        
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
        return np.array(re), measures
    
def cal_recall(test_y, pred_y):
    """Calculate recall (sensitivity) and specificity.

    Parameters
    ----------
    test_y : array-like of shape (n_samples,)
        The true class labels of samples.

    pred_y : array-like of shape (n_samples,)
        The predicted class labels of samples.

    Returns 
    -------
    re : array-like of shape (2,)
        Store the results of recall (sensitivity) and specificity.

    """
    tp = np.sum((pred_y == 1) & (test_y == 1))
    fp = np.sum((pred_y == 1) & (test_y == -1))
    tn = np.sum((pred_y == -1) & (test_y == -1))
    fn = np.sum((pred_y == -1) & (test_y == 1))
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return np.array([tpr, tnr])
