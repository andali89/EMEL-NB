import numpy as np
import copy
import warnings

class FitsArray(object):
    def __init__(
            self, 
            fit_values=None, 
            predic_correct=None, 
            shape:tuple=None,
            no_prd=True) -> None:        
        
        if fit_values is not None:
            self.num_sample, self.num_obj = fit_values.shape
            self._fit_values = fit_values
        elif shape is not None:
            self.num_sample, self.num_obj = shape
            self._fit_values = np.zeros(shape=shape)
        else:
            warnings.warn("The fit_values or shape of FitsArray should be given!")
        
        if predic_correct is None:
            self._predic_correct = [None for _ in range(self.num_sample)]
        else:
            self._predic_correct = predic_correct
        
        self.no_prd = no_prd
        if no_prd:            
            self._predic_correct = None
                
    
    def __getitem__(self, item):
        return self._fit_values[item]        
        
    def __setitem__(self, key, value):
        self._fit_values[key] = value

    @property
    def shape(self):
        return self.num_sample, self.num_obj
    
    def set_predic_correct(self, idx, value):        
        if self.no_prd:
            print("the option of use_prd is off!")
            return
        if isinstance(idx, slice):
            if len(list(range(idx.start,idx.stop,idx.step))) != len(value):
               warnings.warn("The length of idx and value is not equal") 
               return
            self._predic_correct[idx] = value
            return self
        if isinstance(idx, np.ndarray) and idx.dtype=="bool":
            idx = np.nonzero(idx)
            idx = idx[0]
        if isinstance(idx, int) or len(idx) == 1:
            self._predic_correct[idx] = value
        else:
            if len(idx) != len(value):
                warnings.warn("The length of idx and value is not equal") 
                return
            for i, id in enumerate(idx):
                self._predic_correct[id] = value[i]
        return self

    @property
    def fit_values(self):
        return self._fit_values.copy()
    
    @property
    def predic_correct(self):
        return copy.deepcopy(self._predic_correct)
    
    def copy(self):        
        return FitsArray(self.fit_values.copy(), 
                         self.predic_correct, no_prd=self.no_prd)
    
    def sort(self, idx):
        """
        sort the fit values and predic_correct values according 
        to the order given in idx, return a new FitsArray
        """
        if self.no_prd:
            return FitsArray(self._fit_values[idx].copy())

        if isinstance(idx, slice):
            predic_correct_ = copy.deepcopy(self._predic_correct[idx])
            return FitsArray(self._fit_values[idx].copy(), predic_correct_, no_prd=False)
        if isinstance(idx, np.ndarray) and idx.dtype=="bool":
            idx = np.nonzero(idx)
            idx = idx[0]        
        predic_correct_ = []
        for id in idx:
            predic_correct_.append(self._predic_correct[id])        
        return FitsArray(self._fit_values[idx].copy(), 
                         copy.deepcopy(predic_correct_), no_prd=False)       



def stack_fits_array(fits1, fits2):
    """
        stack two Fits objects
    """

    fit_values = np.vstack((fits1.fit_values, fits2.fit_values))
    if fits1.no_prd or fits2.no_prd:
        return FitsArray(fit_values)
    
    predic_correct = fits1.predic_correct + fits2.predic_correct
    return FitsArray(fit_values, predic_correct, no_prd=False)
    