import numpy as np
from ctypes import cdll

from eval.fitsarray import stack_fits_array
import util.util as util


class NonDomiSort(object):
    def __init__(self, mydll) -> None:
        self.dll = mydll
        

    def non_domi_sort(self, pop, fits, sort=False, dup_handling=False, num_re=None):
        """ 
        sort solutions with fast non-dominated sorting and 
        crowding distance calculation.

        Parameters
        ----------
        pop: ndarray 
            the population for sorting

        fits: FitsArray
            the objective fucntion values for each solution in population

        sort: bool
            define whether sort the output solutions or not

        dup_handling: boolen
            define whether handing the duplicate solutions or not

        Returns
        -------
        s_level_cd: ndarray
            the front level and crowding distance of each solution 

        s_pop: ndarray
            sorted population

        s_fits: ndarray
            sorted objective functions

        """
        dll = self.dll

        if num_re is None:
            num_re = pop.shape[0]

        # get unique solutions from pop
        if dup_handling:
            pop_, u_idx, u_idx_inv = np.unique(
                pop,
                return_index=True,
                return_inverse=True,
                axis=0
            )
            #fits_ = fits[u_idx, :]
            fits_ = fits.sort(u_idx)
        else:
            pop_ = pop
            fits_ = fits

        
        # convert inputs into ctypes
        n_pop_ = pop_.shape[0]
        n_obj = fits.shape[-1]
        c_idx = np.ctypeslib.as_ctypes(np.arange(pop_.shape[0]))
        c_objs = np.ctypeslib.as_ctypes(fits_.fit_values)
        c_level_cds = np.ctypeslib.as_ctypes(np.zeros((n_pop_,2)))
        if num_re > n_pop_:
            c_n_retain = n_pop_
        else:
            c_n_retain = num_re
        
        dll.non_domi_sort(c_idx, c_objs, c_level_cds, n_pop_,
                n_obj, c_n_retain)
        
        c_level_cds = np.ctypeslib.as_array(c_level_cds)
        idx = np.ctypeslib.as_array(c_idx)
        idx = idx[0:c_n_retain]
        s_pop = pop_[idx, :]
        s_level_cd = c_level_cds[idx, :]
        s_fits = fits_.sort(idx)

        # sort solutions
        if sort:
            idx_ = np.lexsort((s_level_cd[:, 1], -s_level_cd[:, 0]), axis=0)
            s_pop = s_pop[idx_, :]
            s_level_cd = s_level_cd[idx_, :]
            s_fits = s_fits.sort(idx_)

        if num_re > c_n_retain:
            num_add = num_re - c_n_retain
            n_levels = s_level_cd[-1, 0]
            s_pop = np.vstack((s_pop, s_pop[0:num_add, :]))
            s_fits = stack_fits_array(s_fits, s_fits.sort(np.arange(0, num_add)))
            s_level_cd = np.vstack((s_level_cd, s_level_cd[0:num_add, :]))
            s_level_cd[-num_add:, 0] = s_level_cd[-num_add:, 0] + n_levels
        
        # util.save_pddata((s_fits[:],s_level_cd), 
        #                  (("f1", "f2"),("level", "cd")), path_name="C_sort_re.csv")        

        return s_level_cd, s_pop, s_fits


