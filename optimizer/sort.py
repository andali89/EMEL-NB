import numpy as np
import matplotlib.pyplot as plt
import util.util as util

def non_domi_sort(pop, fits, sort=False, dup_handling=False, num_re=None):
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

    # get n_dominated_set and dominate_set
    n_pop_ = pop_.shape[0]
    dominated_matrix = np.zeros((n_pop_, n_pop_))
    for i in range(n_pop_-1):
        dominated_matrix[i, i+1:] = list(
            map(
            lambda x: dominate(fits_[i, :], x),
            fits_[i+1:, :]
            )
        )
        
    dominated_matrix -= dominated_matrix.T
    for i in range(1, n_pop_):
        idx = np.nonzero(dominated_matrix[i, 0:i] == 0)
        idx = idx[0]
        if len(idx) > 0:
            dominated_matrix[i, idx] = list(
                map(
                lambda x: dominate(fits_[i, :], x),
                fits_[idx, :]
                )
            )

    # set storing the solutions that are dominated by a solution
    dominate_set = [np.argwhere(dominated_matrix[x, :] == 1).flatten()
                    for x in range(n_pop_)]

    # set storing the number of solutions dominate a solution
    n_dominated_set = np.count_nonzero(dominated_matrix == 1, axis=0)

    level_cd_ = np.zeros((n_pop_, 2))

    # obtain front levels
    n_termin = np.min((n_pop_, num_re))
    n_added = 0
    F = []
    level = 0
    while n_added < n_termin:
        # find non-dominated solutions in current level
        level += 1
        idx = np.argwhere(n_dominated_set == 0).flatten()
        F.append(idx)
        n_dominated_set[idx] = -1
        level_cd_[idx, 0] = level
        n_added += len(idx)

        # update n_dominated_set
        for i in idx:
            n_dominated_set[dominate_set[i]] -= 1

    # calculate crowding distance
    num_obj = fits_.shape[1]
    for level_idx in F:
        #dirct = -1                
        for j in range(num_obj): 
            #dirct = -dirct  
            if j==1 and num_obj==2:
                sorted_level_idx = sorted_level_idx[::-1]                 
            else:                
                sorted_level_idx = level_idx[np.argsort(fits_[level_idx, j])]  

            level_cd_[sorted_level_idx[[0, -1]], 1] = np.Inf
            scale = fits_[sorted_level_idx[-1], j] - \
                fits_[sorted_level_idx[0], j]
                        
            if len(sorted_level_idx) < 3:
                continue
            
            if scale == 0:
                scale = 1
                
            for i in range(1, sorted_level_idx.size-1):                
                level_cd_[sorted_level_idx[i], 1] += \
                    (fits_[sorted_level_idx[i+1], j] -
                     fits_[sorted_level_idx[i-1], j]) / scale

    # update F to get num_re solutions    
    if n_added < num_re:
        n_sup = num_re - n_added
        i = 0        
        n_f = len(F)
        while n_sup > 0:
            F.append(F[i])
            i += 1
            i = i % n_f
            n_sup -= F[i].size

        level_idx = F[i]
        sorted_idx = level_idx[np.argsort(level_cd_[level_idx, 1])[::-1]]
        F[i] = sorted_idx[0:level_idx.size+n_sup]
       
    else:
        level_idx = F[-1]
        sorted_idx = level_idx[np.argsort(level_cd_[level_idx, 1])[::-1]]
        F[-1] = sorted_idx[0:level_idx.size-(n_added-num_re)]
        
    # sort solutions
    if sort:
        for i, level_idx in enumerate(F[:-1]):
            F[i] = level_idx[np.argsort(level_cd_[level_idx, 1])[::-1]]

    idx = np.concatenate(F)   
    s_pop = pop_[idx, :]
    s_fits = fits_.sort(idx)
    s_level_cd = level_cd_[idx, :]

    # revise the front level of duplicate solutions added
    if n_added < num_re:
        s_level_cd[n_added:-1, 0] += n_f
    
    # util.save_pddata((s_fits[:],s_level_cd), 
    #                      (("f1", "f2"),("level", "cd")), path_name="Org_sort_re.csv")

    return s_level_cd, s_pop, s_fits


def dominate(a, b):
    # if a dominate b, return True, the smaller the better     
    return np.all(a <= b) and any(np.not_equal(a, b))

def scatter(fits, F):
    fig, ax = plt.subplots()
    for i in F:
        ax.scatter(fits[i,0],fits[i,1])

    

