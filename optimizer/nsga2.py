import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from optimizer.ga import GA
from optimizer.sort import *
import matplotlib.pyplot as plt
import time
from eval.fitsarray import *
import ctypes
from util.util import save_pddata
import pandas as pd


class Nsga2(GA):

    def __init__(
            self,
            var_length: int,
            pop_size: int,
            c_rate: float,
            m_rate: float,
            n_iters=100,
            fit_evaluator=None,
            sort_operator=non_domi_sort,
            no_prd=True,
            random_state=1,
            ifsort=False,
            if_ba_mu=False,
            dll=None
    ) -> None:

        self.sort = sort_operator
        self.iter_status = False
        self.no_prd = no_prd
        self.ifsort = ifsort
        self.sort_time = 0

        super().__init__(
            var_length,
            pop_size,
            c_rate,
            m_rate,
            n_iters,
            fit_evaluator,
            random_state
        )

        if dll is not None:
            self.gen_off = self._gen_off_c
            self.dll = dll
            dll.set_rnd_seed(random_state)
            self.m_rate = ctypes.c_double(self.m_rate)
            self.c_rate = ctypes.c_double(self.c_rate)
            if if_ba_mu:
                self.dll_gen_off = dll.gen_off_bamu
            else:
                self.dll_gen_off = dll.gen_off
        else:
            self.gen_off = self._gen_off

        self.if_ba_mu = if_ba_mu
        if if_ba_mu:
            self._mutate = self._ba_mutate

        if no_prd:
            self.__eval_fits = self.__evaluate_fits
        else:
            self.__eval_fits = self.__evaluate_fits_prd
    
    def _set_evaluator(self, evaluator):
        self.fit_evaluator = evaluator

    def _compare(self, level_cds1, level_cds2):
        """for multiple objectives """
        return list(map(self.__compare_single, level_cds1, level_cds2))

    def __compare_single(self, level_cd1, level_cd2):
        if level_cd1[0] < level_cd2[0] or (level_cd1[0] == level_cd2[0]
                                           and level_cd1[1] >= level_cd2[1]):
            return True
        else:
            return False

    def __evaluate_fits(self, fits, idx, sol):
        fits[idx, :], _ = self.fit_evaluator.evaluate(sol)

    def __evaluate_fits_prd(self, fits, idx, sol, get_proba=False):
        fits[idx, :], info = self.fit_evaluator.evaluate(
            sol, get_proba=get_proba)
        fits.set_predic_correct(idx, info)

    def test_sort_time(self, times):
        # initialization and evaluate fitness
        pop = self._ini_pop()
        fits = FitsArray(shape=(self.pop_size, self.fit_evaluator.num_obj),
                         no_prd=self.no_prd)
        for i in range(fits.shape[0]):
            self.__eval_fits(fits, i, pop[i, :])
        
        time_s = time.perf_counter()
        for i in range(times):
            print(i)
            level_cds, pop, fits = self.sort(
                pop, fits, num_re=self.pop_size, dup_handling=True)
        sort_time = time.perf_counter() - time_s
        return sort_time

    def run(self):
        time_start = time.perf_counter()
        self.iter_sols = []
        self.iter_fits = []
        self.iter_level_cds = []
        # initialization and evaluate fitness
        pop = self._ini_pop()
        fits = FitsArray(shape=(self.pop_size, self.fit_evaluator.num_obj),
                         no_prd=self.no_prd)
        for i in range(fits.shape[0]):
            self.__eval_fits(fits, i, pop[i, :])

        level_cds, pop, fits = self.sort(
            pop, fits, num_re=self.pop_size, sort=self.ifsort, dup_handling=True)
        self.iter_sols.append(pop.copy())
        self.iter_fits.append(fits.copy())
        self.iter_level_cds.append(level_cds.copy())

        # begin iterations
        iter = 0
        off_fits = FitsArray(shape=(self.pop_size, self.fit_evaluator.num_obj),
                             no_prd=self.no_prd)
        off_springs = np.zeros(
            (self.pop_size, self.var_length), dtype=np.int32)
        while iter < self.n_iters:
            iter += 1
            print(f"iteration: {iter}")

            # selection
            idx = self._bin_tour_sel(level_cds)

            # crossover and mutation
            off_springs = self.gen_off(pop, off_springs, idx)

            # fitness evaluation
            for i in range(off_fits.shape[0]):
                self.__eval_fits(off_fits, i, off_springs[i, :])

            # combine pop and off_springs into pool
            pool = np.concatenate((pop, off_springs), axis=0)
            pool_fits = stack_fits_array(fits, off_fits)

            # sort solutions in pool according to fitness in descending order
            level_cds, pop, fits = self.sort(
                pool, pool_fits, sort=self.ifsort,
                num_re=self.pop_size, dup_handling=True)

            if (iter) == self.n_iters:
                # update information
                for i in range(self.pop_size):
                    self.__eval_fits(fits, i, pop[i, :], get_proba=True)

            # iteration information
            self.iter_sols.append(pop.copy())
            self.iter_fits.append(fits.copy())
            self.iter_level_cds.append(level_cds.copy())

            # plot the figure
            self._plot_iter_info(fits, iter)

        # sort and out put the final solution
        # if not self.ifsort:
        #     idx = np.lexsort((-level_cds[:, 1], level_cds[:, 0]))
        #     self.final_pop = pop[idx, :]
        #     self.final_fits = fits.sort(idx)
        #     self.final_level_cds = level_cds[idx, :]
        # else:
        self.final_pop = pop
        self.final_fits = fits
        self.final_level_cds = level_cds
        self.run_time = time.perf_counter() - time_start
        print(f"the running time is {self.run_time} seconds")

    def _gen_off(self, pop, off_springs, idx):
        for i in range(0, idx.size, 2):
            a, b = self._cross_over(pop[idx[i], :], pop[idx[i+1], :])
            off_springs[i, :] = a
            off_springs[i+1, :] = b

            self._mutate(off_springs[i, :])
            self._mutate(off_springs[i+1, :])
        return off_springs

    def _gen_off_c(self, pop, off_springs, idx):
        off_springs[:] = pop[idx, :]
        off = np.ctypeslib.as_ctypes(off_springs)
        self.dll_gen_off(off, self.c_rate,
                         self.m_rate, off_springs.shape[0],
                         off_springs.shape[1])
        off_springs = np.ctypeslib.as_array(off)
        return off_springs

    def get_run_time(self):
        return self.run_time

    def get_best_results(self):
        m_idx = self.iter_level_cds[-1][:, 0] == 1
        best_sols = self.iter_sols[-1][m_idx, :]
        best_fits = self.iter_fits[-1].sort(m_idx)
        best_level_cds = self.iter_level_cds[-1][m_idx, :]
        return best_sols.copy(), best_fits.copy(), best_level_cds.copy()

    def get_final_results(self):
        return self.final_pop.copy(), self.final_fits.copy(), \
            self.final_level_cds.copy()

    def get_iter_results(self):
        return self.iter_sols.copy(), self.iter_fits.copy(), \
            self.iter_level_cds.copy()

    def save_iter_results(self, savepath=None, clf_detail=False):
        idx = np.zeros((self.pop_size,))
        idx = idx.reshape((self.pop_size, 1))
        pd_data = []
        for i in range(self.n_iters):
            idx[:, :] = i
            data = [idx,
                    self.iter_level_cds[i],
                    self.iter_fits[i][:, :],
                    self.iter_sols[i]]
            header = [("iteration",),
                      ("level", "crowding_distance"),
                      "$f",
                      "$x"]
            if clf_detail:
                prd = [x[0] for x in self.iter_fits[i].predic_correct]
                prd_tf = [x[1] for x in self.iter_fits[i].predic_correct]
                data.extend([np.vstack(prd), np.vstack(prd_tf)])
                header.extend(["$sample", "$tf_sample"])
            pd_data_ = save_pddata(data,
                                   header,
                                   path_name=None)
            pd_data.append(pd_data_)
        save_data = pd.concat(pd_data, axis=0)
        if savepath is not None:
            save_data.to_csv(savepath)
        return save_data

    def get_params(self):
        params = {
            "num_iters": self.n_iters,
            "pop_size": self.pop_size,
            "cross_over_rate": self.c_rate,
            "mutation_rate": self.m_rate,
            "mutation_operator": "balanced" if self.if_ba_mu else "standard",
            "dll_sort": self.sort.__str__(),
            "random_state": self.random_state
        }
        return params

    def ini_iter_figure(self):
        self.fig, self.ax = plt.subplots()
        self.iter_status = True
        return self.fig, self.ax

    def _plot_iter_info(self, fits, iter_num=-1):
        if self.iter_status:
            self.ax.clear()
            self.ax.scatter(fits[:, 0], fits[:, 1])
            self.ax.set_title(f"iteration {iter_num}")
            plt.pause(0.01)

    def _ba_mutate(self, sol):
        """
        Balanced crossover, mutation on sol inplace

        """
        
        idx1 = sol == 1
        idx0 = sol == 0
        r = self.rng.random((sol.shape[-1],))
        thred = np.zeros(sol.shape[-1])
        thred[idx0] = self.m_rate * (np.sum(idx1) /
                                     np.sum(idx0))
        thred[idx1] = self.m_rate
        idx = r <= thred
        sol[idx] = 1 - sol[idx]
        return sol
