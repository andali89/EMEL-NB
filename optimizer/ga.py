
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng


class GA(object):

    def __init__(
            self,
            var_length: int,
            pop_size: int,
            c_rate: float,
            m_rate: float,
            n_iters=100,
            fit_evaluator=None,
            random_state=1
    ) -> None:

        self.pop_size = pop_size
        self.var_length = var_length
        self.c_rate = c_rate
        self.m_rate = m_rate
        self.n_iters = n_iters
        
        self.random_state = random_state
        self.reset_random_state()
        self._ini_probs = None
        self._use_mi_ini = False
        self.fit_evaluator = fit_evaluator



    def set_ini_probs(self, ini_probs):
        """Set per-feature initialization probabilities for population initialization.

        Parameters
        ----------
        ini_probs : array-like of shape (var_length,) or None
            Probability that each feature bit is initialized to 1.
            Pass None to revert to the default uniform random initialization.
        """
        self._ini_probs = ini_probs
        self._use_mi_ini = True if ini_probs is not None else False

    def _ini_pop(self):
        if self._use_mi_ini:
            r = self.rng.random(size=(self.pop_size, self.var_length))
            print(f"MI guided Initialization Method used")
            return (r < self._ini_probs).astype(np.int32)
        return self.rng.integers(2, size=(self.pop_size, self.var_length), dtype=np.int32)
    
    def reset_random_state(self):
        self.rng = default_rng(self.random_state)


    def _cross_over(self, sol1, sol2):
        n_sol1 = sol1.copy()
        n_sol2 = sol2.copy()
        if self.rng.random() < self.c_rate:
            c_point = self.rng.integers(self.var_length - 1) + 1
            n_sol1[c_point:] = sol2[c_point:]
            n_sol2[c_point:] = sol1[c_point:]
        return n_sol1, n_sol2

    def _mutate(self, sol) -> None:
        r = self.rng.random(self.var_length)
        m_idx = r < self.m_rate
        sol[m_idx] = 1 - sol[m_idx]
        return sol

    def _bin_tour_sel(self, pop_fitness):
        """
        Peform binary tournament selection

        Parameters
        -----------
        pop_fitness: ndarray
          the fitness values of solutions in single objective scenario
          the ranking information in multi-objective scenario

        """
        f_size = pop_fitness.shape[0]
        idx = np.concatenate(
            (self.rng.permutation(f_size),
             self.rng.permutation(f_size)),
            axis=0
        )
        sel_idx = idx[::2]
        n_sel_idx = idx[1::2]
        # c_idx = self._compare(pop_fitness[sel_idx], pop_fitness[n_sel_idx]) org
        c_idx = self._compare(pop_fitness[n_sel_idx], pop_fitness[sel_idx])
        sel_idx[c_idx] = n_sel_idx[c_idx]
        return sel_idx

    def _compare(self, fits1, fits2):
        """for single objective """
        return fits1 <= fits2

    def run(self):
        self.best_sols = []
        self.best_fits = []

        # initialization and evaluate fitness
        pop = self._ini_pop()
        fits = np.zeros(self.pop_size)
        for i in range(fits.size):
            fits[i] = self.fit_evaluator.evaluate(pop[i, :])

        m_idx = np.argmax(fits)
        self.best_sols.append(pop[m_idx, :].copy())
        self.best_fits.append(fits[m_idx].copy())

        # begin iterations
        iter = 0
        off_fits = np.zeros(self.pop_size)
        off_springs = np.zeros((self.pop_size, self.var_length))
        while iter < self.n_iters:
            iter += 1
            print(f"iteration: {iter}")

            # selection
            idx = self._bin_tour_sel(fits)

            # crossover and mutation
            for i in range(0, idx.size, 2):
                a, b = self._cross_over(pop[idx[i], :], pop[idx[i+1], :])
                off_springs[i, :] = a
                off_springs[i+1, :] = b

                self._mutate(off_springs[i, :])
                self._mutate(off_springs[i+1, :])

            # fitness evaluation
            for i in range(fits.size):
                off_fits[i] = self.fit_evaluator.evaluate(off_springs[i, :])

            # combine pop and off_springs into pool
            pool = np.concatenate((pop, off_springs), axis=0)
            pool_fits = np.concatenate((fits, off_fits), axis=0)

            # sort solutions in pool according to fitness in descending order
            s_idx = np.argsort(pool_fits)[-1:-self.pop_size-1:-1]

            # evironment selection
            pop = pool[s_idx]
            fits = pool_fits[s_idx]

            # iteration information
            m_idx = np.argmax(fits)
            self.best_sols.append(pop[m_idx, :].copy())
            self.best_fits.append(fits[m_idx].copy())

    def set_evaluator(self, evaluator):
        self.fit_evaluator = evaluator

    def get_best_results(self):
        return self.best_sols[-1].copy(), self.best_fits[-1].copy()

    def get_iter_results(self):
        return self.best_sols.copy(), self.best_fits.copy()

    def plot_iter_fits(self, figname="ConvergenceCurve.pdf"):
        """plot the convergence curve.
        """
        fig, ax = plt.subplots()
        ax.plot(range(1, len(self.best_fits)+1), self.best_fits)
        ax.set_xlabel("Number of Generations")
        ax.set_ylabel("Fitness")
        fig.savefig(figname)
