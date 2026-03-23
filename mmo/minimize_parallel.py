#!/usr/bin/env python
"""Parallelized version of MultiModalMinimizer using joblib-enabled SSC."""

# std libs
import warnings
import importlib

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.patches import Polygon

import numpy as np
import numpy.linalg as la
import scipy.spatial.distance as ssd

# own
from mmo.domain import Domain
from mmo.config import Config
from mmo.solutions import Solutions
from mmo.function import Function
from mmo.cma import CMA
from mmo.ga_seed import Seed
from mmo.ssc_parallel import SeedSolveCollectParallel
from mmo.verbose import Verbose
from mmo.nelder_mead import NelderMead

###############################################################################
# class
###############################################################################
class MultiModalMinimizer_Iterator:
    def __init__(self, xy, n_fev, number, f, plot):
        self.xy = xy
        self.x = self.xy[:,:-1]
        self.y = self.xy[:,-1]
        self.n_fev = n_fev
        self.n_sol = self.xy.shape[0]
        self.number = number
        self.plot__ = plot
        self.f = f

    def __str__(self):
        s = f'iteration: {self.number}\n'
        s += f'  n_fev: {self.n_fev}\n'
        s += f'  n_sol: {self.n_sol}\n'
        return(s)

    def plot(self):
        self.plot__(f = self.f)
        
    def plot_with_s(self, seeds):
        self.plot__(seeds=seeds, f = self.f)

class MultiModalMinimizerParallel:
    """Parallelized multimodal minimization algorithm."""

    def __init__(self, f=None, domain=None, true_solutions=None, verbose=False, 
                 budget=np.inf, max_iter=np.inf, max_sol=np.inf, n_jobs=-1):
        """Initialize the parallel multimodal minimization algorithm.

        Parameters
        ----------
        f : function (mandatory)
            Function to map d-dimensional input (1d numpy array) to a floating point output.
        domain : domain (mandatory)
            Instance of the Domain class.
        true_solutions: 2d numpy array (optional)
            True solutions, one solution per row.
        verbose : 0, 1, 2, 3
            0: no output, >=1: summary output, >=2: details on seeds-GA, >=3: details on CMA-ES.
        budget : int (optional)
            Stopping criterion - function evaluation budget.
        max_iter: int (optional)
            Stopping criterion - maximum outer iterations.
        max_sol: int (optional)
            Stopping criterion - maximum solutions to find.
        n_jobs: int (optional)
            Number of parallel jobs (-1 = all cores). NEW PARAMETER!
        """

        assert(f is not None)
        assert(domain is not None)
        self.f = f
        self.domain = domain
        self.dim = domain.dim
        self.true_solutions = true_solutions
        self.xy = np.zeros((0, self.dim + 1))
        self.iteration = 0
        self.budget = budget
        self.max_iter = max_iter
        self.max_sol = max_sol
        self.n_fev = 0
        self.n_sol = 0
        self.verbose_1 = verbose >= 1
        self.verbose_2 = verbose >= 2
        self.verbose_3 = verbose >= 3
        self.n_jobs = n_jobs  # NEW

        # solver components
        self.config = Config(dim = self.dim, verbose = self.verbose_1)
        self.fct = Function(f = self.f, domain = self.domain, project=True)
        self.solutions = Solutions(domain = self.domain, true_solutions = self.true_solutions, plot = 1, verbose = self.verbose_1)
        if self.dim == 1:
            self.ls = NelderMead(f = self.fct, domain = self.domain, tol = 1e-6, verbose = self.verbose_3)
        else:
            self.ls = CMA(f = self.fct, domain = self.domain, popsize = self.config.cma_popsize, verbose = self.verbose_3)
        self.seed = Seed(f = self.fct, domain = self.domain, solutions = self.solutions, plot = 0, verbose = self.verbose_2, config = self.config)

        self.ssc = SeedSolveCollectParallel(
            f = self.fct,
            domain = self.domain,
            localsolver = self.ls,
            seed = self.seed,
            solutions = self.solutions,
            budget = np.inf,
            verbose = self.verbose_1,
            plot = 0,
            n_sol = -1,
            n_iter = 1,
            n_jobs = self.n_jobs,  # NEW
        )

    def __iter__(self):
        return(self)

    def __next__(self):
        """Next iteration

        Raises
        ------
        StopIteration
            If stopping criterion is met

        Returns
        -------
        r : MultiModalMinimizer_Iterator
        """

        if self.n_fev >= self.budget or self.iteration >= self.max_iter or self.n_sol >= self.max_sol:
            raise StopIteration

        self.xy = self.ssc.solve()
        self.n_fev = self.ssc.f.n_fev
        self.n_sol = self.xy.shape[0]
        self.iteration += 1

        return(MultiModalMinimizer_Iterator(self.xy, self.n_fev, self.iteration - 1, self.f, self.solutions.plot))
