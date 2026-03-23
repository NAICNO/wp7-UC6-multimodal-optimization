#!/usr/bin/env python

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
from mmo.ssc import SeedSolveCollect
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

class MultiModalMinimizer:
    """Multimodal minimization algorithm provided as iteratable class."""

    def __init__(self, f = None, domain = None, true_solutions = None, verbose = False, budget = np.inf, max_iter = np.inf, max_sol = np.inf):
        """Initialize the multimodal minimization algorithm.

        Mandatorily required are a function to minimize and a domain in which multiple local or global minimizers are sought.

        Parameters
        ----------
        f : function (mandatory)
            Function to map d-dimensional input (1d numpy array) to a floating point output.
        domain : domain (mandatory)
            Instance of the Domain class, dimensionality of the domain determines the dinensionality of the function input.
        true_solutions: 2d numpy array (optional)
            True solutions, one solution per row. Used to plot found vs all solutions.
        verbose : 0, 1, 2, 3
            0: no output, >=1: summary output, >=2: details on the seeds-GA, >=3: details on the CMA-ES local solves .
        budget : int (optional)
            stopping criterion, if provided stops iteration if number of fct evaluations exceeds budget.
        max_iter: int (optional)
            stopping criterion, if provided stops iteration if number of outer iterations reaches max_iter.
        max_sol: int (optional)
            stopping criterion, if provided stops iteration if number of solutions found reaches max_sol.
            
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

        # solver components
        self.config = Config(dim = self.dim, verbose = self.verbose_1)
        self.fct = Function(f = self.f, domain = self.domain, project=True)
        self.solutions = Solutions(domain = self.domain, true_solutions = self.true_solutions, plot = 1, verbose = self.verbose_1)
        if self.dim == 1:
            self.ls = NelderMead(f = self.fct, domain = self.domain, tol = 1e-6, verbose = self.verbose_3)
        else:
            self.ls = CMA(f = self.fct, domain = self.domain, popsize = self.config.cma_popsize, verbose = self.verbose_3)
        self.seed = Seed(f = self.fct, domain = self.domain, solutions = self.solutions, plot = 0, verbose = self.verbose_2, config = self.config)

        self.ssc = SeedSolveCollect(
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
        )


    def __iter__(self):
        return(self)

    def __next__(self):
        """Next iteration

        Raises
        ------
        StopIteration
            If stopping one of the criteria is met

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



