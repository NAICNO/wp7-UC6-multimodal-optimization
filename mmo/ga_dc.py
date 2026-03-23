#!/usr/bin/env python

# std libs
import warnings
import importlib
import copy

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.patches import Polygon

import numpy as np
import numpy.linalg as la
import random
from operator import attrgetter
from functools import reduce
from operator import mul
import scipy.spatial.distance as ssd
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import NearestNeighbors

# deap (installed via pip, LGPL-3.0 license)
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap.algorithms import *

# own
from mmo.gatools import remove_identical
from mmo.seed_result import SeedResult
import itertools
from mmo.q_pol import has_minimum

###############################################################################
# functions
###############################################################################
def neighborhood(dim):
    x = np.array(list(itertools.product(*zip([-1]*dim, [0]*dim, [1]*dim))))
    s = la.norm(x, axis=1)
    idx = s!=0
    x = x[idx]
    s = s[idx]
    x = x/s.reshape(-1,1)
    return(x)

###############################################################################
# toolbox functions
###############################################################################
def create_individual_from_nparray(individual_class, x):
        return(individual_class(x))

def create_population_from_nparray(f, x):
    l = [f(x[i,:]) for i in range(x.shape[0])]
    return(l)

def population_to_nparray(p):
    return(np.array(p))

def eval_f_wrapper(eval_f, population):
    e = eval_f(np.array(population))
    for ind, f in zip(population, e):
        ind.fitness.values = (f,)
    return(len(population))

def assign_fitness(population, fitness):
    for ind, f in zip(population, fitness):
        ind.fitness.values = (f,)
    return(len(population))

def f0_to_nparray(p):
    n = len(p)
    f = np.zeros(n)
    for k,ind in zip(range(n),p):
        f[k] = ind.fitness.values[0]
    return(f)

def population2xy(population):
    x = population_to_nparray(population)
    y = f0_to_nparray(population).reshape(-1,1)
    xy = np.hstack((x,y))
    return(xy)

###############################################################################
# class
###############################################################################
class GA_DC:
    def __init__(self, f=None, domain=None, pop=None, verbose=0, config=None):
        assert(f.dim==domain.dim)
        self.dim = f.dim
        self.domain = domain

        # ga specific
        self.n_pop = pop.shape[0]
        self.eval_f = f
        self.CXPB = 0.3
        self.CXIDX = 10
        self.MUTPB = 0.3
        self.MUTIDX = 10
        self.verbose=verbose
        self.MU = self.n_pop
        self.MUTLOW = domain.ll.tolist()
        self.MUTUP = domain.ur.tolist()
        self.config = config

        # output
        if self.verbose:
            print("--- GA ---")

        # setup toolbox
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.random)
        self.toolbox.register("individual", create_individual_from_nparray, creator.Individual)
        self.toolbox.register("population", create_population_from_nparray, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=self.CXIDX, low=self.MUTLOW, up=self.MUTUP)
        self.toolbox.register("mutate", tools.mutPolynomialBounded, eta=self.MUTIDX, low=self.MUTLOW, up=self.MUTUP, indpb=1)
        self.toolbox.register("evaluate_f", eval_f_wrapper, self.eval_f)
        self.toolbox.register("assign_fitness", assign_fitness)
        self.toolbox.register("remove", remove_identical)

        # setup statistics
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)

        # create population and evaluate
        self._population = self.toolbox.population(pop[:,:self.dim])
        self._initial_population = self._population
        self.toolbox.assign_fitness(self._population, pop[:,self.dim])

        # output
        if self.verbose:
            self.info(0)

    def iter(self, n=100, convergence=None):

        def prob_decision(f1, f2):
            r = random.random()
            if r <= self.config.deterministic:
                return(f1 < f2)
            else:
                r = random.random()
                return(r<0.5)

        for g in range(1,n+1):

            # randomize population
            random.shuffle(self._population)

            # select
            offspring = list(map(self.toolbox.clone, self._population))

            # crossover
            for i in range(1, len(offspring), 2):
                if random.random() < self.CXPB:
                    offspring[i-1], offspring[i] = self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

            # mutation
            for i in range(len(offspring)):
                if random.random() < self.MUTPB:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # evaluate f the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self.toolbox.evaluate_f(invalid_ind)

            # selection (deterministic crowding)
            population_new = []
            for i in range(1, len(offspring), 2):
                # individuals
                p1 = self._population[i-1]
                p2 = self._population[i]
                c1 = offspring[i-1]
                c2 = offspring[i]

                # fitnesses 
                f_p1 = p1.fitness.values[0]
                f_p2 = p2.fitness.values[0]
                f_c1 = c1.fitness.values[0]
                f_c2 = c2.fitness.values[0]

                # select
                d1 = la.norm(np.array(p1)-np.array(c1)) + la.norm(np.array(p2)-np.array(c2))
                d2 = la.norm(np.array(p1)-np.array(c2)) + la.norm(np.array(p2)-np.array(c1))
                if d1 <= d2:
                    if prob_decision(f_c1, f_p1):
                        population_new += [c1]
                    else:
                        population_new += [p1]
                    if prob_decision(f_c2, f_p2):
                        population_new += [c2]
                    else:
                        population_new += [p2]
                else:
                    if prob_decision(f_c1, f_p2):
                        population_new += [c1]
                    else:
                        population_new += [p2]
                    if prob_decision(f_c2, f_p1):
                        population_new += [c2]
                    else:
                        population_new += [p1]

            self._population = population_new

            if convergence is not None:
                convergence.add(population=self._population)

            # output
            if self.verbose:
                self.info(g)

    def iter_dc(self, n=100):
        for g in range(1,n+1):

            # randomize population
            random.shuffle(self._population)

            # select
            offspring = list(map(self.toolbox.clone, self._population))

            # crossover
            for i in range(1, len(offspring), 2):
                if random.random() < self.CXPB:
                    offspring[i-1], offspring[i] = self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

            # mutation
            for i in range(len(offspring)):
                if random.random() < self.MUTPB:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # evaluate f the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self.toolbox.evaluate_f(invalid_ind)

            # selection (deterministic crowding)
            population_new = []
            for i in range(1, len(offspring), 2):
                # individuals
                p1 = self._population[i-1]
                p2 = self._population[i]
                c1 = offspring[i-1]
                c2 = offspring[i]

                # fitnesses 
                f_p1 = p1.fitness.values[0]
                f_p2 = p2.fitness.values[0]
                f_c1 = c1.fitness.values[0]
                f_c2 = c2.fitness.values[0]

                # select
                d1 = la.norm(np.array(p1)-np.array(c1)) + la.norm(np.array(p2)-np.array(c2))
                d2 = la.norm(np.array(p1)-np.array(c2)) + la.norm(np.array(p2)-np.array(c1))
                if d1 <= d2:
                    if f_c1 < f_p1: 
                        population_new += [c1]
                    else:     
                        population_new += [p1]
                    if f_c2 < f_p2:
                        population_new += [c2]
                    else:
                        population_new += [p2]
                else:
                    if f_c1 < f_p2:
                        population_new += [c1]
                    else: 
                        population_new += [p2]
                    if f_c2 < f_p1:
                        population_new += [c2]
                    else:
                        population_new += [p1]

            self._population = population_new

            # output
            if self.verbose:
                self.info(g)

    def seeds(self, nb_scale = 1):
        if self.config.quadratic_fit:
            return(self.seeds_q_fit())
        else:
            return(self.seeds_std(nb_scale = nb_scale))

    def seeds_q_fit(self):
        # get nearest neighbors
        N = max(self.config.seed_n_nb1, self.config.seed_n_nb2)
        population = self.population()
        f_values = self.f0()
        nbrs = NearestNeighbors(n_neighbors=1+N, algorithm='ball_tree').fit(population)
        distances, indices = nbrs.kneighbors(population)

        # indices of individuals being mimina in population neighborhood of size self.config.seed_n_nb1
        idx = np.where(np.all(f_values[indices[:,0]].reshape(-1,1) <= f_values[indices[:,1:(self.config.seed_n_nb1 + 1)]], axis=1))[0]
        idx_seeds = []
        for k in range(idx.shape[0]):
            seed_and_nb = population[indices[idx[k]]]
            f_values_seed_and_nb = f_values[indices[idx[k]]]
            if has_minimum(seed_and_nb, f_values_seed_and_nb):
                idx_seeds += [idx[k]]
                
        seeds = population[idx_seeds]
        spread = np.max(distances[idx_seeds], axis=1)

        self.cluster_seeds = seeds
        return(seeds, spread)

    def seeds_std(self, nb_scale = 1):
        # get nearest neighbors
        N = int (nb_scale * self.config.seed_n_nb1)
        population = self.population()
        f_values = self.f0()
        nbrs = NearestNeighbors(n_neighbors=1+N, algorithm='ball_tree').fit(population)
        distances, indices = nbrs.kneighbors(population)
        distances = distances[:,1:]

        if reduce(mul, distances.shape) == 0:
            return(None, None, None)

        # mimina in population neighborhood
        idx_min = np.all(f_values[indices[:,0]].reshape(-1,1) <= f_values[indices[:,1:]], axis=1)
        seeds = population[idx_min]
        f_seeds = f_values[idx_min]
        n_seeds = seeds.shape[0]
        distances = distances[idx_min]
        spread = np.max(distances, axis=1)

        # check if seeds are minima in nb neighborhood
        nb = self.config.nb
        corr_matrices = []

        # fixed neighborhood 
        if nb is not None:
            idx_min = []
            for k in range(n_seeds):
                seed = seeds[k]
                seed_nb = seed + spread[k] * nb
                idx = self.domain.is_in(seed_nb)
                seed_nb = seed_nb[idx]
                y_nb = self.eval_f(seed_nb)
                if np.all(f_seeds[k] < y_nb):
                    idx_min += [k]
            seeds = seeds[idx_min]
            spread = spread[idx_min]

        # dynamic neighborhood
        if nb is None:
            nb_all_seeds = population[indices[idx_min,1:]] - seeds.reshape(seeds.shape[0], 1, seeds.shape[1])
            idx_min = []
            for k in range(n_seeds):
                seed = seeds[k]
                seed_nb = np.vstack((seed + nb_all_seeds[k], seed - nb_all_seeds[k]))
                C = nb_all_seeds[k]
                C = C.T @ C
                w, v = la.eig(C)
                C = C / la.norm(C, ord=2) 
                idx = self.domain.is_in(seed_nb)
                seed_nb = seed_nb[idx]
                y_nb = self.eval_f(seed_nb)
                if np.all(f_seeds[k] < y_nb):
                    idx_min += [k]
                    corr_matrices += [C]
            seeds = seeds[idx_min]
            spread = spread[idx_min]

        # reformat centroid
        if len(corr_matrices) == 0:
            corr_matrices = None
        else:
            corr_matrices = np.array(corr_matrices)

        return(seeds, spread, corr_matrices)

    def population(self):
        return(population_to_nparray(self._population))

    def xy(self):
        return(population2xy(self._population))

    def initial_population(self):
        return(population_to_nparray(self._initial_population))

    def f0(self):
        return(f0_to_nparray(self._population))

    def xfr(self):
        x = self.population()
        y = self.fitness()[:,1]

        h = np.mean(ssd.pdist(x, metric='euclidean'))/10
        kr = KernelRidge(alpha=0.01, kernel='rbf', gamma=1/h/h)
        kr.fit(x,y)

        n_res = 100
        x_lin = np.linspace(self.MUTLOW[0], self.MUTUP[0], n_res)
        y_lin = np.linspace(self.MUTLOW[1], self.MUTUP[1], n_res)
        X, Y = np.meshgrid(x_lin, y_lin)
        xy = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
        f_kr = kr.predict(xy)

        p = np.hstack((xy,f_kr.reshape(-1,1)))
        return(p)

    def info(self, k):
        f_values = np.sort(self.f0())

        f_min = f_values[0]
        f_max = f_values[-1]
        n_eval = self.eval_f.n_fev

        print("%.3d: eval = [%d], f = [%.3e %.3e]" % (k, n_eval, f_min, f_max))

    def plot(self, out=None):
        if self.dim==2:
            self.plot2d(out=out)

    def plot2d(self, out=None):
        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        n_res = 100

        # plot 0: countour values with initial population
        x_lin = np.linspace(self.domain.ll[0], self.domain.ur[0], n_res)
        y_lin = np.linspace(self.domain.ll[1], self.domain.ur[1], n_res)
        X, Y = np.meshgrid(x_lin, y_lin)
        xy = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
        f_values = self.eval_f(xy)
        F_values = f_values.reshape(n_res, n_res)
        axs[0].contour(X, Y, F_values)
        x = self.initial_population()
        axs[0].scatter(x[:,0], x[:,1], s=10, c='b')

        # plot 1: countour values with population
        x_lin = np.linspace(self.domain.ll[0], self.domain.ur[0], n_res)
        y_lin = np.linspace(self.domain.ll[1], self.domain.ur[1], n_res)
        X, Y = np.meshgrid(x_lin, y_lin)
        xy = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
        f_values = self.eval_f(xy)
        F_values = f_values.reshape(n_res, n_res)
        axs[1].contour(X, Y, F_values)
        x = self.population()
        axs[1].scatter(x[:,0], x[:,1], s=10, c='b')
        x = self.cluster_seeds
        axs[1].scatter(x[:,0], x[:,1], s=10, c='r')

        # titles
        axs[0].set_title('f, initial population')
        axs[1].set_title('f, final population, seed points')

        if out is None:
            plt.show()
        else:
            plt.savefig(out)
            plt.close()






