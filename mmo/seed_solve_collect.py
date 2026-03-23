import numpy as np

class SeedSolveCollect:
    def __init__(self, f, domain, localsolver, seed, solutions, budget=np.inf, verbose=0, plot=0, n_sol=-1, n_iter=1):
        self.f = f
        self.domain = domain
        self.localsolver = localsolver
        self.seed = seed
        self.solutions = solutions
        self.budget = budget
        self.verbose = verbose
        self.plot = plot
        self.n_sol = n_sol
        self.n_iter = n_iter

    def run(self):
        """Run the seed-solve-collect process."""
        seeds = self.seed.generate(self.n_sol)
        fitness = self.seed.evaluate(seeds)
        for i in range(self.n_iter):
            self.localsolver.optimize(seeds, fitness)
            self.solutions.add(seeds, fitness)