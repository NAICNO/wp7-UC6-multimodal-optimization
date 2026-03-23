import numpy as np

class Seed:
    def __init__(self, f, domain, solutions, plot=0, verbose=0, config=None):
        self.f = f
        self.domain = domain
        self.solutions = solutions
        self.plot = plot
        self.verbose = verbose
        self.config = config

    def generate(self, n):
        """Generate n seed points within the domain."""
        seeds = np.random.uniform(self.domain.lower_bound, self.domain.upper_bound, (n, self.domain.dim))
        return seeds

    def evaluate(self, seeds):
        """Evaluate the seed points."""
        fitness = np.apply_along_axis(self.f, 1, seeds)
        return fitness