#!/usr/bin/env python

# std libs
import warnings
import importlib

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import numpy.linalg as la
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from mmo.integrate import xy_add, xy_reduce_xequal_ymin

###############################################################################
# function class
###############################################################################
class CircleRegions:
    def __init__(self, domain=None, solutions=None):
        self.domain = domain
        self.dim = domain.dim
        self.points = None
        self.radius = None
        self.solutions = solutions

    def set(self, population, percentile=-1):
        self.radius = None
        self.points = None
        if population is None:
            return
        self.population = population[:,:self.dim]
        self.points = self.solutions.sol[:,:self.dim]
        if self.points is not None and self.points.shape[0]>0:
            if self.population is not None and self.population.shape[0]>0:
                self.radius = np.zeros(self.points.shape[0])
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.points)
                distances, indices = nbrs.kneighbors(self.population)
                indices = indices.reshape(-1)
                distances = distances.reshape(-1)
                for k in range(self.points.shape[0]):
                    idx = indices==k
                    if sum(idx):
                        self.radius[k] = np.percentile(distances[idx], percentile)
                    else:
                        self.radius[k] = 0

    def get(self):
        return(self.points, self.radius)

class Solutions():
    def __init__(self, domain = None, xtol = None, true_solutions = None, plot = False, verbose = False):
        self.domain = domain
        self.dim = domain.dim
        self.verbose = verbose
        if xtol is None:
            self.xtol = 1e-6 * domain.diameter
        else:
            self.xtol = xtol
        self.n_sol_previous = 0
        self.n_sol = 0
        self.n_duplicates = 0
        self.n_duplicates_previous = 0
        self.sol = np.zeros((0, self.dim+1))
        self.true_solutions = true_solutions
        self._plot = plot
        self._plot_seeds = False
        self.seeds = None
        self.radius = None
        self.regions = None
        self.n_true_solutions = 0
        if true_solutions is not None:
            self.true_solutions = true_solutions
            self.n_true_solutions = true_solutions[1]

        if self.verbose:
            print(self)

    def set_radius(self, population=None):
        percentile = 50
        self.regions = CircleRegions(domain=self.domain, solutions=self)
        self.regions.set(population, percentile=percentile)

    def store_previous(self):
        self.n_sol_previous = self.n_sol
        self.n_duplicates_previous = self.n_duplicates

    def add(self, x, y):
        if x is None: return
        suggested_xy = np.hstack((x.reshape(-1, self.dim), y.reshape(-1, 1)))
        n_suggested = suggested_xy.shape[0]
        n_old = self.sol.shape[0]
        self.sol = xy_add(self.sol, suggested_xy)
        self.sol = xy_reduce_xequal_ymin(self.sol, tol = self.xtol)
        n_new = self.sol.shape[0]
        self.n_sol = n_new
        self.n_duplicates += n_suggested + n_old - n_new

    def solution_in_range(self, x, radius):
        for k in range(self.sol.shape[0]):
            if la.norm(x - self.sol[k,:self.dim]) < radius:
                return(True)
        return(False)

    def plot(self, population=None, out='screen', f=None, seeds=None):
        
        if seeds is not None:
            self.seeds = seeds
            self._plot_seeds =True
        print("out", out, self._plot, self.dim)
        if self._plot == 0: return
        if self.dim == 1:
            self.plot_1d(out=out, population=population, f=f)
        elif self.dim == 2:
            #self.plot_2d(out=out, population=population)
            # Instead of plot_2d, we call plot_3d if we want a 3D surface
            self.plot_3d(population=population, out=out, f=f, mesh_res=50)
        else:
            self.plot_composite(out=out, f=f, population=population)
            #self.plot_parallel_coordinates(out=out, population=population)




    def plot_composite(self, population=None, out='screen', f=None, mesh_res=50):
        """
        Composite plot for high-dimensional problems:
        - Left subplot: 2D contour/scatter plot (using PCA projection)
        - Right subplot: 3D scatter plot (PCA x-y and f(x) as z)
        """
        if f is None:
            print("No function provided to evaluate f(x). Cannot plot.")
            return

        # Gather available data from the various datasets (all assumed to be of shape (n, d))
        data_list = []
        if population is not None and population.shape[0] > 0:
            data_list.append(population[:, :self.domain.dim])
        if self._plot_seeds and self.seeds is not None and self.seeds.shape[0] > 0:
            data_list.append(self.seeds[:, :self.domain.dim])
        if self.sol is not None and self.sol.shape[0] > 0:
            data_list.append(self.sol[:, :self.domain.dim])
        if (self.true_solutions is not None and 
            isinstance(self.true_solutions, np.ndarray) and 
            self.true_solutions.shape[0] > 0):
            data_list.append(self.true_solutions[:, :self.domain.dim])
            
        if len(data_list) > 0:
            all_data = np.vstack(data_list)
        else:
            # Fallback: use the domain center if no data is provided.
            center = np.array([(self.domain.ll[i] + self.domain.ur[i]) / 2 for i in range(self.domain.dim)])
            all_data = center.reshape(1, -1)

        # For high-dimensional data, use PCA to reduce to 2D.
        if self.domain.dim > 2:
            pca = PCA(n_components=2)
            pca.fit(all_data)
        else:
            pca = None

        # Create composite figure with two subplots: 2D (left) and 3D (right)
        fig = plt.figure(figsize=(16, 8))
        
        # ----- Left subplot: 2D contour and scatter plot -----
        ax2d = fig.add_subplot(121)
        ax2d.set_title("2D PCA Projection & Contour Map")

        # Create a grid in the PCA space. If using PCA, derive limits from the projected data.
        if pca is not None:
            projected_all = pca.transform(all_data)
            x_min, x_max = projected_all[:, 0].min(), projected_all[:, 0].max()
            y_min, y_max = projected_all[:, 1].min(), projected_all[:, 1].max()
        else:
            # Use the original 2D domain limits if data is 2D.
            x_min, x_max = self.domain.ll[0], self.domain.ur[0]
            y_min, y_max = self.domain.ll[1], self.domain.ur[1]
            
        # Add a margin to the grid limits
        x_margin = 0.1 * (x_max - x_min)
        y_margin = 0.1 * (y_max - y_min)
        x_grid = np.linspace(x_min - x_margin, x_max + x_margin, mesh_res)
        y_grid = np.linspace(y_min - y_margin, y_max + y_margin, mesh_res)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # If PCA is used, convert grid points back to the original space using inverse transform.
        if pca is not None:
            grid_points = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
            grid_orig = pca.inverse_transform(grid_points)
        else:
            grid_orig = np.column_stack((X_grid.ravel(), Y_grid.ravel()))

        # Clip grid points to domain bounds to avoid domain errors (e.g., log(x) with x<=0)
        grid_orig_clipped = np.clip(grid_orig, self.domain.ll, self.domain.ur)

        # Evaluate function on the grid (each point in original space)
        Z_grid = np.array([f.evaluate(point) for point in grid_orig_clipped])
        Z_grid = Z_grid.reshape(X_grid.shape)
        
        # Plot contour
        contour = ax2d.contourf(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.6)
        fig.colorbar(contour, ax=ax2d)
        
        # Helper function to plot scatter data on the 2D subplot.
        def plot_scatter_2d(ax, data_array, label, marker, color, size):
            if data_array is not None and data_array.shape[0] > 0:
                pts = data_array[:, :self.domain.dim]
                if pca is not None:
                    proj = pca.transform(pts)
                else:
                    proj = pts[:, :2]
                ax.scatter(proj[:, 0], proj[:, 1], c=color, marker=marker, s=size, label=label)
        
        plot_scatter_2d(ax2d, population, "Population", 'o', 'lightgrey', 20)
        if self._plot_seeds:
            plot_scatter_2d(ax2d, self.seeds, "Seeds", '^', 'blue', 40)
        plot_scatter_2d(ax2d, self.sol, "Solutions", '*', 'red', 50)
        plot_scatter_2d(ax2d, self.true_solutions, "True Solutions", 'X', 'orange', 60)
        
        ax2d.set_xlabel("PCA Component 1")
        ax2d.set_ylabel("PCA Component 2")
        ax2d.legend()

        # ----- Right subplot: 3D scatter plot with f(x) as z -----
        ax3d = fig.add_subplot(122, projection='3d')
        ax3d.set_title("3D Projection with f(x) Value")

        # Collect z-values for percentile-based z-axis limiting
        data_z = []

        # Helper function for 3D scatter (modified to return z-values)
        def plot_scatter_3d(ax, data_array, label, marker, color, size):
            if data_array is not None and data_array.shape[0] > 0:
                pts = data_array[:, :self.domain.dim]
                # Clip points to domain bounds to prevent math domain errors
                pts_clipped = np.clip(pts, self.domain.ll, self.domain.ur)
                if pca is not None:
                    proj = pca.transform(pts)
                else:
                    proj = pts[:, :2]
                # Use function value as z coordinate, evaluated in original space with clipped points
                zvals = np.array([f.evaluate(point) for point in pts_clipped])
                ax.scatter(proj[:, 0], proj[:, 1], zvals, c=color, marker=marker, s=size, label=label)
                return zvals
            return np.array([])

        # Plot and collect z-values
        z_pop = plot_scatter_3d(ax3d, population, "Population", 'o', 'lightgrey', 20)
        data_z.extend(z_pop)
        if self._plot_seeds:
            z_seeds = plot_scatter_3d(ax3d, self.seeds, "Seeds", '^', 'blue', 40)
            data_z.extend(z_seeds)
        z_sol = plot_scatter_3d(ax3d, self.sol, "Solutions", '*', 'red', 50)
        data_z.extend(z_sol)
        z_true = plot_scatter_3d(ax3d, self.true_solutions, "True Solutions", 'X', 'orange', 60)
        data_z.extend(z_true)

        # Set z-axis limits based on percentiles to focus on solution region
        if len(data_z) > 0:
            data_z = np.array(data_z)
            lower_z, upper_z = np.percentile(data_z, [5, 95])
            z_center = (lower_z + upper_z) / 2
            z_range = (upper_z - lower_z) * 1.2  # 20% margin for better visibility
            ax3d.set_zlim(z_center - z_range/2, z_center + z_range/2)

        ax3d.set_xlabel("PCA Component 1")
        ax3d.set_ylabel("PCA Component 2")
        ax3d.set_zlabel("f(x)")
        ax3d.legend()

        if out == 'screen':
            plt.show()
        else:
            plt.savefig(out)
            plt.close()


    def plot_3d(self, population=None, out='screen', f=None, mesh_res=50):
        """
        3D plot for the 2D domain: 
        - Plots a surface mesh of f(x,y) in the bounding box.
        - Plots population points, solutions, seeds, etc. as 3D scatter.
        Employs anomaly removal when computing the center and range to focus on 
        the main cluster of data.
        """
        if f is None:
            print("No function provided to evaluate z = f(x,y). Cannot plot 3D surface.")
            return

        # Prepare figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f.get_info()["name"] + " Solutions Visualization")

        # 1) Create a meshgrid over [ll[0], ur[0]] x [ll[1], ur[1]]
        x_vals = np.linspace(self.domain.ll[0], self.domain.ur[0], mesh_res)
        y_vals = np.linspace(self.domain.ll[1], self.domain.ur[1], mesh_res)
        X, Y = np.meshgrid(x_vals, y_vals)

        # 2) Evaluate f(x, y) on the grid
        Z = np.zeros_like(X)
        for i in range(mesh_res):
            for j in range(mesh_res):
                Z[i, j] = f.evaluate([X[i, j], Y[i, j]])

        # 3) Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=10, ax=ax)

        # 4) Plot population as 3D scatter (if provided)
        if population is not None and population.shape[0] > 0:
            px = population[:, 0]
            py = population[:, 1]
            # Clip to domain bounds before evaluation
            px_clipped = np.clip(px, self.domain.ll[0], self.domain.ur[0])
            py_clipped = np.clip(py, self.domain.ll[1], self.domain.ur[1])
            pz = np.array([f.evaluate([px_clipped[k], py_clipped[k]]) for k in range(len(px))])
            ax.scatter(px, py, pz, c='lightgrey', marker='o', s=20, label='Population')

        # 5) Plot seeds if requested
        if self._plot_seeds and self.seeds is not None and self.seeds.shape[0] > 0:
            sx = self.seeds[:, 0]
            sy = self.seeds[:, 1]
            # Clip to domain bounds before evaluation
            sx_clipped = np.clip(sx, self.domain.ll[0], self.domain.ur[0])
            sy_clipped = np.clip(sy, self.domain.ll[1], self.domain.ur[1])
            sz = np.array([f.evaluate([sx_clipped[k], sy_clipped[k]]) for k in range(len(sx))])
            ax.scatter(sx, sy, sz, c='blue', marker='^', s=40, label='Global Optima')

        # 6) Plot the solutions we have found (self.sol)
        if self.sol is not None and self.sol.shape[0] > 0:
            solx = self.sol[:, 0]
            soly = self.sol[:, 1]
            # Clip to domain bounds before evaluation
            solx_clipped = np.clip(solx, self.domain.ll[0], self.domain.ur[0])
            soly_clipped = np.clip(soly, self.domain.ll[1], self.domain.ur[1])
            solz = np.array([f.evaluate([solx_clipped[k], soly_clipped[k]]) for k in range(len(solx))])
            ax.scatter(solx, soly, solz, c='red', marker='*', s=50, label='Solutions')

        # 7) Plot known "true solutions" (global optima), if provided
        if (self.true_solutions is not None and
            isinstance(self.true_solutions, np.ndarray) and
            self.true_solutions.shape[0] > 0):
            tx = self.true_solutions[:, 0]
            ty = self.true_solutions[:, 1]
            # Clip to domain bounds before evaluation
            tx_clipped = np.clip(tx, self.domain.ll[0], self.domain.ur[0])
            ty_clipped = np.clip(ty, self.domain.ll[1], self.domain.ur[1])
            tz = np.array([f.evaluate([tx_clipped[k], ty_clipped[k]]) for k in range(len(tx))])
            ax.scatter(tx, ty, tz, c='orange', marker='X', s=60, label='True Solutions')

        # Set axis labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y)')

        # Gather x, y, and z coordinates from all scatter data to determine the main region
        data_x, data_y, data_z = [], [], []
        if population is not None and population.shape[0] > 0:
            data_x.extend(population[:, 0])
            data_y.extend(population[:, 1])
            pz = np.array([f.evaluate([population[k, 0], population[k, 1]]) for k in range(len(population))])
            data_z.extend(pz)
        if self._plot_seeds and self.seeds is not None and self.seeds.shape[0] > 0:
            data_x.extend(self.seeds[:, 0])
            data_y.extend(self.seeds[:, 1])
            sz = np.array([f.evaluate([self.seeds[k, 0], self.seeds[k, 1]]) for k in range(len(self.seeds))])
            data_z.extend(sz)
        if self.sol is not None and self.sol.shape[0] > 0:
            data_x.extend(self.sol[:, 0])
            data_y.extend(self.sol[:, 1])
            solz = np.array([f.evaluate([self.sol[k, 0], self.sol[k, 1]]) for k in range(len(self.sol))])
            data_z.extend(solz)
        if (self.true_solutions is not None and
            isinstance(self.true_solutions, np.ndarray) and
            self.true_solutions.shape[0] > 0):
            data_x.extend(self.true_solutions[:, 0])
            data_y.extend(self.true_solutions[:, 1])
            tz = np.array([f.evaluate([self.true_solutions[k, 0], self.true_solutions[k, 1]]) for k in range(len(self.true_solutions))])
            data_z.extend(tz)

        if len(data_x) > 0 and len(data_y) > 0:
            data_x = np.array(data_x)
            data_y = np.array(data_y)
            data_z = np.array(data_z)

            # Use 5th and 95th percentiles to remove extreme anomalies
            lower_x, upper_x = np.percentile(data_x, [5, 95])
            lower_y, upper_y = np.percentile(data_y, [5, 95])
            lower_z, upper_z = np.percentile(data_z, [5, 95])

            x_center = (lower_x + upper_x) / 2
            y_center = (lower_y + upper_y) / 2
            z_center = (lower_z + upper_z) / 2

            x_range = (upper_x - lower_x) * 1.1  # 10% extra margin
            y_range = (upper_y - lower_y) * 1.1
            z_range = (upper_z - lower_z) * 1.2  # 20% margin for z-axis for better visibility

            ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
            ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
            ax.set_zlim(z_center - z_range/2, z_center + z_range/2)

        ax.legend(loc='best')

        if out == 'screen':
            plt.show()
        else:
            plt.savefig(out)
            plt.close()


    def plot_1d(self, population=None, out='screen', f=None):
        fig, axs = plt.subplots(1, 1, figsize=(16,9))
        plt.xlim((self.domain.ll[0],self.domain.ur[0]))
        plt.title(f.get_info()["name"]+" Solutions Visualization")

        # function
        if f is not None:
            x = np.linspace(self.domain.ll[0], self.domain.ur[0], 1000) 
            #f.record(False)
            y = np.zeros(1000)
            for k in range(1000):
                y[k] = f.evaluate([x[k]])
            #f.record(True)
            axs.plot(x, y, c='green')

        # population
        if population is not None and population.shape[0]>0:
            x = population[:,0]
            y = population[:,1]
            if x is not None:
                axs.scatter(x, y, s=30, c='lightgrey')

        # plot solutions
        if self.true_solutions is not None and self.true_solutions.shape[0]>0:
            x = self.true_solutions
            if x is not None:
                #f.record(False)
                y = np.zeros(x.shape[0])
                for k in range(x.shape[0]):
                    y[k] = f(x[k])
                #f.record(True)
                axs.scatter(x[:,0], y, s=50, c='orange')

        # seeds
        if self._plot_seeds and self.seeds is not None and self.seeds.shape[0]>0:
            x = self.seeds[:,:self.dim]
            if x is not None:
                #f.record(False)
                y = np.zeros(x.shape[0])
                for k in range(x.shape[0]):
                    y[k] = f.evaluate(x[k].flatten())
                #f.record(True)
                axs.scatter(x[:,0], y, s=50, c='blue')

        # solutions
        if self.sol is not None and self.sol.shape[0]>0:
            x = self.sol[:,:self.dim]
            if x is not None:
                #f.record(False)
                y = np.zeros(x.shape[0])
                for k in range(x.shape[0]):
                    y[k] = f.evaluate([x[k]])
                #f.record(True)
                axs.scatter(x[:,0], y, s=20, c='red')

        # global solutions
        if self.sol.shape[0]>0:
            x = self.sol[:, :self.dim]
            print("x--", x)
            if x is not None:
                #f.record(False)
                y = np.zeros(x.shape[0])
                for k in range(x.shape[0]):
                    y[k] = f.evaluate([x[k]])
                #f.record(True)
                axs.scatter(x[:,0], y, s=20, c='green')

        if out=='screen': plt.show()
        else:
            plt.savefig(out)
            plt.close()

    def plot_2d(self, population=None, out='screen'):
        fig, axs = plt.subplots(1, 1, figsize=(9,9))
        plt.xlim((self.domain.ll[0],self.domain.ur[0]))
        plt.ylim((self.domain.ll[1],self.domain.ur[1]))
        plt.title("Solutions")

        # population
        if population is not None and population.shape[0]>0:
            x = population[:,:self.dim]
            if x is not None:
                axs.scatter(x[:,0], x[:,1], s=10, c='lightgrey')

        # plot solutions
        if self.true_solutions is not None and self.true_solutions.shape[0]>0:
            x = self.true_solutions
            if x is not None:
                axs.scatter(x[:,0], x[:,1], s=50, c='orange')

        # seeds
        if self._plot_seeds and self.seeds is not None and self.seeds.shape[0]>0:
            x = self.seeds[:,:self.dim]
            if x is not None:
                axs.scatter(x[:,0], x[:,1], s=50, c='blue')

        # solutions
        if self.sol is not None and self.sol.shape[0]>0:
            x = self.sol[:,:self.dim]
            if x is not None:
                axs.scatter(x[:,0], x[:,1], s=20, c='red')

        # global solutions
        if self.sol.shape[0]>0:
            x = self.sol[:, :self.dim]
            if x is not None:
                axs.scatter(x[:,0], x[:,1], s=20, c='green')

        if out=='screen': plt.show()
        else:
            plt.savefig(out)
            plt.close()

    def plot_parallel_coordinates(self, out='screen', population=None):

        # init
        colors = []
        linewidth= []
        x = np.zeros((0,self.dim))
        plt.figure(figsize=(12,9))
        columns = ['x%d' % k for k in range(x.shape[1])]

        # true solutions
        if self.true_solutions is not None and self.true_solutions.shape[0]>0:
            x = np.vstack((x, self.true_solutions[:,:self.dim]))
            colors += ['orange']*self.true_solutions.shape[0]
            df = pd.DataFrame(data=x, columns=columns)
            df['names'] = np.arange(x.shape[0])
            ax = pd.plotting.parallel_coordinates(df, 'names', cols=columns, color=colors, linewidth=8)
            ax.legend().remove()

        # population
        if population is not None and population.shape[0]>0:
            x = population[:,:self.dim]
            colors = ['lightgrey']*population.shape[0]
            df = pd.DataFrame(data=x, columns=columns)
            df['names'] = np.arange(x.shape[0])
            ax = pd.plotting.parallel_coordinates(df, 'names', cols=columns, color=colors, linewidth=1)
            ax.legend().remove()

        # population in radius
        if self.radius is not None:
            x = population[:,:self.dim]
            y = self.sol[:,:self.dim]
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(y)
            distances, indices = nbrs.kneighbors(x)
            distances = distances.reshape(-1)
            indices = indices.reshape(-1)
            radius = self.regions.radius[indices]
            idx = distances<radius

            x = population[idx,:self.dim]
            colors = ['grey']*population.shape[0]
            df = pd.DataFrame(data=x, columns=columns)
            df['names'] = np.arange(x.shape[0])
            ax = pd.plotting.parallel_coordinates(df, 'names', cols=columns, color=colors, linewidth=1)
            ax.legend().remove()

        # seeds
        if self._plot_seeds and self.seeds is not None and self.seeds.shape[0]>0:
            x = self.seeds[:,:self.dim]
            colors = ['blue']*self.seeds.shape[0]
            df = pd.DataFrame(data=x, columns=columns)
            df['names'] = np.arange(x.shape[0])
            ax = pd.plotting.parallel_coordinates(df, 'names', cols=columns, color=colors, linewidth=1)
            ax.legend().remove()

        # solutions
        if self.sol.shape[0]>0:
            x = self.sol[:,:self.dim]
            colors = ['red']*self.sol.shape[0]
            df = pd.DataFrame(data=x, columns=columns)
            df['names'] = np.arange(x.shape[0])
            ax = pd.plotting.parallel_coordinates(df, 'names', cols=columns, color=colors, linewidth=2)
            ax.legend().remove()

        if out=='screen': plt.show()
        else:
            plt.savefig(out)
            plt.close()

    def __str__(self):
        s = 'solutions\n'
        if self.true_solutions is not None:
            s += f'  n_true_solutions: {self.n_true_solutions}\n'
        s += f'  x_tol: {self.xtol}\n'
        s += f'  n_sol: {self.n_sol} ({self.n_sol_previous})\n'
        s += f'  n_duplicates: {self.n_duplicates} ({self.n_duplicates_previous})\n'

        return(s)

    def min_dist_to_solution(self, population):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.solutions.sol)
        distances, indices = nbrs.kneighbors(population[:, :self.dim])
        distances = np.min(distances, axis=1).reshape(-1)
        return(distances)

    def population_wrt_regions(self, population):
        self.set_radius(population=population)
        if self.regions is None or self.regions.points is None or self.regions.points.shape[0] == 0:
            return(None, None)

        nbrs = NearestNeighbors(n_neighbors=self.regions.points.shape[0], algorithm='ball_tree').fit(self.regions.points)
        distances, indices = nbrs.kneighbors(population[:,:self.dim])
        idx = np.any(distances < self.regions.radius[indices], axis=1)
        idx_in = np.where(idx==True)[0].tolist()
        idx_out = np.where(idx==False)[0].tolist()
        return(idx_in, idx_out)











