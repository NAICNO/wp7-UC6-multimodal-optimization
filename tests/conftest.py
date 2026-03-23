"""
Shared pytest fixtures for the MMO unit test suite.

Provides reusable Domain objects, synthetic callables, and small population arrays
so individual test modules stay concise and avoid repeated setup boilerplate.
"""

import numpy as np
import pytest

from mmo.domain import Domain
from mmo.config import Config


# ---------------------------------------------------------------------------
# Domain fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def domain_1d():
    """Unit interval [0, 1] in 1D."""
    return Domain(boundary=[[0.0], [1.0]])


@pytest.fixture
def domain_2d():
    """Unit square [0,1]^2 in 2D."""
    return Domain(boundary=[[0.0, 0.0], [1.0, 1.0]])


@pytest.fixture
def domain_2d_asymmetric():
    """Asymmetric 2D domain [-2, 3] x [-1, 4]."""
    return Domain(boundary=[[-2.0, -1.0], [3.0, 4.0]])


@pytest.fixture
def domain_3d():
    """Unit cube [0,1]^3 in 3D."""
    return Domain(boundary=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


@pytest.fixture
def domain_5d():
    """Hypercube [0,1]^5 in 5D."""
    ll = [0.0] * 5
    ur = [1.0] * 5
    return Domain(boundary=[ll, ur])


@pytest.fixture
def domain_negative():
    """Domain with negative coordinates: [-5, -1] x [-5, -1]."""
    return Domain(boundary=[[-5.0, -5.0], [-1.0, -1.0]])


# ---------------------------------------------------------------------------
# Population / point array fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_pop_2d(domain_2d):
    """Small population of 5 points inside the 2D unit square."""
    np.random.seed(42)
    x = np.random.rand(5, 2)
    y = np.sum(x ** 2, axis=1).reshape(-1, 1)
    return np.hstack((x, y))


@pytest.fixture
def single_point_2d():
    """A single 2D point as shape-(1, 2) array."""
    return np.array([[0.3, 0.7]])


@pytest.fixture
def points_on_boundary_2d():
    """Points on the boundary of the unit square."""
    return np.array([
        [0.0, 0.0],  # corner LL
        [1.0, 1.0],  # corner UR
        [0.0, 0.5],  # left edge
        [1.0, 0.5],  # right edge
        [0.5, 0.0],  # bottom edge
        [0.5, 1.0],  # top edge
    ])


# ---------------------------------------------------------------------------
# Synthetic callable fixtures (duck-typed to match Function.f interface)
# ---------------------------------------------------------------------------

class _QuadraticCallable:
    """Simple x^T x callable with an .evaluate method for Function wrapping."""

    def evaluate(self, x):
        x = np.asarray(x).reshape(-1)
        return float(np.dot(x, x))


class _ConstantCallable:
    """Callable that always returns a fixed constant."""

    def __init__(self, value=1.0):
        self.value = value

    def evaluate(self, x):
        return float(self.value)


class _LinearCallable:
    """Callable that returns the sum of coordinates."""

    def evaluate(self, x):
        x = np.asarray(x).reshape(-1)
        return float(np.sum(x))


@pytest.fixture
def quadratic_callable():
    """Returns an object with .evaluate(x) = |x|^2."""
    return _QuadraticCallable()


@pytest.fixture
def constant_callable():
    """Returns an object with .evaluate(x) = 1.0."""
    return _ConstantCallable(value=1.0)


@pytest.fixture
def linear_callable():
    """Returns an object with .evaluate(x) = sum(x)."""
    return _LinearCallable()


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_2d():
    """Standard Config for dim=2."""
    return Config(dim=2, profile='standard')


@pytest.fixture
def config_1d():
    """Standard Config for dim=1."""
    return Config(dim=1, profile='standard')


@pytest.fixture
def config_5d():
    """Standard Config for dim=5."""
    return Config(dim=5, profile='standard')
