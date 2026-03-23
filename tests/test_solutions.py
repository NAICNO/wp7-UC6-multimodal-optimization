"""
test_solutions.py — Unit tests for mmo.solutions.Solutions

Covers:
- Construction: default xtol, zero initial solutions, dim, n_sol
- add(): single point, deduplication within tolerance, multiple distinct points
- solution_in_range(): point within / outside radius
- __str__ representation
- store_previous() tracks n_sol_previous
"""

import numpy as np
import pytest

from mmo.domain import Domain
from mmo.solutions import Solutions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_solutions(dim=2, xtol=None):
    domain = Domain(boundary=[[0.0] * dim, [1.0] * dim])
    return Solutions(domain=domain, xtol=xtol), domain


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestSolutionsConstruction:

    def test_initial_n_sol_is_zero(self):
        s, _ = _make_solutions()
        assert s.n_sol == 0

    def test_initial_sol_array_shape(self):
        s, _ = _make_solutions(dim=2)
        assert s.sol.shape == (0, 3)  # dim + 1

    def test_dim_stored_correctly(self):
        s, _ = _make_solutions(dim=3)
        assert s.dim == 3

    def test_default_xtol_is_fraction_of_diameter(self):
        s, domain = _make_solutions()
        assert s.xtol == pytest.approx(1e-6 * domain.diameter)

    def test_custom_xtol_stored(self):
        s, _ = _make_solutions(xtol=0.1)
        assert s.xtol == pytest.approx(0.1)

    def test_n_duplicates_starts_at_zero(self):
        s, _ = _make_solutions()
        assert s.n_duplicates == 0


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------

class TestSolutionsAdd:

    def test_add_single_solution_increases_n_sol(self):
        s, _ = _make_solutions()
        x = np.array([0.5, 0.5])
        y = np.array([1.0])
        s.add(x, y)
        assert s.n_sol == 1

    def test_add_none_x_is_no_op(self):
        s, _ = _make_solutions()
        s.add(None, np.array([1.0]))
        assert s.n_sol == 0

    def test_add_two_distinct_solutions(self):
        s, _ = _make_solutions(xtol=0.01)
        s.add(np.array([0.1, 0.1]), np.array([0.5]))
        s.add(np.array([0.9, 0.9]), np.array([0.3]))
        assert s.n_sol == 2

    def test_add_duplicate_x_deduplicates(self):
        s, _ = _make_solutions(xtol=0.1)
        x = np.array([0.5, 0.5])
        s.add(x, np.array([5.0]))
        s.add(x, np.array([2.0]))  # same location, lower y
        assert s.n_sol == 1

    def test_add_duplicate_keeps_lower_y(self):
        s, _ = _make_solutions(xtol=0.1)
        x = np.array([0.5, 0.5])
        s.add(x, np.array([5.0]))
        s.add(x, np.array([2.0]))
        assert s.sol[0, -1] == pytest.approx(2.0)

    def test_add_increments_n_duplicates(self):
        s, _ = _make_solutions(xtol=0.1)
        x = np.array([0.5, 0.5])
        s.add(x, np.array([5.0]))
        s.add(x, np.array([2.0]))
        assert s.n_duplicates > 0

    def test_add_sol_shape_correct_after_two_distinct(self):
        s, _ = _make_solutions(dim=2, xtol=0.01)
        s.add(np.array([0.1, 0.1]), np.array([1.0]))
        s.add(np.array([0.9, 0.9]), np.array([2.0]))
        assert s.sol.shape == (2, 3)


# ---------------------------------------------------------------------------
# solution_in_range()
# ---------------------------------------------------------------------------

class TestSolutionInRange:

    def _setup_with_solution(self, sol_x=None, sol_y=1.0, xtol=0.01):
        s, _ = _make_solutions(dim=2, xtol=xtol)
        if sol_x is None:
            sol_x = np.array([0.5, 0.5])
        s.add(sol_x, np.array([sol_y]))
        return s

    def test_point_within_radius_returns_true(self):
        s = self._setup_with_solution()
        assert s.solution_in_range(np.array([0.5, 0.5]), radius=0.1) == True

    def test_point_outside_radius_returns_false(self):
        s = self._setup_with_solution()
        assert s.solution_in_range(np.array([0.9, 0.9]), radius=0.1) == False

    def test_no_solutions_always_returns_false(self):
        s, _ = _make_solutions()
        assert s.solution_in_range(np.array([0.5, 0.5]), radius=1.0) == False

    def test_exact_location_returns_true(self):
        s = self._setup_with_solution()
        assert s.solution_in_range(np.array([0.5, 0.5]), radius=0.001) == True

    def test_multiple_solutions_one_in_range(self):
        s, _ = _make_solutions(dim=2, xtol=0.01)
        s.add(np.array([0.1, 0.1]), np.array([1.0]))
        s.add(np.array([0.9, 0.9]), np.array([2.0]))
        # Query close to first solution
        assert s.solution_in_range(np.array([0.12, 0.12]), radius=0.1) == True
        # Query close to second solution
        assert s.solution_in_range(np.array([0.88, 0.88]), radius=0.1) == True
        # Query far from both
        assert s.solution_in_range(np.array([0.5, 0.5]), radius=0.1) == False


# ---------------------------------------------------------------------------
# store_previous
# ---------------------------------------------------------------------------

class TestStorePrevious:

    def test_store_previous_saves_n_sol(self):
        s, _ = _make_solutions(xtol=0.01)
        s.add(np.array([0.5, 0.5]), np.array([1.0]))
        s.store_previous()
        s.add(np.array([0.1, 0.1]), np.array([2.0]))
        assert s.n_sol_previous == 1
        assert s.n_sol == 2


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------

class TestSolutionsStr:

    def test_str_contains_n_sol(self):
        s, _ = _make_solutions()
        assert 'n_sol' in str(s)

    def test_str_contains_x_tol(self):
        s, _ = _make_solutions()
        assert 'tol' in str(s).lower() or 'xtol' in str(s).lower()

    def test_str_is_string(self):
        s, _ = _make_solutions()
        assert isinstance(str(s), str)
