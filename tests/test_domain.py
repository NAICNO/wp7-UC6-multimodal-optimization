"""
test_domain.py — Unit tests for mmo.domain.Domain

Covers:
- Construction and property correctness (ll, ur, center, dim, diameter)
- project_into_domain: interior, on-boundary, and exterior points
- is_in: single point, batches, boundary, outside, edge cases
- __str__ representation
- Invalid / edge case construction guard
"""

import numpy as np
import numpy.linalg as la
import pytest

from mmo.domain import Domain


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestDomainConstruction:

    def test_1d_properties(self, domain_1d):
        d = domain_1d
        assert d.dim == 1
        np.testing.assert_array_equal(d.ll, [0.0])
        np.testing.assert_array_equal(d.ur, [1.0])
        np.testing.assert_array_almost_equal(d.center, [0.5])
        assert d.diameter == pytest.approx(1.0)

    def test_2d_properties(self, domain_2d):
        d = domain_2d
        assert d.dim == 2
        np.testing.assert_array_almost_equal(d.center, [0.5, 0.5])
        assert d.diameter == pytest.approx(np.sqrt(2))

    def test_asymmetric_2d_center(self, domain_2d_asymmetric):
        d = domain_2d_asymmetric
        np.testing.assert_array_almost_equal(d.center, [0.5, 1.5])

    def test_asymmetric_2d_diameter(self, domain_2d_asymmetric):
        d = domain_2d_asymmetric
        expected = la.norm(np.array([3.0, 4.0]) - np.array([-2.0, -1.0]))
        assert d.diameter == pytest.approx(expected)

    def test_3d_dim(self, domain_3d):
        assert domain_3d.dim == 3

    def test_3d_diameter(self, domain_3d):
        assert domain_3d.diameter == pytest.approx(np.sqrt(3))

    def test_5d_dim(self, domain_5d):
        assert domain_5d.dim == 5

    def test_domain_stores_numpy_arrays(self, domain_2d):
        assert isinstance(domain_2d.ll, np.ndarray)
        assert isinstance(domain_2d.ur, np.ndarray)
        assert isinstance(domain_2d.center, np.ndarray)

    def test_negative_coordinate_domain(self, domain_negative):
        d = domain_negative
        np.testing.assert_array_almost_equal(d.center, [-3.0, -3.0])

    def test_construction_with_list_input(self):
        d = Domain(boundary=[[1, 2], [3, 4]])
        assert d.dim == 2
        np.testing.assert_array_equal(d.ll, [1, 2])
        np.testing.assert_array_equal(d.ur, [3, 4])

    def test_degenerate_domain_raises(self):
        """ll == ur should raise an AssertionError."""
        with pytest.raises((AssertionError, Exception)):
            Domain(boundary=[[0.0, 0.0], [0.0, 1.0]])


# ---------------------------------------------------------------------------
# project_into_domain tests
# ---------------------------------------------------------------------------

class TestProjectIntoDomain:

    def test_interior_point_unchanged(self, domain_2d):
        x = np.array([0.3, 0.7])
        result = domain_2d.project_into_domain(x)
        np.testing.assert_array_almost_equal(result, x)

    def test_center_point_unchanged(self, domain_2d):
        x = domain_2d.center.copy()
        result = domain_2d.project_into_domain(x)
        np.testing.assert_array_almost_equal(result, x)

    def test_point_below_ll_projected_to_ll(self, domain_2d):
        x = np.array([-1.0, -1.0])
        result = domain_2d.project_into_domain(x)
        np.testing.assert_array_almost_equal(result, domain_2d.ll)

    def test_point_above_ur_projected_to_ur(self, domain_2d):
        x = np.array([2.0, 2.0])
        result = domain_2d.project_into_domain(x)
        np.testing.assert_array_almost_equal(result, domain_2d.ur)

    def test_partial_out_of_bounds(self, domain_2d):
        x = np.array([-0.5, 0.5])
        result = domain_2d.project_into_domain(x)
        np.testing.assert_array_almost_equal(result, [0.0, 0.5])

    def test_on_boundary_ll_unchanged(self, domain_2d):
        x = domain_2d.ll.copy()
        result = domain_2d.project_into_domain(x)
        np.testing.assert_array_almost_equal(result, domain_2d.ll)

    def test_on_boundary_ur_unchanged(self, domain_2d):
        x = domain_2d.ur.copy()
        result = domain_2d.project_into_domain(x)
        np.testing.assert_array_almost_equal(result, domain_2d.ur)

    def test_1d_clamp_below(self, domain_1d):
        x = np.array([-5.0])
        result = domain_1d.project_into_domain(x)
        np.testing.assert_array_almost_equal(result, [0.0])

    def test_1d_clamp_above(self, domain_1d):
        x = np.array([5.0])
        result = domain_1d.project_into_domain(x)
        np.testing.assert_array_almost_equal(result, [1.0])

    def test_projected_point_is_in_domain(self, domain_2d):
        x = np.array([-10.0, 10.0])
        proj = domain_2d.project_into_domain(x)
        assert np.all(proj >= domain_2d.ll)
        assert np.all(proj <= domain_2d.ur)


# ---------------------------------------------------------------------------
# is_in tests
# ---------------------------------------------------------------------------

class TestIsIn:

    def test_interior_point_is_in(self, domain_2d):
        p = np.array([[0.5, 0.5]])
        assert domain_2d.is_in(p)[0] == True

    def test_ll_corner_is_in(self, domain_2d):
        p = domain_2d.ll.reshape(1, -1)
        assert domain_2d.is_in(p)[0] == True

    def test_ur_corner_is_in(self, domain_2d):
        p = domain_2d.ur.reshape(1, -1)
        assert domain_2d.is_in(p)[0] == True

    def test_exterior_point_not_in(self, domain_2d):
        p = np.array([[2.0, 2.0]])
        assert domain_2d.is_in(p)[0] == False

    def test_partially_outside_not_in(self, domain_2d):
        p = np.array([[-0.1, 0.5]])
        assert domain_2d.is_in(p)[0] == False

    def test_batch_mixed_points(self, domain_2d):
        p = np.array([
            [0.5, 0.5],   # inside
            [2.0, 0.5],   # outside x
            [0.5, -0.1],  # outside y
            [0.0, 0.0],   # corner (on boundary)
        ])
        result = domain_2d.is_in(p)
        expected = np.array([True, False, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_all_inside_batch(self, domain_2d):
        p = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
        assert np.all(domain_2d.is_in(p))

    def test_all_outside_batch(self, domain_2d):
        p = np.array([[-1.0, 0.5], [0.5, 2.0], [2.0, 2.0]])
        assert not np.any(domain_2d.is_in(p))


# ---------------------------------------------------------------------------
# __str__ tests
# ---------------------------------------------------------------------------

class TestDomainStr:

    def test_str_contains_dim(self, domain_2d):
        s = str(domain_2d)
        assert 'dim' in s
        assert '2' in s

    def test_str_contains_ll_ur(self, domain_2d):
        s = str(domain_2d)
        assert 'LL' in s or 'll' in s.lower()
        assert 'UR' in s or 'ur' in s.lower()

    def test_str_contains_diameter(self, domain_2d):
        s = str(domain_2d)
        assert 'diameter' in s

    def test_str_returns_string(self, domain_2d):
        assert isinstance(str(domain_2d), str)

    def test_verbose_true_prints_str(self, capsys):
        """verbose=True should print the domain string during __init__."""
        Domain(boundary=[[0.0, 0.0], [1.0, 1.0]], verbose=True)
        captured = capsys.readouterr()
        assert 'dim' in captured.out or 'diameter' in captured.out
