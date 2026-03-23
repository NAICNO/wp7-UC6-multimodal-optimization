"""
test_ga_seed_utils.py — Unit tests for the seven pure geometry functions in mmo.ga_seed

Functions under test:
  volume_of_orthotope(p1, p2)
  volume_of_orthotope_axis_1(p1, p2)
  is_lower(p1, p2)
  is_lower_axis_1(p1, p2)
  in_domain(domain, p)
  p_in_domain_distance_to_boundary(domain, p)
  distance_to_domain(domain, p)
"""

import numpy as np
import pytest

from mmo.ga_seed import (
    volume_of_orthotope,
    volume_of_orthotope_axis_1,
    is_lower,
    is_lower_axis_1,
    in_domain,
    p_in_domain_distance_to_boundary,
    distance_to_domain,
)
from mmo.domain import Domain


# ---------------------------------------------------------------------------
# volume_of_orthotope
# ---------------------------------------------------------------------------

class TestVolumeOfOrthotope:

    def test_unit_square_volume(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 1.0])
        assert volume_of_orthotope(p1, p2) == pytest.approx(1.0)

    def test_rectangle_volume(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([2.0, 3.0])
        assert volume_of_orthotope(p1, p2) == pytest.approx(6.0)

    def test_volume_is_positive_regardless_of_order(self):
        p1 = np.array([1.0, 1.0])
        p2 = np.array([0.0, 0.0])
        assert volume_of_orthotope(p1, p2) == pytest.approx(1.0)

    def test_1d_volume_is_length(self):
        p1 = np.array([2.0])
        p2 = np.array([5.0])
        assert volume_of_orthotope(p1, p2) == pytest.approx(3.0)

    def test_3d_volume(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([2.0, 3.0, 4.0])
        assert volume_of_orthotope(p1, p2) == pytest.approx(24.0)

    def test_zero_length_edge_gives_zero_volume(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        assert volume_of_orthotope(p1, p2) == pytest.approx(0.0)

    def test_identical_points_give_zero(self):
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([1.0, 2.0, 3.0])
        assert volume_of_orthotope(p1, p2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# volume_of_orthotope_axis_1
# ---------------------------------------------------------------------------

class TestVolumeOfOrthotopAxis1:

    def test_single_row(self):
        p1 = np.array([[0.0, 0.0]])
        p2 = np.array([[2.0, 3.0]])
        result = volume_of_orthotope_axis_1(p1, p2)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(6.0)

    def test_multiple_rows_unit_squares(self):
        p1 = np.zeros((3, 2))
        p2 = np.ones((3, 2))
        result = volume_of_orthotope_axis_1(p1, p2)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0])

    def test_different_volumes_per_row(self):
        p1 = np.zeros((2, 2))
        p2 = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = volume_of_orthotope_axis_1(p1, p2)
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(12.0)

    def test_result_non_negative(self):
        p1 = np.random.rand(5, 3)
        p2 = np.random.rand(5, 3)
        result = volume_of_orthotope_axis_1(p1, p2)
        assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# is_lower
# ---------------------------------------------------------------------------

class TestIsLower:

    def test_strictly_lower_returns_true(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 1.0])
        assert is_lower(p1, p2) == True

    def test_equal_returns_false(self):
        p1 = np.array([1.0, 1.0])
        p2 = np.array([1.0, 1.0])
        assert is_lower(p1, p2) == False

    def test_one_component_equal_returns_false(self):
        p1 = np.array([0.0, 1.0])
        p2 = np.array([1.0, 1.0])
        assert is_lower(p1, p2) == False

    def test_greater_returns_false(self):
        p1 = np.array([2.0, 2.0])
        p2 = np.array([1.0, 1.0])
        assert is_lower(p1, p2) == False

    def test_1d_lower(self):
        assert is_lower(np.array([0.0]), np.array([1.0])) == True

    def test_1d_not_lower(self):
        assert is_lower(np.array([1.0]), np.array([1.0])) == False


# ---------------------------------------------------------------------------
# is_lower_axis_1
# ---------------------------------------------------------------------------

class TestIsLowerAxis1:

    def test_all_rows_lower_returns_all_true(self):
        p1 = np.zeros((3, 2))
        p2 = np.ones((3, 2))
        result = is_lower_axis_1(p1, p2)
        assert np.all(result)

    def test_no_rows_lower_returns_all_false(self):
        p1 = np.ones((3, 2))
        p2 = np.zeros((3, 2))
        result = is_lower_axis_1(p1, p2)
        assert not np.any(result)

    def test_mixed_rows(self):
        p1 = np.array([[0.0, 0.0], [1.0, 1.0]])
        p2 = np.array([[1.0, 1.0], [0.0, 0.0]])
        result = is_lower_axis_1(p1, p2)
        assert result[0] == True
        assert result[1] == False

    def test_equal_values_not_lower(self):
        p1 = np.array([[1.0, 1.0]])
        p2 = np.array([[1.0, 2.0]])
        result = is_lower_axis_1(p1, p2)
        assert result[0] == False


# ---------------------------------------------------------------------------
# in_domain
# ---------------------------------------------------------------------------

class TestInDomain:

    def test_interior_point_is_in_domain(self, domain_2d):
        p = np.array([0.5, 0.5])
        assert in_domain(domain=domain_2d, p=p) == True

    def test_point_at_ll_is_not_strictly_in(self, domain_2d):
        # in_domain uses strict inequality (is_lower checks p1 < p2)
        p = domain_2d.ll.copy()
        result = in_domain(domain=domain_2d, p=p)
        assert result == False

    def test_point_outside_x_is_not_in_domain(self, domain_2d):
        p = np.array([1.5, 0.5])
        assert in_domain(domain=domain_2d, p=p) == False

    def test_point_outside_y_is_not_in_domain(self, domain_2d):
        p = np.array([0.5, 1.5])
        assert in_domain(domain=domain_2d, p=p) == False

    def test_deep_interior_is_in(self, domain_2d):
        p = np.array([0.3, 0.7])
        assert in_domain(domain=domain_2d, p=p) == True


# ---------------------------------------------------------------------------
# p_in_domain_distance_to_boundary
# ---------------------------------------------------------------------------

class TestPInDomainDistanceToBoundary:

    def test_center_of_unit_square(self, domain_2d):
        # Center [0.5, 0.5]: distance to each wall is 0.5, min is 0.5
        p = np.array([[0.5, 0.5]])
        result = p_in_domain_distance_to_boundary(domain=domain_2d, p=p)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(0.5)

    def test_asymmetric_interior_point(self):
        domain = Domain(boundary=[[0.0, 0.0], [4.0, 4.0]])
        p = np.array([[1.0, 2.0]])
        result = p_in_domain_distance_to_boundary(domain=domain, p=p)
        # distances to walls: left=1, right=3, bottom=2, top=2 -> min of mins
        # min per axis: x->min(1,3)=1, y->min(2,2)=2 -> overall min=1
        assert result[0] == pytest.approx(1.0)

    def test_multiple_interior_points(self, domain_2d):
        p = np.array([[0.1, 0.5], [0.5, 0.5]])
        result = p_in_domain_distance_to_boundary(domain=domain_2d, p=p)
        assert result.shape == (2,)
        assert result[0] == pytest.approx(0.1)
        assert result[1] == pytest.approx(0.5)

    def test_point_outside_domain_raises(self, domain_2d):
        p = np.array([[1.5, 0.5]])
        with pytest.raises((AssertionError, Exception)):
            p_in_domain_distance_to_boundary(domain=domain_2d, p=p)

    def test_distance_is_non_negative(self, domain_2d):
        p = np.array([[0.3, 0.7]])
        result = p_in_domain_distance_to_boundary(domain=domain_2d, p=p)
        assert result[0] >= 0.0


# ---------------------------------------------------------------------------
# distance_to_domain
# ---------------------------------------------------------------------------

class TestDistanceToDomain:

    def test_interior_point_distance_zero(self, domain_2d):
        p = np.array([0.5, 0.5])
        d = distance_to_domain(domain=domain_2d, p=p)
        assert d == pytest.approx(0.0)

    def test_point_on_boundary_distance_zero(self, domain_2d):
        p = np.array([0.0, 0.5])
        d = distance_to_domain(domain=domain_2d, p=p)
        assert d == pytest.approx(0.0)

    def test_point_outside_x_axis(self, domain_2d):
        p = np.array([2.0, 0.5])
        d = distance_to_domain(domain=domain_2d, p=p)
        assert d == pytest.approx(1.0)

    def test_point_outside_y_axis(self, domain_2d):
        p = np.array([0.5, -1.0])
        d = distance_to_domain(domain=domain_2d, p=p)
        assert d == pytest.approx(1.0)

    def test_point_far_outside(self, domain_2d):
        p = np.array([-5.0, 0.5])
        d = distance_to_domain(domain=domain_2d, p=p)
        assert d == pytest.approx(5.0)

    def test_distance_is_non_negative_for_interior(self, domain_2d):
        p = np.array([0.3, 0.8])
        d = distance_to_domain(domain=domain_2d, p=p)
        assert d >= 0.0
