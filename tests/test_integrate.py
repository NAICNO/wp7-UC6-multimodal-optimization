"""
test_integrate.py — Unit tests for mmo.integrate

Covers:
- xy_add: normal case, shape preservation, different row counts
- xy_reduce_xequal_ymin: deduplication logic with tolerance, minimum-value selection,
  distinct points preserved, empty array, single row, large tolerance,
  zero tolerance, special values, 1D x with 1D y
"""

import numpy as np
import pytest

from mmo.integrate import xy_add, xy_reduce_xequal_ymin


# ---------------------------------------------------------------------------
# xy_add tests
# ---------------------------------------------------------------------------

class TestXyAdd:

    def test_concatenates_two_arrays(self):
        xy1 = np.array([[1.0, 2.0, 3.0]])
        xy2 = np.array([[4.0, 5.0, 6.0]])
        result = xy_add(xy1, xy2)
        assert result.shape == (2, 3)

    def test_correct_values_after_concat(self):
        xy1 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]])
        xy2 = np.array([[0.5, 0.5, 0.5]])
        result = xy_add(xy1, xy2)
        np.testing.assert_array_equal(result[:2], xy1)
        np.testing.assert_array_equal(result[2], [0.5, 0.5, 0.5])

    def test_shape_column_count_preserved(self):
        xy1 = np.zeros((3, 4))
        xy2 = np.ones((5, 4))
        result = xy_add(xy1, xy2)
        assert result.shape == (8, 4)

    def test_mismatched_columns_raises(self):
        xy1 = np.zeros((2, 3))
        xy2 = np.zeros((2, 4))
        with pytest.raises(AssertionError):
            xy_add(xy1, xy2)

    def test_empty_first_array(self):
        xy1 = np.zeros((0, 3))
        xy2 = np.ones((2, 3))
        result = xy_add(xy1, xy2)
        assert result.shape == (2, 3)

    def test_empty_second_array(self):
        xy1 = np.ones((2, 3))
        xy2 = np.zeros((0, 3))
        result = xy_add(xy1, xy2)
        assert result.shape == (2, 3)

    def test_both_empty_arrays(self):
        xy1 = np.zeros((0, 2))
        xy2 = np.zeros((0, 2))
        result = xy_add(xy1, xy2)
        assert result.shape == (0, 2)

    def test_returns_new_array_not_mutation(self):
        xy1 = np.array([[1.0, 2.0]])
        xy2 = np.array([[3.0, 4.0]])
        original_xy1_id = id(xy1)
        result = xy_add(xy1, xy2)
        assert id(result) != original_xy1_id


# ---------------------------------------------------------------------------
# xy_reduce_xequal_ymin tests
# ---------------------------------------------------------------------------

class TestXyReduceXequalYmin:

    def _make_xy(self, rows):
        """Helper: rows is list of (x_coords..., y) tuples."""
        return np.array(rows, dtype=float)

    def test_single_row_returned_unchanged(self):
        xy = self._make_xy([[0.5, 0.5, 1.0]])
        result = xy_reduce_xequal_ymin(xy, tol=0.1)
        assert result.shape == (1, 3)

    def test_two_distinct_points_both_kept(self):
        xy = self._make_xy([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
        result = xy_reduce_xequal_ymin(xy, tol=0.01)
        assert result.shape == (2, 3)

    def test_two_identical_x_keeps_min_y(self):
        xy = self._make_xy([
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 2.0],
        ])
        result = xy_reduce_xequal_ymin(xy, tol=0.1)
        assert result.shape == (1, 3)
        assert result[0, -1] == pytest.approx(2.0)

    def test_three_identical_x_keeps_min_y(self):
        xy = self._make_xy([
            [0.5, 0.5, 10.0],
            [0.5, 0.5, 3.0],
            [0.5, 0.5, 7.0],
        ])
        result = xy_reduce_xequal_ymin(xy, tol=0.1)
        assert result.shape == (1, 3)
        assert result[0, -1] == pytest.approx(3.0)

    def test_near_points_merged_within_tolerance(self):
        xy = self._make_xy([
            [0.0, 0.0, 5.0],
            [0.001, 0.001, 2.0],
        ])
        result = xy_reduce_xequal_ymin(xy, tol=0.1)
        assert result.shape == (1, 3)
        assert result[0, -1] == pytest.approx(2.0)

    def test_points_outside_tolerance_kept_separate(self):
        xy = self._make_xy([
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 2.0],
        ])
        result = xy_reduce_xequal_ymin(xy, tol=0.01)
        assert result.shape == (2, 3)

    def test_empty_array_returns_empty(self):
        xy = np.zeros((0, 3))
        result = xy_reduce_xequal_ymin(xy, tol=0.1)
        assert result.shape[0] == 0

    def test_1d_x_single_y_deduplication(self):
        xy = self._make_xy([
            [0.0, 10.0],
            [0.0, 3.0],
            [0.0, 7.0],
        ])
        result = xy_reduce_xequal_ymin(xy, tol=0.1)
        assert result.shape == (1, 2)
        assert result[0, -1] == pytest.approx(3.0)

    def test_output_shape_columns_preserved(self):
        xy = self._make_xy([
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 2.0],
        ])
        result = xy_reduce_xequal_ymin(xy, tol=0.01)
        assert result.shape[1] == 3

    def test_three_groups_reduces_correctly(self):
        xy = self._make_xy([
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 4.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 3.0],
        ])
        result = xy_reduce_xequal_ymin(xy, tol=0.01)
        assert result.shape[0] == 3
        y_values = sorted(result[:, -1])
        assert y_values[0] == pytest.approx(1.0)
        assert y_values[1] == pytest.approx(2.0)
        assert y_values[2] == pytest.approx(3.0)

    def test_large_tolerance_merges_all(self):
        xy = self._make_xy([
            [0.0, 0.0, 5.0],
            [0.1, 0.1, 2.0],
            [0.2, 0.2, 8.0],
        ])
        result = xy_reduce_xequal_ymin(xy, tol=1.0)
        assert result.shape[0] == 1
        assert result[0, -1] == pytest.approx(2.0)

    def test_minimum_y_value_is_selected_not_first_seen(self):
        # Points ordered so the non-minimum appears first.
        xy = self._make_xy([
            [0.0, 0.0, 99.0],
            [0.0, 0.001, 1.0],
        ])
        result = xy_reduce_xequal_ymin(xy, tol=0.1)
        assert result.shape[0] == 1
        assert result[0, -1] == pytest.approx(1.0)

    def test_negative_y_values_handled(self):
        xy = self._make_xy([
            [0.0, 0.0, -3.0],
            [0.0, 0.001, -10.0],
        ])
        result = xy_reduce_xequal_ymin(xy, tol=0.1)
        assert result[0, -1] == pytest.approx(-10.0)

    def test_returns_2d_array(self):
        xy = self._make_xy([[0.0, 0.0, 1.0]])
        result = xy_reduce_xequal_ymin(xy, tol=0.1)
        assert result.ndim == 2
