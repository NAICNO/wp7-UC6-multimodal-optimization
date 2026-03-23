"""
test_function.py — Unit tests for mmo.function.Function

Covers:
- Construction and property defaults
- __call__ with 1D/2D x arrays, empty x, dimension mismatch
- n_fev / n_call tracking
- tag / notag mechanism and tagged_xy, tagged_x, tagged_y
- record(False) suppresses tracking
- xy(), x(), y() accessors
- project=True uses domain.project_into_domain
- xy_in_domain filters correctly
- __str__ representation
"""

import numpy as np
import pytest

from mmo.domain import Domain
from mmo.function import Function


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SumSquares:
    """f(x) = sum(x^2)"""
    def evaluate(self, x):
        return float(np.dot(x, x))


class _Identity:
    """f(x) = x[0] (1D)"""
    def evaluate(self, x):
        return float(x[0])


def _make_function(dim=2, project=False):
    domain = Domain(boundary=[[0.0] * dim, [1.0] * dim])
    f_raw = _SumSquares()
    return Function(f=f_raw, domain=domain, project=project), domain


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestFunctionConstruction:

    def test_dim_matches_domain(self):
        f, domain = _make_function(dim=2)
        assert f.dim == 2

    def test_n_fev_starts_at_zero(self):
        f, _ = _make_function()
        assert f.n_fev == 0

    def test_n_call_starts_at_zero(self):
        f, _ = _make_function()
        assert f.n_call == 0

    def test_record_defaults_to_one(self):
        f, _ = _make_function()
        assert f._record == 1

    def test_current_tag_defaults_to_minus_one(self):
        f, _ = _make_function()
        assert f.current_tag == -1

    def test_xyt_starts_empty(self):
        f, _ = _make_function(dim=3)
        assert f._xyt.shape == (0, 5)  # dim(3) + y(1) + tag(1)

    def test_5d_xyt_shape(self):
        f, _ = _make_function(dim=5)
        assert f._xyt.shape[1] == 7  # 5 + 1 + 1


# ---------------------------------------------------------------------------
# __call__ behaviour
# ---------------------------------------------------------------------------

class TestFunctionCall:

    def test_evaluates_2d_single_row(self):
        f, _ = _make_function(dim=2)
        x = np.array([[0.3, 0.4]])
        y = f(x)
        assert y.shape == (1,)
        assert y[0] == pytest.approx(0.3 ** 2 + 0.4 ** 2)

    def test_evaluates_2d_multiple_rows(self):
        f, _ = _make_function(dim=2)
        x = np.array([[0.0, 0.0], [1.0, 0.0]])
        y = f(x)
        assert y.shape == (2,)
        assert y[0] == pytest.approx(0.0)
        assert y[1] == pytest.approx(1.0)

    def test_1d_input_reshaped_correctly(self):
        f, _ = _make_function(dim=2)
        x = np.array([0.5, 0.5])  # shape (2,) not (1,2)
        y = f(x)
        assert y.shape == (1,)
        assert y[0] == pytest.approx(0.5)

    def test_empty_input_returns_empty(self):
        f, _ = _make_function(dim=2)
        x = np.zeros((0, 2))
        y = f(x)
        assert y.shape == (0,)

    def test_wrong_dim_raises(self):
        f, _ = _make_function(dim=2)
        x = np.array([[0.1, 0.2, 0.3]])  # wrong dim
        with pytest.raises(AssertionError):
            f(x)

    def test_n_fev_increments_per_point(self):
        f, _ = _make_function(dim=2)
        f(np.array([[0.1, 0.2], [0.3, 0.4]]))
        assert f.n_fev == 2

    def test_n_call_increments_per_point(self):
        f, _ = _make_function(dim=2)
        f(np.array([[0.1, 0.2], [0.3, 0.4]]))
        assert f.n_call == 2

    def test_empty_call_does_not_increment_n_fev(self):
        f, _ = _make_function(dim=2)
        f(np.zeros((0, 2)))
        assert f.n_fev == 0


# ---------------------------------------------------------------------------
# record(False) suppresses tracking
# ---------------------------------------------------------------------------

class TestRecord:

    def test_record_false_suppresses_n_fev(self):
        f, _ = _make_function(dim=2)
        f.record(False)
        f(np.array([[0.1, 0.2]]))
        assert f.n_fev == 0

    def test_record_false_suppresses_xyt_growth(self):
        f, _ = _make_function(dim=2)
        f.record(False)
        f(np.array([[0.1, 0.2]]))
        assert f._xyt.shape[0] == 0

    def test_record_restored_increments_again(self):
        f, _ = _make_function(dim=2)
        f.record(False)
        f(np.array([[0.1, 0.2]]))
        f.record(True)
        f(np.array([[0.3, 0.4]]))
        assert f.n_fev == 1


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------

class TestTagging:

    def test_tag_increments_n_tag(self):
        f, _ = _make_function(dim=2)
        t = f.tag()
        assert t == 0
        assert f.n_tag == 1

    def test_consecutive_tags_distinct(self):
        f, _ = _make_function(dim=2)
        t0 = f.tag()
        t1 = f.tag()
        assert t0 != t1

    def test_notag_resets_current_tag(self):
        f, _ = _make_function(dim=2)
        f.tag()
        f.notag()
        assert f.current_tag == -1

    def test_tagged_xy_returns_only_tagged_rows(self):
        f, _ = _make_function(dim=2)
        t0 = f.tag()
        f(np.array([[0.1, 0.2]]))
        t1 = f.tag()
        f(np.array([[0.5, 0.6]]))

        xy0 = f.tagged_xy(t0)
        xy1 = f.tagged_xy(t1)
        assert xy0.shape[0] == 1
        assert xy1.shape[0] == 1

    def test_tagged_x_returns_correct_coords(self):
        f, _ = _make_function(dim=2)
        t = f.tag()
        x_in = np.array([[0.3, 0.7]])
        f(x_in)
        tx = f.tagged_x(t)
        np.testing.assert_array_almost_equal(tx, x_in)

    def test_tagged_y_returns_correct_values(self):
        f, _ = _make_function(dim=2)
        t = f.tag()
        x_in = np.array([[0.0, 0.0]])
        f(x_in)
        ty = f.tagged_y(t)
        assert ty[0] == pytest.approx(0.0)

    def test_unknown_tag_returns_empty(self):
        f, _ = _make_function(dim=2)
        f(np.array([[0.1, 0.2]]))
        xy = f.tagged_xy(999)
        assert xy.shape[0] == 0


# ---------------------------------------------------------------------------
# xy, x, y accessors
# ---------------------------------------------------------------------------

class TestAccessors:

    def test_xy_shape_grows_with_evaluations(self):
        f, _ = _make_function(dim=2)
        f(np.array([[0.1, 0.2], [0.3, 0.4]]))
        xy = f.xy()
        assert xy.shape == (2, 3)  # 2 rows, dim+1 cols

    def test_x_returns_input_coordinates(self):
        f, _ = _make_function(dim=2)
        x_in = np.array([[0.3, 0.7]])
        f(x_in)
        x_out = f.x()
        np.testing.assert_array_almost_equal(x_out, x_in)

    def test_y_returns_function_values(self):
        f, _ = _make_function(dim=2)
        x_in = np.array([[0.0, 0.0]])
        f(x_in)
        y = f.y()
        assert y[0] == pytest.approx(0.0)

    def test_xy_in_domain_filters_outside_points(self):
        f, domain = _make_function(dim=2)
        f(np.array([[0.5, 0.5]]))   # inside
        f(np.array([[2.0, 2.0]]))   # outside (but function evaluates anyway)
        xy = f.xy_in_domain(domain=domain)
        assert xy.shape[0] == 1
        assert xy[0, 0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# project=True
# ---------------------------------------------------------------------------

class TestProjectMode:

    def test_out_of_domain_projected_before_eval(self):
        domain = Domain(boundary=[[0.0, 0.0], [1.0, 1.0]])
        f_raw = _SumSquares()
        f = Function(f=f_raw, domain=domain, project=True)
        # Point outside domain: [-1, -1] -> projected to [0, 0] -> f=0
        x = np.array([[-1.0, -1.0]])
        y = f(x)
        assert y[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------

class TestFunctionStr:

    def test_str_contains_dim(self):
        f, _ = _make_function(dim=2)
        s = str(f)
        assert '2' in s

    def test_str_contains_fev(self):
        f, _ = _make_function(dim=2)
        s = str(f)
        assert 'fev' in s or 'fev' in s.lower()

    def test_str_returns_string_type(self):
        f, _ = _make_function(dim=2)
        assert isinstance(str(f), str)
