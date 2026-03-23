"""
test_config.py — Unit tests for mmo.config.Config

Covers:
- Valid construction for dim=1, 2, 5
- dof_quad_polynomial formula
- n_pop, n_gen, seed_n_nb1 for standard profile
- Other standard defaults (keep_old_pop, deterministic, quadratic_fit, cma_popsize, nb)
- Invalid dim raises AssertionError
- Invalid profile raises AssertionError
- __str__ output
"""

import pytest

from mmo.config import Config


# ---------------------------------------------------------------------------
# Valid construction
# ---------------------------------------------------------------------------

class TestConfigConstruction:

    def test_dim_stored_correctly(self):
        c = Config(dim=2)
        assert c.dim == 2

    def test_profile_stored_correctly(self):
        c = Config(dim=2, profile='standard')
        assert c.profile == 'standard'

    def test_dim_1_construction(self):
        c = Config(dim=1)
        assert c.dim == 1

    def test_dim_5_construction(self):
        c = Config(dim=5)
        assert c.dim == 5

    def test_dim_10_construction(self):
        c = Config(dim=10)
        assert c.dim == 10


# ---------------------------------------------------------------------------
# dof_quad_polynomial formula: 1 + (3*dim + dim^2) // 2
# ---------------------------------------------------------------------------

class TestDofQuadPolynomial:

    def test_dof_dim1(self):
        c = Config(dim=1)
        expected = 1 + (3 * 1 + 1 * 1) // 2
        assert c.dof_quad_polynomial == expected  # 1 + 2 = 3

    def test_dof_dim2(self):
        c = Config(dim=2)
        expected = 1 + (3 * 2 + 2 * 2) // 2
        assert c.dof_quad_polynomial == expected  # 1 + 5 = 6

    def test_dof_dim3(self):
        c = Config(dim=3)
        expected = 1 + (3 * 3 + 3 * 3) // 2
        assert c.dof_quad_polynomial == expected  # 1 + 9 = 10

    def test_dof_dim5(self):
        c = Config(dim=5)
        expected = 1 + (3 * 5 + 5 * 5) // 2
        assert c.dof_quad_polynomial == expected  # 1 + 20 = 21


# ---------------------------------------------------------------------------
# Standard profile parameters
# ---------------------------------------------------------------------------

class TestStandardProfileParams:

    def test_n_pop_is_ten_times_dof(self):
        c = Config(dim=2)
        assert c.n_pop == 10 * c.dof_quad_polynomial

    def test_n_gen_default(self):
        c = Config(dim=2)
        assert c.n_gen == 20

    def test_seed_n_nb1_is_dof_minus_one(self):
        c = Config(dim=2)
        assert c.seed_n_nb1 == c.dof_quad_polynomial - 1

    def test_keep_old_pop_is_true(self):
        c = Config(dim=2)
        assert c.keep_old_pop == True

    def test_deterministic_is_one(self):
        c = Config(dim=2)
        assert c.deterministic == pytest.approx(1.0)

    def test_quadratic_fit_is_false(self):
        c = Config(dim=2)
        assert c.quadratic_fit == False

    def test_cma_popsize_is_none(self):
        c = Config(dim=2)
        assert c.cma_popsize is None

    def test_nb_is_none(self):
        c = Config(dim=2)
        assert c.nb is None


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

class TestConfigInvalid:

    def test_dim_zero_raises(self):
        with pytest.raises(AssertionError):
            Config(dim=0)

    def test_dim_negative_raises(self):
        with pytest.raises(AssertionError):
            Config(dim=-1)

    def test_unknown_profile_raises(self):
        with pytest.raises(AssertionError):
            Config(dim=2, profile='unknown')

    def test_none_profile_raises(self):
        with pytest.raises(AssertionError):
            Config(dim=2, profile=None)


# ---------------------------------------------------------------------------
# Branch coverage for __str__ conditional paths
# ---------------------------------------------------------------------------

class TestConfigStrBranches:

    def test_str_with_verbose_true_does_not_raise(self, capsys):
        """verbose=True prints to stdout but should not raise."""
        c = Config(dim=2, verbose=True)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_str_cma_popsize_branch_when_set(self):
        """When cma_popsize is not None, __str__ uses the value branch."""
        c = Config(dim=2)
        # Manually set cma_popsize to exercise the non-None branch in __str__
        c.cma_popsize = 10
        s = str(c)
        assert '10' in s

    def test_str_nb_not_none_branch(self):
        """When nb is not None (and quadratic_fit=False), __str__ shows nb.shape[0]."""
        import numpy as np
        c = Config(dim=2)
        c.nb = np.ones((5, 2))
        s = str(c)
        assert '5' in s


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------

class TestConfigStr:

    def test_str_contains_profile(self):
        c = Config(dim=2)
        assert 'standard' in str(c)

    def test_str_contains_dim(self):
        c = Config(dim=2)
        assert '2' in str(c)

    def test_str_contains_n_pop(self):
        c = Config(dim=2)
        assert str(c.n_pop) in str(c)

    def test_str_is_string(self):
        c = Config(dim=2)
        assert isinstance(str(c), str)
