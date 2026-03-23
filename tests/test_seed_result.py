"""
test_seed_result.py — Unit tests for mmo.seed_result

Covers:
- SeedResult construction: seeds=None, seeds with spread, corr_matrices
- n_seeds property
- __eq__ comparison
- info_prepend / info_append
- __str__
- SeedsExclude filtering
"""

import numpy as np
import pytest

from mmo.seed_result import SeedResult, SeedsExclude


# ---------------------------------------------------------------------------
# SeedResult construction
# ---------------------------------------------------------------------------

class TestSeedResultConstruction:

    def test_empty_seed_result_n_seeds_zero(self):
        sr = SeedResult()
        assert sr.n_seeds == 0

    def test_seeds_none_n_seeds_zero(self):
        sr = SeedResult(seeds=None)
        assert sr.n_seeds == 0

    def test_n_seeds_matches_array_length(self):
        seeds = np.array([[0.1, 0.2], [0.3, 0.4]])
        sr = SeedResult(seeds=seeds)
        assert sr.n_seeds == 2

    def test_spread_stored(self):
        seeds = np.array([[0.1, 0.2]])
        spread = np.array([0.5])
        sr = SeedResult(seeds=seeds, spread=spread)
        np.testing.assert_array_equal(sr.spread, spread)

    def test_spread_shape_mismatch_raises(self):
        seeds = np.array([[0.1, 0.2], [0.3, 0.4]])
        spread = np.array([0.5])  # only 1 element for 2 seeds
        with pytest.raises(AssertionError):
            SeedResult(seeds=seeds, spread=spread)

    def test_corr_matrices_stored(self):
        seeds = np.array([[0.1, 0.2]])
        spread = np.array([0.5])
        corr = np.eye(2).reshape(1, 2, 2)
        sr = SeedResult(seeds=seeds, spread=spread, corr_matrices=corr)
        assert sr.corr_matrices.shape == (1, 2, 2)

    def test_corr_matrices_shape_mismatch_raises(self):
        seeds = np.array([[0.1, 0.2], [0.3, 0.4]])
        spread = np.array([0.5, 0.6])
        corr = np.eye(2).reshape(1, 2, 2)  # only 1 for 2 seeds
        with pytest.raises(AssertionError):
            SeedResult(seeds=seeds, spread=spread, corr_matrices=corr)

    def test_info_defaults_to_empty_string(self):
        sr = SeedResult()
        assert sr.info == ''


# ---------------------------------------------------------------------------
# __eq__
# ---------------------------------------------------------------------------

class TestSeedResultEq:

    def test_eq_matches_n_seeds(self):
        seeds = np.array([[0.1, 0.2], [0.3, 0.4]])
        sr = SeedResult(seeds=seeds)
        assert sr == 2

    def test_eq_wrong_count_false(self):
        seeds = np.array([[0.1, 0.2]])
        sr = SeedResult(seeds=seeds)
        assert not (sr == 5)

    def test_empty_eq_zero(self):
        sr = SeedResult()
        assert sr == 0


# ---------------------------------------------------------------------------
# info_prepend / info_append
# ---------------------------------------------------------------------------

class TestInfoMethods:

    def test_info_prepend(self):
        sr = SeedResult(info='world')
        sr.info_prepend('hello ')
        assert sr.info == 'hello world'

    def test_info_append(self):
        sr = SeedResult(info='hello')
        sr.info_append(' world')
        assert sr.info == 'hello world'

    def test_info_prepend_then_append(self):
        sr = SeedResult(info='middle')
        sr.info_prepend('start ')
        sr.info_append(' end')
        assert sr.info == 'start middle end'


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------

class TestSeedResultStr:

    def test_str_contains_n_seeds(self):
        seeds = np.array([[0.1, 0.2]])
        sr = SeedResult(seeds=seeds)
        assert 'n_seeds' in str(sr) or '1' in str(sr)

    def test_str_is_string(self):
        sr = SeedResult()
        assert isinstance(str(sr), str)

    def test_str_with_corr_matrices_shows_eigenvalues(self):
        """__str__ with corr_matrices exercises the eigenvalue display branch."""
        seeds = np.array([[0.1, 0.2], [0.3, 0.4]])
        spread = np.array([0.1, 0.2])
        corr = np.array([np.eye(2), 4.0 * np.eye(2)])
        sr = SeedResult(seeds=seeds, spread=spread, corr_matrices=corr)
        s = str(sr)
        assert 'Correlation' in s or 'matrices' in s.lower()
        assert isinstance(s, str)


# ---------------------------------------------------------------------------
# SeedsExclude
# ---------------------------------------------------------------------------

class TestSeedsExclude:

    def _make_sr(self, n=3, dim=2):
        seeds = np.random.rand(n, dim)
        spread = np.random.rand(n)
        return SeedResult(seeds=seeds, spread=spread)

    def test_exclude_none_keeps_all(self):
        sr = self._make_sr(n=3)
        idx = np.array([False, False, False])
        result = SeedsExclude(sr, idx)
        assert result.n_seeds == 3

    def test_exclude_all_returns_empty(self):
        sr = self._make_sr(n=3)
        idx = np.array([True, True, True])
        result = SeedsExclude(sr, idx)
        assert result.n_seeds == 0

    def test_exclude_first_keeps_rest(self):
        seeds = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        spread = np.array([0.1, 0.2, 0.3])
        sr = SeedResult(seeds=seeds, spread=spread)
        idx = np.array([True, False, False])
        result = SeedsExclude(sr, idx)
        assert result.n_seeds == 2
        np.testing.assert_array_equal(result.seeds[0], [0.3, 0.4])

    def test_exclude_wrong_size_idx_raises(self):
        sr = self._make_sr(n=3)
        idx = np.array([True, False])  # length mismatch
        with pytest.raises(AssertionError):
            SeedsExclude(sr, idx)

    def test_exclude_with_corr_matrices(self):
        seeds = np.array([[0.1, 0.2], [0.3, 0.4]])
        spread = np.array([0.1, 0.2])
        corr = np.array([np.eye(2), 2.0 * np.eye(2)])
        sr = SeedResult(seeds=seeds, spread=spread, corr_matrices=corr)
        idx = np.array([True, False])
        result = SeedsExclude(sr, idx)
        assert result.n_seeds == 1
        np.testing.assert_array_almost_equal(result.corr_matrices[0], 2.0 * np.eye(2))

    def test_exclude_with_none_spread(self):
        seeds = np.array([[0.1, 0.2], [0.3, 0.4]])
        sr = SeedResult(seeds=seeds, spread=None)
        idx = np.array([False, True])
        result = SeedsExclude(sr, idx)
        assert result.n_seeds == 1
        assert result.spread is None
