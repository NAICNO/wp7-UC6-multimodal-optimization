"""
test_gatools.py — Unit tests for mmo.gatools

Covers:
- remove_identical: empty list, no duplicates, all duplicates, preserves order
- dublicates: empty list, no duplicates, all duplicates, multiple duplicates
- assert_distinct: passes on distinct, raises on duplicate
- clearing_1d_2f: DEAP-style individuals with 2-objective fitness
"""

import pytest

from mmo.gatools import remove_identical, dublicates, assert_distinct, clearing_1d_2f


# ---------------------------------------------------------------------------
# Helper to create fake "individuals" matching the interface:
# each individual is a sequence where ind[0] is the key.
# ---------------------------------------------------------------------------

def _ind(key, val=None):
    """Create a fake individual as a list where ind[0] = key."""
    return [key, val]


# ---------------------------------------------------------------------------
# remove_identical
# ---------------------------------------------------------------------------

class TestRemoveIdentical:

    def test_empty_list_returns_empty(self):
        result = remove_identical([])
        assert result == []

    def test_single_individual_returned_as_is(self):
        ind = _ind('a', 1)
        result = remove_identical([ind])
        assert len(result) == 1
        assert result[0][0] == 'a'

    def test_all_distinct_all_kept(self):
        inds = [_ind('a'), _ind('b'), _ind('c')]
        result = remove_identical(inds)
        assert len(result) == 3

    def test_exact_duplicate_removed(self):
        inds = [_ind('a', 1), _ind('a', 2)]
        result = remove_identical(inds)
        assert len(result) == 1
        assert result[0][0] == 'a'

    def test_first_occurrence_kept(self):
        """When duplicates exist, the first occurrence must be preserved."""
        inds = [_ind('x', 'first'), _ind('x', 'second')]
        result = remove_identical(inds)
        assert result[0][1] == 'first'

    def test_multiple_duplicates_of_same_key(self):
        inds = [_ind('z', k) for k in range(5)]
        result = remove_identical(inds)
        assert len(result) == 1
        assert result[0][1] == 0  # first value preserved

    def test_mixed_duplicates_and_uniques(self):
        inds = [_ind('a'), _ind('b'), _ind('a'), _ind('c'), _ind('b')]
        result = remove_identical(inds)
        keys = [r[0] for r in result]
        assert set(keys) == {'a', 'b', 'c'}
        assert len(result) == 3

    def test_order_preserved_for_unique_keys(self):
        inds = [_ind('c'), _ind('a'), _ind('b')]
        result = remove_identical(inds)
        assert [r[0] for r in result] == ['c', 'a', 'b']

    def test_integer_keys_work(self):
        inds = [_ind(1), _ind(2), _ind(1), _ind(3)]
        result = remove_identical(inds)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# dublicates
# ---------------------------------------------------------------------------

class TestDublicates:

    def test_empty_list_returns_empty(self):
        assert dublicates([]) == []

    def test_no_duplicates_returns_empty(self):
        inds = [_ind('a'), _ind('b'), _ind('c')]
        assert dublicates(inds) == []

    def test_single_pair_returns_one_duplicate(self):
        inds = [_ind('a', 'first'), _ind('a', 'second')]
        result = dublicates(inds)
        assert len(result) == 1
        assert result[0][1] == 'second'

    def test_three_copies_returns_two_duplicates(self):
        inds = [_ind('a', 1), _ind('a', 2), _ind('a', 3)]
        result = dublicates(inds)
        assert len(result) == 2

    def test_mixed_returns_only_duplicates(self):
        inds = [_ind('a'), _ind('b'), _ind('a'), _ind('c'), _ind('b')]
        result = dublicates(inds)
        assert len(result) == 2
        dup_keys = [r[0] for r in result]
        assert 'a' in dup_keys
        assert 'b' in dup_keys

    def test_all_same_key_all_but_first_are_dups(self):
        inds = [_ind('x', k) for k in range(4)]
        result = dublicates(inds)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# assert_distinct
# ---------------------------------------------------------------------------

class TestAssertDistinct:

    def test_empty_list_passes(self):
        assert_distinct([])  # should not raise

    def test_all_distinct_passes(self):
        inds = [_ind('a'), _ind('b'), _ind('c')]
        assert_distinct(inds)  # should not raise

    def test_duplicate_raises_assertion(self):
        inds = [_ind('a'), _ind('b'), _ind('a')]
        with pytest.raises(AssertionError):
            assert_distinct(inds)

    def test_single_individual_passes(self):
        assert_distinct([_ind('only')])

    def test_all_same_key_raises(self):
        inds = [_ind('dup')] * 3
        with pytest.raises(AssertionError):
            assert_distinct(inds)


# ---------------------------------------------------------------------------
# clearing_1d_2f
# ---------------------------------------------------------------------------

class _MockFitness:
    """Minimal DEAP-like fitness container with a .values tuple."""
    def __init__(self, f0, f1):
        self.values = (f0, f1)


class _MockIndividual:
    """Minimal DEAP-like individual supporting ind[0] and .fitness.values."""
    def __init__(self, key, f0, f1):
        self._key = key
        self.fitness = _MockFitness(f0, f1)

    def __getitem__(self, idx):
        return self._key if idx == 0 else None


class TestClearingOneD2F:

    def test_negative_delta0_raises(self):
        with pytest.raises(AssertionError):
            clearing_1d_2f([], delta_0=-1, delta_1=0)

    def test_negative_delta1_raises(self):
        with pytest.raises(AssertionError):
            clearing_1d_2f([], delta_0=0, delta_1=-1)

    def test_both_zero_raises(self):
        with pytest.raises(AssertionError):
            clearing_1d_2f([], delta_0=0, delta_1=0)

    def test_distinct_individuals_unchanged(self):
        """Two individuals far apart should keep their original fitness values."""
        inds = [
            _MockIndividual('a', 0.0, 0.0),
            _MockIndividual('b', 10.0, 10.0),
        ]
        clearing_1d_2f(inds, delta_0=1.0, delta_1=1.0)
        values = [i.fitness.values for i in inds]
        # Both should have valid fitness (not 2*max boosted)
        assert any(v[0] < 20 for v in values)

    def test_close_individuals_get_cleared(self):
        """Two individuals close together: the second should get boosted fitness."""
        inds = [
            _MockIndividual('a', 1.0, 1.0),
            _MockIndividual('b', 1.01, 1.01),  # very close to 'a'
        ]
        clearing_1d_2f(inds, delta_0=0.1, delta_1=0.1)
        sorted_inds = sorted(inds, key=lambda i: i.fitness.values[0])
        # The second one (when sorted by f0) should be penalised
        assert sorted_inds[1].fitness.values[0] >= 2 * 1.01
