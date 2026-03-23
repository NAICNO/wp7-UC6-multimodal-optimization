"""
test_imports.py — Verify every public MMO module is importable without error.

This is the first gate: if any import fails the whole suite is blocked.
Tests run in isolation — no shared state across modules.
"""

import pytest


class TestCoreModuleImports:
    """All primary MMO modules must be importable."""

    def test_import_domain(self):
        from mmo.domain import Domain
        assert Domain is not None

    def test_import_integrate(self):
        from mmo.integrate import xy_add, xy_reduce_xequal_ymin
        assert callable(xy_add)
        assert callable(xy_reduce_xequal_ymin)

    def test_import_ga_seed_functions(self):
        from mmo.ga_seed import (
            volume_of_orthotope,
            volume_of_orthotope_axis_1,
            is_lower,
            is_lower_axis_1,
            in_domain,
            p_in_domain_distance_to_boundary,
            distance_to_domain,
        )
        for fn in (
            volume_of_orthotope, volume_of_orthotope_axis_1,
            is_lower, is_lower_axis_1, in_domain,
            p_in_domain_distance_to_boundary, distance_to_domain,
        ):
            assert callable(fn)

    def test_import_q_pol(self):
        from mmo.q_pol import FitQuadraticPolynomial, has_minimum
        assert FitQuadraticPolynomial is not None
        assert callable(has_minimum)

    def test_import_gatools(self):
        from mmo.gatools import remove_identical, dublicates, assert_distinct
        assert callable(remove_identical)
        assert callable(dublicates)
        assert callable(assert_distinct)

    def test_import_function(self):
        from mmo.function import Function
        assert Function is not None

    def test_import_solutions(self):
        from mmo.solutions import Solutions
        assert Solutions is not None

    def test_import_seed_result(self):
        from mmo.seed_result import SeedResult, SeedsExclude
        assert SeedResult is not None
        assert callable(SeedsExclude)

    def test_import_config(self):
        from mmo.config import Config
        assert Config is not None

    def test_import_verbose(self):
        from mmo.verbose import Verbose
        assert Verbose is not None

    def test_import_ga_dc_functions(self):
        from mmo.ga_dc import (
            neighborhood,
            population_to_nparray,
            f0_to_nparray,
            population2xy,
        )
        for fn in (neighborhood, population_to_nparray, f0_to_nparray, population2xy):
            assert callable(fn)
