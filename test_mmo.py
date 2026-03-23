#!/usr/bin/env python
"""
Quick verification test for Multi-Modal Optimization (MMO).
Tests SHGA on F4 (Himmelblau) and F5 (Six-Hump Camel Back).

For more comprehensive tests, see:
- test_f1_f5.py          - All simple functions (F1-F5)
- test_full_benchmark.py - Full benchmark (F1-F14)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from mmo.minimize import MultiModalMinimizer
from mmo.domain import Domain
from cec2013.cec2013 import CEC2013, how_many_goptima


def test_function(func_id, expected_optima):
    """Run SHGA on a CEC2013 benchmark and check peak ratio."""
    f = CEC2013(func_id)
    info = f.get_info()
    dim = f.get_dimension()

    lb = [f.get_lbound(k) for k in range(dim)]
    ub = [f.get_ubound(k) for k in range(dim)]
    domain = Domain(boundary=[lb, ub])

    print(f"\n=== Testing F{func_id}: {info['name']} ===")
    print(f"Dimension: {dim}, Known optima: {info['nogoptima']}")

    # Run optimizer
    for result in MultiModalMinimizer(f=f, domain=domain, budget=info['maxfes'],
                                       max_iter=50, verbose=0):
        pass

    # Check peak ratio
    count, seeds = how_many_goptima(result.x, f, 0.0001)
    pr = count / info['nogoptima']

    print(f"Solutions found: {result.n_sol}")
    print(f"Function evals: {result.n_fev}")
    print(f"Global optima: {count}/{info['nogoptima']}")
    print(f"Peak Ratio: {pr:.1%}")

    return count >= expected_optima


def main():
    print("Multi-Modal Optimization - Quick Test")
    print("="*50)

    # Test F4 (Himmelblau - 4 optima) and F5 (Six-Hump Camel - 2 optima)
    f4_pass = test_function(4, 4)  # Expect 100% (4/4)
    f5_pass = test_function(5, 2)  # Expect 100% (2/2)

    print("\n" + "="*50)
    if f4_pass and f5_pass:
        print("ALL TESTS PASSED!")
        return 0
    else:
        print("SOME TESTS FAILED")
        if not f4_pass:
            print("- F4 (Himmelblau) did not find all 4 optima")
        if not f5_pass:
            print("- F5 (Six-Hump Camel) did not find all 2 optima")
        return 1


if __name__ == "__main__":
    exit(main())
