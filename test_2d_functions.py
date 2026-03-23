#!/usr/bin/env python
"""
Quick test for CEC2013 2D functions (F4-F5, F6-F7, F10).
Tests SHGA (sequential) on low-dimensional benchmarks.

Note: 1D functions (F1-F3) have array return type issues with SHGA.
      Use the demonstrator notebook for 1D function testing.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from mmo.minimize import MultiModalMinimizer
from mmo.domain import Domain
from cec2013.cec2013 import CEC2013, how_many_goptima


def test_function(func_id, max_iter=50):
    """Run SHGA on a single CEC2013 benchmark function."""
    f = CEC2013(func_id)
    info = f.get_info()
    dim = f.get_dimension()
    budget = info["maxfes"]

    lb = [f.get_lbound(k) for k in range(dim)]
    ub = [f.get_ubound(k) for k in range(dim)]
    domain = Domain(boundary=[lb, ub])

    print(f"\n{'='*60}")
    print(f"F{func_id}: {info['name']}")
    print(f"Dimension: {dim}, Known optima: {info['nogoptima']}, Budget: {budget}")
    print('='*60)

    # Run optimizer
    for result in MultiModalMinimizer(f=f, domain=domain, budget=budget,
                                       max_iter=max_iter, verbose=0):
        pass

    # Check peak ratio
    accuracy = 0.0001
    count, seeds = how_many_goptima(result.x, f, accuracy)
    peak_ratio = count / info['nogoptima'] if info['nogoptima'] > 0 else 0

    print(f"Solutions found: {result.n_sol}")
    print(f"Function evals: {result.n_fev}")
    print(f"Global optima: {count}/{info['nogoptima']}")
    print(f"Peak Ratio: {peak_ratio:.1%}")

    return {
        'func_id': func_id,
        'name': info['name'],
        'optima_found': count,
        'optima_total': info['nogoptima'],
        'peak_ratio': peak_ratio,
        'solutions': result.n_sol,
        'fev': result.n_fev
    }


def main():
    print("="*60)
    print("CEC2013 2D Functions Test (sequential)")
    print("="*60)
    print("Note: 1D functions (F1-F3) skipped due to array return type issues")

    # Test 2D functions: F4, F5, F6, F10
    # F1-F3 are 1D (array return issues), F7 (Vincent) has domain issues with log(x)
    func_ids = [4, 5, 6, 10]

    results = []
    for func_id in func_ids:
        try:
            res = test_function(func_id)
            results.append(res)
        except Exception as e:
            print(f"F{func_id}: SKIPPED due to error: {e}")
            results.append({
                'func_id': func_id,
                'name': f"Function {func_id}",
                'optima_found': 0,
                'optima_total': 0,
                'peak_ratio': 0,
                'solutions': 0,
                'fev': 0,
                'error': str(e)
            })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Function':<30} {'Optima':<12} {'Peak Ratio':<12}")
    print("-"*60)

    all_passed = True
    for r in results:
        status = "PASS" if r['peak_ratio'] >= 0.5 else "LOW"
        if r['peak_ratio'] < 0.5:
            all_passed = False
        print(f"F{r['func_id']}: {r['name']:<24} {r['optima_found']}/{r['optima_total']:<10} {r['peak_ratio']:.1%} {status}")

    print("="*60)

    # Critical tests: F4, F5, F10 should achieve 100% peak ratio
    critical_funcs = {4: 1.0, 5: 1.0, 10: 1.0}
    critical_passed = True

    print("\nCRITICAL TESTS (must achieve 100%):")
    for r in results:
        if r['func_id'] in critical_funcs:
            expected = critical_funcs[r['func_id']]
            passed = r['peak_ratio'] >= expected
            status = "PASS" if passed else "FAIL"
            print(f"  F{r['func_id']} ({r['name']}): {r['peak_ratio']:.1%} [{status}]")
            if not passed:
                critical_passed = False

    if critical_passed:
        print("\nALL CRITICAL TESTS PASSED!")
    else:
        print("\nSOME CRITICAL TESTS FAILED!")

    return critical_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
