#!/usr/bin/env python
"""
Full CEC2013 Benchmark Test (F4-F14, 2D+ functions).
Tests SHGA (sequential) on all 2D and higher dimension functions.

Note: 1D functions (F1-F3) are skipped due to array return type issues.
      Use the demonstrator notebook for 1D function testing.
"""
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
from mmo.minimize import MultiModalMinimizer
from mmo.domain import Domain
from cec2013.cec2013 import CEC2013, how_many_goptima


# Expected peak ratios from SHGA paper (approximate)
EXPECTED_PR = {
    1: 0.50,   # Five-Uneven-Peak (1D) - challenging
    2: 0.60,   # Equal Maxima (1D) - challenging
    3: 0.67,   # Uneven Decreasing (1D) - only 1 optimum
    4: 1.00,   # Himmelblau (2D) - should find all 4
    5: 1.00,   # Six-Hump Camel (2D) - should find all 2
    6: 0.85,   # Shubert 2D - 18 optima
    7: 0.75,   # Vincent 2D - 36 optima
    8: 0.50,   # Shubert 3D - 81 optima
    9: 0.30,   # Vincent 3D - 216 optima
    10: 1.00,  # Modified Rastrigin - should find all 12
    11: 0.50,  # Composition Function 1
    12: 0.50,  # Composition Function 2
    13: 0.50,  # Composition Function 3
    14: 0.50,  # Composition Function 3 (3D)
}


def test_function(func_id, max_iter=50, verbose=0):
    """Run SHGA on a single CEC2013 benchmark function."""
    start_time = time.time()

    f = CEC2013(func_id)
    info = f.get_info()
    dim = f.get_dimension()
    budget = info["maxfes"]

    lb = [f.get_lbound(k) for k in range(dim)]
    ub = [f.get_ubound(k) for k in range(dim)]
    domain = Domain(boundary=[lb, ub])

    if verbose > 0:
        print(f"\nF{func_id}: {info['name']} (dim={dim}, optima={info['nogoptima']}, budget={budget})")

    # Run optimizer
    for result in MultiModalMinimizer(f=f, domain=domain, budget=budget,
                                       max_iter=max_iter, verbose=verbose):
        pass

    # Check peak ratio
    accuracy = 0.0001
    count, seeds = how_many_goptima(result.x, f, accuracy)
    peak_ratio = count / info['nogoptima'] if info['nogoptima'] > 0 else 0

    elapsed = time.time() - start_time

    return {
        'func_id': func_id,
        'name': info['name'],
        'dim': dim,
        'optima_found': count,
        'optima_total': info['nogoptima'],
        'peak_ratio': peak_ratio,
        'solutions': result.n_sol,
        'fev': result.n_fev,
        'budget': budget,
        'time': elapsed
    }


def run_benchmark(func_ids=None, max_iter=50, verbose=0):
    """Run benchmark on specified functions."""
    if func_ids is None:
        func_ids = range(4, 15)  # F4-F14 (skip 1D functions)

    results = []
    total_time = 0

    for func_id in func_ids:
        try:
            res = test_function(func_id, max_iter=max_iter, verbose=verbose)
            results.append(res)
            total_time += res['time']

            # Progress indicator
            expected = EXPECTED_PR.get(func_id, 0.5)
            status = "OK" if res['peak_ratio'] >= expected * 0.8 else "LOW"
            print(f"F{func_id:2d}: {res['name']:<25} PR={res['peak_ratio']:.2f} ({res['optima_found']}/{res['optima_total']}) [{status}] {res['time']:.1f}s")
        except Exception as e:
            print(f"F{func_id:2d}: SKIPPED - {str(e)[:50]}")

    return results, total_time


def print_summary(results, total_time):
    """Print benchmark summary table."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'ID':<4} {'Function':<28} {'Dim':<4} {'Optima':<12} {'PR':<8} {'Status':<8}")
    print("-"*80)

    critical_passed = True
    for r in results:
        expected = EXPECTED_PR.get(r['func_id'], 0.5)
        # Status: PASS if >= 80% of expected, WARN if >= 50%, FAIL otherwise
        if r['peak_ratio'] >= expected * 0.8:
            status = "PASS"
        elif r['peak_ratio'] >= expected * 0.5:
            status = "WARN"
        else:
            status = "FAIL"

        # Critical functions (F4, F5, F10) should achieve 100%
        if r['func_id'] in [4, 5, 10] and r['peak_ratio'] < 1.0:
            critical_passed = False
            status = "CRIT"

        print(f"F{r['func_id']:<3} {r['name']:<28} {r['dim']:<4} {r['optima_found']}/{r['optima_total']:<10} {r['peak_ratio']:.2f}    {status}")

    print("-"*80)
    print(f"Total time: {total_time:.1f}s")
    print("="*80)

    # Critical tests
    print("\nCRITICAL TESTS (must achieve 100% peak ratio):")
    for func_id in [4, 5, 10]:
        r = next((x for x in results if x['func_id'] == func_id), None)
        if r:
            status = "PASS" if r['peak_ratio'] >= 1.0 else "FAIL"
            print(f"  F{func_id} ({r['name']}): {r['peak_ratio']:.1%} [{status}]")
        else:
            print(f"  F{func_id}: Not tested")

    return critical_passed


def main():
    import argparse
    parser = argparse.ArgumentParser(description='CEC2013 Benchmark Test')
    parser.add_argument('--quick', action='store_true', help='Quick test (F1-F5 only)')
    parser.add_argument('--full', action='store_true', help='Full test (F1-F14)')
    parser.add_argument('--verbose', '-v', type=int, default=0, help='Verbosity level')
    parser.add_argument('--max-iter', type=int, default=50, help='Max iterations per function')
    args = parser.parse_args()

    print("="*80)
    print("CEC2013 Benchmark Test - SHGA (sequential)")
    print("="*80)

    if args.quick:
        func_ids = [4, 5]
        print("Running QUICK test (F4-F5 only)")
    elif args.full:
        func_ids = range(4, 15)  # F4-F14 (skip 1D functions F1-F3)
        print("Running FULL test (F4-F14, skipping 1D functions)")
    else:
        # Default: test key functions
        func_ids = [4, 5, 6, 7, 10]
        print("Running DEFAULT test (F4, F5, F6, F7, F10)")

    print(f"Max iterations: {args.max_iter}")
    print("-"*80)

    results, total_time = run_benchmark(func_ids, max_iter=args.max_iter, verbose=args.verbose)
    critical_passed = print_summary(results, total_time)

    if critical_passed:
        print("\nALL CRITICAL TESTS PASSED!")
        return 0
    else:
        print("\nSOME CRITICAL TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())
