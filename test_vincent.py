#!/usr/bin/env python
"""Test Vincent functions (F7, F9) which previously failed with math domain error."""
import warnings
warnings.filterwarnings("ignore")

from mmo.minimize import MultiModalMinimizer
from mmo.domain import Domain
from cec2013.cec2013 import CEC2013, how_many_goptima

def test_vincent(func_id, max_iter=20):
    """Test Vincent function with domain projection."""
    f = CEC2013(func_id)
    info = f.get_info()
    dim = f.get_dimension()

    lb = [f.get_lbound(k) for k in range(dim)]
    ub = [f.get_ubound(k) for k in range(dim)]
    domain = Domain(boundary=[lb, ub])

    print(f"\nF{func_id}: {info['name']} (dim={dim})")
    print(f"Domain: {lb} to {ub}")
    print(f"Known optima: {info['nogoptima']}")

    # Use full budget to properly test
    budget = info['maxfes']

    try:
        for result in MultiModalMinimizer(f=f, domain=domain, budget=budget,
                                          max_iter=max_iter, verbose=0):
            pass

        count, seeds = how_many_goptima(result.x, f, 0.0001)
        pr = count / info['nogoptima']

        print(f"Solutions found: {result.n_sol}")
        print(f"Global optima: {count}/{info['nogoptima']}")
        print(f"Peak Ratio: {pr:.1%}")
        print(f"STATUS: PASS (no domain error)")
        return True
    except Exception as e:
        print(f"STATUS: FAIL - {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Testing Vincent Functions (F7, F9) with Domain Projection Fix")
    print("="*60)

    f7_pass = test_vincent(7)   # Vincent 2D
    f9_pass = test_vincent(9)   # Vincent 3D

    print("\n" + "="*60)
    if f7_pass and f9_pass:
        print("ALL VINCENT TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
        if not f7_pass:
            print("  - F7 (Vincent 2D) failed")
        if not f9_pass:
            print("  - F9 (Vincent 3D) failed")
