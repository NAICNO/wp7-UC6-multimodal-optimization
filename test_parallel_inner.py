#!/usr/bin/env python
"""Test inner-loop parallelization of SHGA."""
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
from mmo.minimize_parallel import MultiModalMinimizerParallel
from mmo.domain import Domain
from cec2013.cec2013 import CEC2013, how_many_goptima

def test_function_parallel(func_id, max_iter=10, n_jobs=-1):
    """Test parallelized SHGA on a single function."""
    f = CEC2013(func_id)
    info = f.get_info()
    dim = f.get_dimension()
    budget = info['maxfes']
    
    lb = [f.get_lbound(k) for k in range(dim)]
    ub = [f.get_ubound(k) for k in range(dim)]
    domain = Domain(boundary=[lb, ub])
    
    print(f"Testing F{func_id}: {info['name']} (dim={dim}, optima={info['nogoptima']}, budget={budget})")
    print(f"Using n_jobs={n_jobs if n_jobs != -1 else 'all cores'}")
    
    start_time = time.time()
    
    # Run parallelized optimizer
    for result in MultiModalMinimizerParallel(
        f=f, domain=domain, budget=budget,
        max_iter=max_iter, verbose=0, n_jobs=n_jobs
    ):
        pass
    
    elapsed = time.time() - start_time
    
    # Check results
    accuracy = 0.0001
    count, seeds = how_many_goptima(result.x, f, accuracy)
    peak_ratio = count / info['nogoptima'] if info['nogoptima'] > 0 else 0
    
    print(f"Results:")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Solutions found: {result.n_sol}")
    print(f"  Function evals: {result.n_fev}")
    print(f"  Global optima: {count}/{info['nogoptima']}")
    print(f"  Peak Ratio: {peak_ratio:.1%}")
    print(f"  Status: {'PASS' if peak_ratio >= 1.0 else 'PARTIAL'}")
    
    return elapsed, peak_ratio

if __name__ == '__main__':
    print('='*70)
    print('SHGA Inner-Loop Parallelization Test')
    print('='*70)
    print()
    
    # Test F4 (Himmelblau) - quick test
    print('Testing F4 (Himmelblau) with 10 iterations...')
    elapsed, pr = test_function_parallel(4, max_iter=10, n_jobs=-1)
    
    print()
    print('='*70)
    print(f'Test completed in {elapsed:.1f}s with peak ratio {pr:.1%}')
    print('='*70)
