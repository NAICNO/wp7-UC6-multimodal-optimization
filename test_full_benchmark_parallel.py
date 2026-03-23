#!/usr/bin/env python
"""Full F4-F14 benchmark with parallel SHGA."""
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
from mmo.minimize_parallel import MultiModalMinimizerParallel
from mmo.domain import Domain
from cec2013.cec2013 import CEC2013, how_many_goptima

def test_function(func_id, max_iter=50):
    """Test parallel SHGA on a single function."""
    start_time = time.time()
    
    f = CEC2013(func_id)
    info = f.get_info()
    dim = f.get_dimension()
    budget = info['maxfes']
    
    lb = [f.get_lbound(k) for k in range(dim)]
    ub = [f.get_ubound(k) for k in range(dim)]
    domain = Domain(boundary=[lb, ub])
    
    # Run parallel optimizer
    for result in MultiModalMinimizerParallel(
        f=f, domain=domain, budget=budget,
        max_iter=max_iter, verbose=0, n_jobs=-1
    ):
        pass
    
    # Check results
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
        'n_sol': result.n_sol,
        'n_fev': result.n_fev,
        'time': elapsed
    }

if __name__ == '__main__':
    print('='*80)
    print('Full CEC2013 Benchmark (F4-F14) with Parallel SHGA')
    print('='*80)
    print('Functions: F4-F14 (all 2D+ functions)')
    print('Max iterations: 50 per function')
    print('Parallelization: Inner-loop (seed processing) using all 16 cores')
    print('='*80)
    print()
    
    func_ids = range(4, 15)  # F4-F14
    results = []
    
    overall_start = time.time()
    
    for func_id in func_ids:
        print(f'Testing F{func_id}...', end=' ', flush=True)
        try:
            res = test_function(func_id, max_iter=50)
            results.append(res)
            status = 'PASS' if res['peak_ratio'] >= 1.0 else f"PARTIAL({res['peak_ratio']:.0%})"
            print(f"{res['time']:6.1f}s  {res['optima_found']}/{res['optima_total']} optima  [{status}]")
        except Exception as e:
            print(f'FAILED: {str(e)[:40]}')
    
    overall_time = time.time() - overall_start
    
    # Summary
    print()
    print('='*80)
    print('BENCHMARK SUMMARY')
    print('='*80)
    print(f"{'ID':<4} {'Function':<28} {'Dim':<4} {'Optima':<12} {'PR':<8} {'Time':<8}")
    print('-'*80)
    
    for r in results:
        print(f"F{r['func_id']:<3} {r['name']:<28} {r['dim']:<4} {r['optima_found']}/{r['optima_total']:<10} {r['peak_ratio']:.2f}    {r['time']:6.1f}s")
    
    print('-'*80)
    print(f"Total time: {overall_time:.1f}s ({overall_time/60:.1f} minutes)")
    print('='*80)
    
    # Critical tests
    print()
    print('CRITICAL TESTS (F4, F5, F10 - must achieve 100%):')
    for func_id in [4, 5, 10]:
        r = next((x for x in results if x['func_id'] == func_id), None)
        if r:
            status = 'PASS' if r['peak_ratio'] >= 1.0 else 'FAIL'
            print(f"  F{func_id} ({r['name']}): {r['peak_ratio']:.1%} [{status}]")
