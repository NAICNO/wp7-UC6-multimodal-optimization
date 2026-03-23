#!/usr/bin/env python
"""Compare sequential vs parallel SHGA performance."""
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import multiprocessing
from mmo.minimize import MultiModalMinimizer
from mmo.minimize_parallel import MultiModalMinimizerParallel
from mmo.domain import Domain
from cec2013.cec2013 import CEC2013, how_many_goptima

def test_function(func_id, max_iter, parallel=False, n_jobs=-1):
    """Test SHGA on a single function."""
    f = CEC2013(func_id)
    info = f.get_info()
    dim = f.get_dimension()
    budget = info['maxfes']
    
    lb = [f.get_lbound(k) for k in range(dim)]
    ub = [f.get_ubound(k) for k in range(dim)]
    domain = Domain(boundary=[lb, ub])
    
    start_time = time.time()
    
    if parallel:
        optimizer = MultiModalMinimizerParallel(
            f=f, domain=domain, budget=budget,
            max_iter=max_iter, verbose=0, n_jobs=n_jobs
        )
    else:
        optimizer = MultiModalMinimizer(
            f=f, domain=domain, budget=budget,
            max_iter=max_iter, verbose=0
        )
    
    for result in optimizer:
        pass
    
    elapsed = time.time() - start_time
    
    # Check results
    accuracy = 0.0001
    count, seeds = how_many_goptima(result.x, f, accuracy)
    peak_ratio = count / info['nogoptima'] if info['nogoptima'] > 0 else 0
    
    return {
        'func_id': func_id,
        'name': info['name'],
        'dim': dim,
        'optima_found': count,
        'optima_total': info['nogoptima'],
        'peak_ratio': peak_ratio,
        'n_sol': result.n_sol,
        'n_fev': result.n_fev,
        'time': elapsed,
        'parallel': parallel
    }

if __name__ == '__main__':
    n_cores = multiprocessing.cpu_count()
    print('='*80)
    print('SHGA Sequential vs Parallel Comparison')
    print('='*80)
    print(f'System: {n_cores} CPU cores')
    print(f'Test functions: F4 (Himmelblau), F5 (Six-Hump Camel), F10 (Modified Rastrigin)')
    print(f'Max iterations: 10 (reduced for quick test)')
    print('='*80)
    print()
    
    test_funcs = [4, 5, 10]
    max_iter = 10
    
    results = []
    
    # Run sequential tests
    print('Running SEQUENTIAL tests...')
    print('-'*80)
    for func_id in test_funcs:
        res = test_function(func_id, max_iter, parallel=False)
        results.append(res)
        print(f"F{func_id:2d} {res['name']:<25} {res['time']:6.1f}s  PR={res['peak_ratio']:.0%} ({res['optima_found']}/{res['optima_total']})")
    
    seq_time = sum(r['time'] for r in results if not r['parallel'])
    print(f"\nTotal sequential time: {seq_time:.1f}s")
    print()
    
    # Run parallel tests
    print('Running PARALLEL tests (inner-loop parallelization)...')
    print('-'*80)
    for func_id in test_funcs:
        res = test_function(func_id, max_iter, parallel=True, n_jobs=-1)
        results.append(res)
        print(f"F{func_id:2d} {res['name']:<25} {res['time']:6.1f}s  PR={res['peak_ratio']:.0%} ({res['optima_found']}/{res['optima_total']})")
    
    par_time = sum(r['time'] for r in results if r['parallel'])
    print(f"\nTotal parallel time: {par_time:.1f}s")
    print()
    
    # Summary
    print('='*80)
    print('PERFORMANCE SUMMARY')
    print('='*80)
    print(f'Sequential time: {seq_time:.1f}s')
    print(f'Parallel time:   {par_time:.1f}s')
    print(f'Speedup:         {seq_time/par_time:.2f}x')
    print(f'Efficiency:      {100*seq_time/par_time/n_cores:.1f}%')
    print()
    
    # Detailed comparison
    print('Function-level comparison:')
    print(f"{'Func':<25} {'Sequential':<12} {'Parallel':<12} {'Speedup':<10}")
    print('-'*80)
    for func_id in test_funcs:
        seq_res = next(r for r in results if r['func_id'] == func_id and not r['parallel'])
        par_res = next(r for r in results if r['func_id'] == func_id and r['parallel'])
        speedup = seq_res['time'] / par_res['time']
        print(f"F{func_id:2d} {seq_res['name']:<20} {seq_res['time']:6.1f}s      {par_res['time']:6.1f}s      {speedup:.2f}x")
    
    print('='*80)
