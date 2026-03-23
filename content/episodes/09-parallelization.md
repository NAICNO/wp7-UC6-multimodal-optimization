# Parallelization

```{objectives}
- Understand inner-loop vs function-level parallelization strategies
- Use MultiModalMinimizerParallel for multi-core execution
- Know the expected speedups on NAIC VMs
```

## Overview

This document describes the parallelization implementation added to the Multi-Modal Optimization (MMO) project to utilize multi-core CPUs efficiently on NAIC VMs.

## Implementation Strategy

### Inner-Loop Parallelization (Selected Approach)

We implemented **inner-loop parallelization** within the Seed-Solve-Collect (SSC) algorithm:

- **Sequential outer loop**: Benchmark functions are processed one at a time
- **Parallel inner loop**: Within each function, seed processing is parallelized using `joblib`
- **Independent seed solves**: Each seed's local CMA-ES solve runs independently on separate CPU cores

### Files Modified/Created

1. **mmo/ssc_parallel.py** (NEW)
   - Parallelized version of Seed-Solve-Collect loop
   - Uses `joblib.Parallel` with 'loky' backend for process-based parallelism
   - Processes multiple seeds simultaneously across CPU cores

2. **mmo/minimize_parallel.py** (NEW)
   - Parallelized version of MultiModalMinimizer
   - Adds `n_jobs` parameter (default: -1 = all cores)
   - Drop-in replacement for sequential MultiModalMinimizer

3. **demonstrator.ipynb** (MODIFIED)
   - Updated imports to include parallel versions
   - Modified `run_shga_for_function` to use MultiModalMinimizerParallel
   - Added parallelization explanation markdown cell
   - Changed benchmark loop to sequential outer / parallel inner

## Performance Results

### Test Configuration
- **System**: NAIC VM with 16 CPU cores, NVIDIA L40S GPU
- **Test functions**: F4 (Himmelblau), F5 (Six-Hump Camel), F10 (Modified Rastrigin)
- **Max iterations**: 10 per function (for quick testing)

### Comparison: Sequential vs Parallel (Inner-Loop)

| Function | Sequential | Parallel | Speedup |
|----------|------------|----------|---------|
| F4 (Himmelblau) | 30.2s | 15.5s | 1.95x |
| F5 (Six-Hump Camel) | 13.6s | 11.2s | 1.21x |
| F10 (Modified Rastrigin) | 195.6s | 48.0s | **4.08x** |
| **Total** | **239.4s** | **74.7s** | **3.20x** |

**Key findings**:
- Overall speedup: **3.20x** on 16 cores
- Efficiency: **20.0%** (good for embarrassingly parallel workload with overhead)
- Best speedup on expensive functions (F10: 4.08x)
- All tests achieved **100% peak ratio** (correctness maintained)

### Why Inner-Loop Beats Function-Level Parallelization

We previously tried parallelizing across functions (running F4, F5, F10 simultaneously), achieving only **1.27x speedup** due to workload imbalance:
- F10 took 205s while F4/F5 took ~28s each
- Most cores idle while F10 runs sequentially
- Efficiency: only 7.9%

Inner-loop parallelization solves this:
- All 16 cores work on F10's seeds simultaneously
- 4.08x speedup on F10 specifically
- No idle cores waiting for expensive functions

## Technical Details

### joblib Configuration
```python
Parallel(n_jobs=-1, backend='loky')(
    delayed(process_seed_parallel)(...) for seed in seeds
)
```

- `n_jobs=-1`: Use all available CPU cores
- `backend='loky'`: Process-based parallelism (avoids GIL, proper isolation)
- Each worker process gets a copy of localsolver and function evaluator

### Thread Safety Considerations

1. **Function evaluation counting**: Each worker maintains its own `n_fev` counter; counts are aggregated in main process after parallel execution

2. **Solution collection**: Solutions are added sequentially after parallel seed processing completes (avoids race conditions)

3. **Budget tracking**: Each worker checks budget independently; main process handles final budget enforcement

4. **State isolation**: 'loky' backend creates separate processes, so no shared state between workers

## Usage

### In Python Scripts
```python
from mmo.minimize_parallel import MultiModalMinimizerParallel
from mmo.domain import Domain
from cec2013.cec2013 import CEC2013

f = CEC2013(4)  # Himmelblau
dim = f.get_dimension()
lb = [f.get_lbound(k) for k in range(dim)]
ub = [f.get_ubound(k) for k in range(dim)]
domain = Domain(boundary=[lb, ub])

# Use parallel version with all cores
for result in MultiModalMinimizerParallel(
    f=f, domain=domain, budget=50000,
    max_iter=50, verbose=1, n_jobs=-1  # n_jobs=-1 uses all cores
):
    print(f"Found {result.n_sol} solutions")
```

### In Jupyter Notebook
The demonstrator notebook automatically uses the parallel version. Just run cells normally - parallelization happens transparently.

## Testing

### Quick Test (3 functions)
```bash
cd /home/ubuntu/multi-modal-optimization
source activate-mmo.sh
python test_parallel_comparison.py
```

Runs F4, F5, F10 with both sequential and parallel versions for comparison (~5 minutes).

### Full Benchmark (F4-F14)
```bash
python test_full_benchmark_parallel.py
```

Runs all 2D+ CEC2013 functions (F4-F14) with parallel SHGA (~15-20 minutes).

## Expected Performance

On a 16-core NAIC VM:
- **Simple 2D functions** (F4, F5): 1.2-2.0x speedup
- **Complex functions** (F10, F6, F7): 3.0-4.5x speedup
- **Overall benchmark**: 3.0-3.5x speedup
- **Efficiency**: 18-25% (accounting for serial portions and overhead)

## Limitations

1. **Not GPU-accelerated**: Current implementation uses only CPU cores; GPU is not utilized
2. **Serial GA phase**: Initial population seeding with Genetic Algorithm is still sequential
3. **Diminishing returns**: Speedup plateaus beyond ~16-32 cores due to Amdahl's Law
4. **Memory overhead**: Each worker process duplicates some state (acceptable for this workload)

## Future Improvements

Potential areas for further optimization:
1. GPU-accelerated function evaluations for expensive benchmarks
2. Parallelize the GA population evolution phase
3. Adaptive n_jobs based on seed count (use fewer cores when seeds are scarce)
4. Shared memory backend for reduced memory footprint on large problems

## References

- Original SHGA paper: Johannsen et al. (2022) - "A scalable, hybrid genetic algorithm for continuous multimodal optimization in moderate dimensions"
- joblib documentation: https://joblib.readthedocs.io/
- CEC2013 benchmark suite: https://github.com/mikeagn/CEC2013

```{keypoints}
- Inner-loop parallelization (parallel seed solves) outperforms function-level parallelization
- Use `MultiModalMinimizerParallel` as a drop-in replacement with `n_jobs=-1`
- Expected 3-4x speedup on 16-core NAIC VMs for complex functions
- Correctness is preserved: all tests achieve the same peak ratios
- The GA phase remains sequential; only CMA-ES seed solves are parallelized
```
