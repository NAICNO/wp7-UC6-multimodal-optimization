# Running Optimization Experiments

```{objectives}
- Run the demonstrator notebook interactively
- Execute optimization from Python scripts
- Manage long-running experiments with tmux
```

## Option 1: Interactive Notebook

The easiest way to get started is the demonstrator notebook:

```bash
cd ~/wp7-UC6-multimodal-optimization
source activate-mmo.sh
jupyter lab
```

Open `demonstrator.ipynb` and run cells interactively.

## Option 2: Python Script

For reproducible experiments, use Python scripts:

```python
#!/usr/bin/env python
"""Example: Optimize Himmelblau function (4 global optima)"""

import numpy as np
from mmo.minimize import MultiModalMinimizer
from mmo.domain import Domain
from cec2013.cec2013 import CEC2013, how_many_goptima

# Load CEC2013 benchmark function F4 (Himmelblau)
f = CEC2013(4)
dim = f.get_dimension()
n_optima = f.get_no_goptima()
print(f"Function: Himmelblau (F4)")
print(f"Dimension: {dim}")
print(f"Known optima: {n_optima}")

# Define search domain (get bounds for each dimension)
lb = [f.get_lbound(k) for k in range(dim)]
ub = [f.get_ubound(k) for k in range(dim)]
domain = Domain(boundary=[lb, ub])

# Create optimizer
optimizer = MultiModalMinimizer(
    f=f,  # Pass the CEC2013 object directly
    domain=domain,
    budget=50000,
    max_iter=50,
    verbose=1
)

# Run optimization
for iteration in optimizer:
    print(f"Iter {iteration.number}: {iteration.n_sol} solutions, {iteration.n_fev} evals")

# Results
print(f"\n=== Final Results ===")
print(f"Solutions found: {optimizer.n_sol}")
print(f"Function evaluations: {optimizer.n_fev}")

# Check how many global optima were found
count, seeds = how_many_goptima(optimizer.xy[:, :-1], f, 0.0001)
print(f"Global optima found: {count}/{n_optima}")
print(f"Peak Ratio: {count/n_optima:.1%}")
```

```{figure} ../images/shga_himmelblau_result.png
:alt: SHGA results on Himmelblau function showing 4 found optima and convergence
:width: 90%

Example output: SHGA finds all 4 global optima of the Himmelblau function. Left: contour plot with found solutions (blue stars). Right: convergence showing optima discovered per iteration.
```

## Running Long Experiments

For experiments that take hours, use tmux:

```bash
# Start a new tmux session
tmux new -s mmo_experiment

# Inside tmux: run your experiment
cd ~/wp7-UC6-multimodal-optimization
source activate-mmo.sh
python my_experiment.py > experiment.log 2>&1

# Detach from tmux: Ctrl+B, then D

# Later, reattach to check progress
tmux attach -t mmo_experiment

# Kill session when done
tmux kill-session -t mmo_experiment
```

## Saving Results

Save experiment results for later analysis:

```python
import pandas as pd
import numpy as np
import json
from datetime import datetime

# After optimization completes...

# Create results directory
exp_name = f"himmelblau_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
results_dir = f"results/{exp_name}"
os.makedirs(results_dir, exist_ok=True)

# Save solutions
solutions_df = pd.DataFrame(
    optimizer.xy,
    columns=[f'x_{i}' for i in range(optimizer.dim)] + ['f_value']
)
solutions_df.to_csv(f"{results_dir}/solutions.csv", index=False)
np.save(f"{results_dir}/solutions.npy", optimizer.xy)

# Save summary
summary = {
    'n_solutions': optimizer.n_sol,
    'n_function_evaluations': optimizer.n_fev,
    'n_iterations': optimizer.iteration,
    'budget': optimizer.budget
}
with open(f"{results_dir}/summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Results saved to {results_dir}")
```

## Experiment Checklist

Before running experiments:

- [ ] Activate environment: `source activate-mmo.sh`
- [ ] For long runs: use tmux
- [ ] Save results to timestamped directory

```{keypoints}
- Use the demonstrator notebook for interactive exploration
- Write Python scripts for reproducible experiments
- Use tmux for long-running experiments
- Always save results with timestamps for reproducibility
```
