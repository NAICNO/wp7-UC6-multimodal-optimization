# AI Agent Instructions for Multi-Modal Optimization (SHGA)

This file provides structured instructions for AI coding assistants (Claude, Copilot, etc.) to set up and run the SHGA multimodal optimization algorithm.

> **Also available:** `AGENT.yaml` - Same instructions in YAML format for programmatic parsing.

## Algorithm Overview

The SHGA (Scalable Hybrid Genetic Algorithm) is designed to find multiple local and global optima in continuous optimization problems. It combines:
- **Global Search:** Deterministic crowding genetic algorithm for exploration
- **Local Refinement:** CMA-ES for fine-tuning solutions


Reference: Johannsen et al. (2022) - "A scalable, hybrid genetic algorithm for continuous multimodal optimization in moderate dimensions" (Nordic Machine Intelligence)

## Prerequisites

- SSH access to NAIC Orchestrator VM
- VM IP address and SSH key path

## Step 1: Connect to VM and Check Environment

```bash
# SSH to VM
ssh -i <SSH_KEY_PATH> ubuntu@<VM_IP>

# Check if module system is available (NAIC VMs)
module avail python/3 2>/dev/null || echo "No module system"

# Check GPU availability
nvidia-smi 2>/dev/null || echo "No NVIDIA GPU"

# Check Python version
python3 --version
```

## Step 2: Initialize VM (first time only)

Run the initialization script:
```bash
./vm-init.sh
```

Or manually:

**Option A: If module system is available (recommended):**
```bash
module load Python/3.11.5-GCCcore-13.2.0
python3 --version
```

**Option B: If no module system:**
```bash
sudo apt update -y
sudo apt install -y build-essential git python3-dev python3-venv python3-pip libssl-dev zlib1g-dev htop tmux
```

## Step 3: Setup Environment

```bash
# Clone repository
git clone https://github.com/NAICNO/wp7-UC6-multimodal-optimization.git

# Navigate to project directory
cd multi-modal-optimization

# Run setup script
./setup.sh

# Activate environment
source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/benchmarks/CEC2013/python3"
```

## Step 4: Verify Installation

```bash
# Test MMO module
python -c "from mmo.minimize import MultiModalMinimizer; print('MMO: OK')"

# Test CEC2013 benchmarks
python -c "from cec2013.cec2013 import CEC2013; f = CEC2013(4); print(f'CEC2013: OK - {f.get_info()[\"name\"]}')"
```

## Step 5: Run Optimization

### Quick test (Himmelblau function):

**Parallel version (recommended for multi-core VMs):**
```python
from mmo.minimize_parallel import MultiModalMinimizerParallel
from mmo.domain import Domain
from cec2013.cec2013 import CEC2013

# Load benchmark
benchmark = CEC2013(4)  # Himmelblau
dim = benchmark.get_dimension()
lb = [benchmark.get_lbound(k) for k in range(dim)]
ub = [benchmark.get_ubound(k) for k in range(dim)]
domain = Domain(boundary=[lb, ub])

# Optimize (uses all CPU cores)
optimizer = MultiModalMinimizerParallel(
    f=benchmark,
    domain=domain,
    budget=50000,
    max_iter=50,
    n_jobs=-1,  # Use all cores
    verbose=1
)

for result in optimizer:
    print(f"Iter {result.number}: {result.n_sol} solutions")

print(f"Found {optimizer.n_sol} solutions")
```

**Sequential version:**
```python
from mmo.minimize import MultiModalMinimizer
from mmo.domain import Domain
from cec2013.cec2013 import CEC2013

# Load benchmark
benchmark = CEC2013(4)  # Himmelblau
dim = benchmark.get_dimension()
lb = [benchmark.get_lbound(k) for k in range(dim)]
ub = [benchmark.get_ubound(k) for k in range(dim)]
domain = Domain(boundary=[lb, ub])

# Optimize
optimizer = MultiModalMinimizer(
    f=benchmark,
    domain=domain,
    budget=50000,
    verbose=1
)

for result in optimizer:
    print(f"Iter {result.number}: {result.n_sol} solutions")

print(f"Found {optimizer.n_sol} solutions")
```



## Step 6: Jupyter Notebook (Interactive)

```bash
# Start Jupyter Lab (on VM)
jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --NotebookApp.password=''
```

Create SSH tunnel from local machine:
```bash
ssh -N -L 8888:localhost:8888 -i <SSH_KEY_PATH> ubuntu@<VM_IP>
```

Open: `http://localhost:8888/lab/tree/demonstrator.ipynb`

## CEC2013 Benchmark Functions

| ID | Name | Dim | Optima | Budget |
|----|------|-----|--------|--------|
| 1 | Five-Uneven-Peak | 1 | 2 | 50,000 |
| 2 | Equal Maxima | 1 | 5 | 50,000 |
| 3 | Uneven Decreasing Max. | 1 | 1 | 50,000 |
| 4 | Himmelblau | 2 | 4 | 50,000 |
| 5 | Six-Hump Camel Back | 2 | 2 | 50,000 |
| 6 | Shubert | 2 | 18 | 200,000 |
| 7 | Vincent | 2 | 36 | 200,000 |
| 8-20 | Higher dimensional | 3-20 | 6-216 | 400,000 |

## Directory Structure

```
multi-modal-optimization/
├── setup.sh                    # Environment setup (run after clone)
├── vm-init.sh                  # VM initialization (run once)
├── requirements.txt            # Python dependencies
├── PARALLELIZATION.md          # Parallelization documentation (NEW)
├── mmo/                        # Core SHGA algorithm
│   ├── minimize.py             # Main MultiModalMinimizer class
│   ├── minimize_parallel.py    # Parallel version (NEW)
│   ├── domain.py               # Search space
│   ├── ga_dc.py                # Genetic Algorithm
│   ├── cma.py                  # CMA-ES local solver
│   ├── ssc.py                  # Seed / solve / collect loop
│   └── ssc_parallel.py         # Parallel SSC loop (NEW)
├── benchmarks/CEC2013/python3/ # Benchmark suite
├── data/                       # Benchmark data files
├── demonstrator.ipynb          # Interactive notebook (uses parallel)
├── test_parallel_comparison.py # Performance benchmarks (NEW)
└── content/                    # Documentation
```

## Parallelization (NEW)

The algorithm now supports multi-core parallelization for improved performance on multi-core VMs.

### Features
- **Inner-loop parallelization**: Seeds processed simultaneously across CPU cores
- **Automatic**: Notebook uses parallel version by default
- **3-4x speedup**: On 16-core VM (e.g., NAIC L40S instances)

### Usage
- Use `MultiModalMinimizerParallel` with `n_jobs=-1` for all cores
- Use `MultiModalMinimizer` for sequential execution (original)
- See [PARALLELIZATION.md](PARALLELIZATION.md) for detailed benchmarks

### Performance on NAIC VM (16 cores)
- F4 (Himmelblau): 1.95x speedup
- F5 (Six-Hump Camel): 1.21x speedup
- F10 (Modified Rastrigin): 4.08x speedup
- Overall: 3.20x speedup with 100% correctness maintained

## Performance Metrics

- **Peak Ratio (PR):** Number of global optima found / Total known optima
  - PR = 1.0 means all optima found
  - PR > 0.7 is good performance
  - PR = 0.0 means no optima found

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: cec2013` | Set PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)/benchmarks/CEC2013/python3"` |
| `ModuleNotFoundError: mmo` | Ensure venv activated and PYTHONPATH includes project root |
| Python version too old | `module load Python/3.11.5-GCCcore-13.2.0` |


## Verification

After running, check:
1. Solutions found > 0
2. Peak Ratio reported (if using CEC2013 benchmarks)
3. No Python errors in output
