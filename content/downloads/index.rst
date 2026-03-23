Downloads & Quick Reference
===========================

.. important::
   This repository is self-contained. After cloning with ``--recursive``, run ``./setup.sh`` to set up the environment automatically.

Setup Script
------------

Run after cloning to set up the environment:

.. code-block:: bash

   git clone --recursive https://github.com/NAICNO/wp7-UC6-multimodal-optimization.git
   cd wp7-UC6-multimodal-optimization
   ./setup.sh
   source activate-mmo.sh

Execution Modes
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mode
     - Description
   * - **Python API (Sequential)**
     - ``MultiModalMinimizer`` -- runs single-threaded
   * - **Python API (Parallel)**
     - ``MultiModalMinimizerParallel`` -- uses all CPU cores (3.2x speedup)
   * - **Interactive Notebook**
     - ``demonstrator.ipynb`` -- visual exploration with plots

Core API Parameters
-------------------

.. code-block:: python

   from mmo.minimize_parallel import MultiModalMinimizerParallel
   from mmo.domain import Domain

   optimizer = MultiModalMinimizerParallel(
       f=function,              # Callable: array → scalar
       domain=domain,           # Domain(boundary=[[lb], [ub]])
       budget=50000,            # Max function evaluations
       max_iter=50,             # Max outer iterations
       n_jobs=-1,               # -1 = all cores
       verbose=1                # 0=silent, 1=summary, 2=GA, 3=CMA
   )

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``f``
     - callable
     - **required**
     - Objective function (d-dim array → scalar)
   * - ``domain``
     - Domain
     - **required**
     - Search space bounds
   * - ``budget``
     - int
     - inf
     - Maximum function evaluations
   * - ``max_iter``
     - int
     - inf
     - Maximum outer iterations
   * - ``n_jobs``
     - int
     - -1
     - CPU cores (-1 = all, >0 = specific)
   * - ``verbose``
     - int
     - 0
     - Verbosity: 0=silent, 1=summary, 2=GA details

CEC2013 Benchmark Functions
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 10 35 15 15 25

   * - ID
     - Name
     - Dim
     - Optima
     - Budget
   * - 1
     - Five-Uneven-Peak
     - 1D
     - 2
     - 50k
   * - 2
     - Equal Maxima
     - 1D
     - 5
     - 50k
   * - 4
     - **Himmelblau**
     - 2D
     - 4
     - 50k
   * - 5
     - **Six-Hump Camel Back**
     - 2D
     - 2
     - 50k
   * - 6
     - **Shubert**
     - 2D
     - 18
     - 200k
   * - 7
     - **Vincent**
     - 2D
     - 36
     - 200k
   * - 8-20
     - Higher dimensions
     - 3-20D
     - 6-216
     - 400k

Load any function: ``CEC2013(id)`` where ``id = 1-20``

Example Commands
----------------

**Quick Test (Sequential)**

.. code-block:: python

   from mmo.minimize import MultiModalMinimizer
   from mmo.domain import Domain
   from cec2013.cec2013 import CEC2013

   f = CEC2013(4)  # Himmelblau
   dim = f.get_dimension()
   lb = [f.get_lbound(k) for k in range(dim)]
   ub = [f.get_ubound(k) for k in range(dim)]
   domain = Domain(boundary=[lb, ub])

   optimizer = MultiModalMinimizer(f=f, domain=domain, budget=50000, verbose=1)

   for result in optimizer:
       print(f"Iter {result.number}: {result.n_sol} solutions found")

**Parallel Optimization (Recommended)**

.. code-block:: python

   from mmo.minimize_parallel import MultiModalMinimizerParallel
   from mmo.domain import Domain
   from cec2013.cec2013 import CEC2013

   f = CEC2013(10)  # Modified Rastrigin
   dim = f.get_dimension()
   lb = [f.get_lbound(k) for k in range(dim)]
   ub = [f.get_ubound(k) for k in range(dim)]
   domain = Domain(boundary=[lb, ub])

   optimizer = MultiModalMinimizerParallel(
       f=f,
       domain=domain,
       budget=50000,
       n_jobs=-1,  # Use all CPU cores
       verbose=1
   )

   for result in optimizer:
       print(f"Solutions: {result.n_sol}, Evaluations: {result.n_fev}")

**Run Test Scripts**

.. code-block:: bash

   # Verify installation
   python test_mmo.py

   # Benchmark parallel vs sequential (F4, F5, F10: ~5 minutes)
   python test_parallel_comparison.py

   # Full benchmark suite (F4-F14: ~15-20 minutes)
   python test_full_benchmark_parallel.py

**Interactive Jupyter Notebook**

.. code-block:: bash

   # Start Jupyter Lab in tmux
   tmux new -s jupyter
   cd ~/wp7-UC6-multimodal-optimization
   source activate-mmo.sh
   jupyter lab --no-browser --ip=0.0.0.0 --port=8888
   # Detach with Ctrl+B, then D

   # Create SSH tunnel (local machine)
   ssh -v -N -L 8888:localhost:8888 -i /path/to/key.pem ubuntu@<VM_IP>

   # Open: http://localhost:8888/lab/tree/demonstrator.ipynb

Performance
-----------

**Parallelization Speedup (16-core NAIC VM):**

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Function
     - Speedup
     - Notes
   * - **Overall Average**
     - **3.2x**
     - Across F4-F14
   * - Himmelblau (F4)
     - 1.95x
     - 4 optima
   * - Six-Hump Camel (F5)
     - 1.21x
     - 2 optima
   * - Modified Rastrigin (F10)
     - 4.08x
     - 12 optima
   * - Efficiency
     - 20%
     - Of ideal 16x

**Peak Ratio (PR):** Percentage of global optima found (1.0 = perfect)

Common Issues
-------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Issue
     - Solution
   * - ``ModuleNotFoundError: cec2013``
     - Use ``source activate-mmo.sh`` (sets PYTHONPATH)
   * - ``ModuleNotFoundError: mmo``
     - Use ``source activate-mmo.sh`` (sets PYTHONPATH)
   * - Submodules not cloned
     - Run ``git submodule update --init --recursive``
   * - Import errors
     - Verify venv activated: ``which python``

For AI Coding Assistants
------------------------

If you're using an AI coding assistant (Claude Code, GitHub Copilot, Cursor, etc.), the repository includes machine-readable instruction files:

- ``AGENT.md`` -- Markdown format (human and agent readable)
- ``AGENT.yaml`` -- YAML format (structured data for programmatic parsing)

These files contain step-by-step instructions that agents can follow to:

1. Set up the environment on the VM
2. Run the Jupyter notebook
3. Execute benchmark tests
4. Verify optimization results

**Quick prompt for your AI assistant:**

.. code-block:: text

   Read AGENT.md and help me run the multi-modal optimization
   demonstrator on my NAIC VM.
   VM IP: <your_vm_ip>
   SSH Key: <path_to_your_key.pem>

The agent will execute the setup and run experiments based on the structured instructions.

References
----------

**Paper:** Johannsen et al. (2022), "A scalable, hybrid genetic algorithm for continuous multimodal optimization in moderate dimensions", Nordic Machine Intelligence

**Algorithm:** SHGA (Scalable Hybrid Genetic Algorithm) - combines Deterministic Crowding GA (global search) with CMA-ES (local refinement)
