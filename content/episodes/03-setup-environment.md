# Setting Up the Environment

```{objectives}
- Connect to your VM via SSH
- Initialize a fresh VM with required packages
- Clone the repository on the VM
- Set up Python environment
- Verify the installation works
```

## 1. Connect to Your VM

Connect to your VM using SSH (see Episode 02 for Windows-specific instructions):

````{tabs}
```{tab} macOS / Linux / Git Bash
chmod 600 /path/to/your-key.pem
ssh -i /path/to/your-key.pem ubuntu@<VM_IP>
```

```{tab} Windows (PowerShell)
ssh -i "C:\Users\YourName\Downloads\your-key.pem" ubuntu@<VM_IP>
```
````

```{note}
**Windows users**: If you see "Permissions for key are too open", fix the key permissions first. See Episode 02, Step 7 for detailed instructions. Git Bash is recommended — it supports `chmod` natively.
```

## 2. Initialize Fresh VM (Run Once)

On a fresh NAIC VM, install required system packages:

```bash
sudo apt update -y
sudo apt install -y build-essential git python3-dev python3-venv python3-pip libssl-dev zlib1g-dev htop tmux
```

This installs:
- `git` - For cloning the repository
- `build-essential` - Compiler toolchain (gcc, make)
- `python3-dev`, `python3-venv`, `python3-pip` - Python development tools
- `libssl-dev`, `zlib1g-dev` - Required for building Python packages
- `htop`, `tmux` - Monitoring and session management

## 3. Clone and Setup

```bash
git clone --recursive https://github.com/NAICNO/wp7-UC6-multimodal-optimization.git
cd wp7-UC6-multimodal-optimization
./setup.sh
source activate-mmo.sh
```

> **Note:** The `--recursive` flag clones the CEC2013 benchmark submodule. If you forgot it, `setup.sh` will initialize submodules automatically.

The `setup.sh` script automatically:
- Detects NAIC module system (if available)
- Creates a Python virtual environment
- Installs all dependencies from `requirements.txt`
- Creates `activate-mmo.sh` wrapper that sets PYTHONPATH correctly

## 4. Verify Installation

After running `source activate-mmo.sh`, verify the installation:

```bash
# Test MMO module
python -c "from mmo.minimize import MultiModalMinimizer; print('MMO: OK')"

# Test CEC2013 benchmarks
python -c "from cec2013.cec2013 import CEC2013; f = CEC2013(4); print(f'CEC2013: OK - Himmelblau (dim={f.get_dimension()}, optima={f.get_no_goptima()})')"
```

> **Note:** The `activate-mmo.sh` script sets PYTHONPATH automatically, so you don't need to export it manually.

## 5. Start Jupyter Lab (Optional)

For interactive exploration, start Jupyter Lab:

```bash
# Use tmux for persistence
tmux new -s jupyter
cd ~/wp7-UC6-multimodal-optimization
source activate-mmo.sh
jupyter lab --no-browser --ip=127.0.0.1 --port=8888 
# Detach with Ctrl+B, then D
```

## 6. Create SSH Tunnel (on your local machine)

To access Jupyter Lab from your local browser, create an SSH tunnel. Open a **new terminal** on your local machine (not the VM):

````{tabs}
```{tab} macOS / Linux / Git Bash
# Verbose mode (recommended - shows connection status)
ssh -v -N -L 8888:localhost:8888 -i /path/to/your-key.pem ubuntu@<VM_IP>
```

```{tab} Windows (PowerShell)
ssh -v -N -L 8888:localhost:8888 -i "C:\Users\YourName\Downloads\your-key.pem" ubuntu@<VM_IP>
```
````

> **Note:** The tunnel will appear to "hang" after connecting - this is normal! It means the tunnel is active. Keep the terminal open while using Jupyter.

Then navigate to: **http://localhost:8888/lab/tree/demonstrator.ipynb**

## Project Structure

After cloning and running `./setup.sh`, you will have:

```
wp7-UC6-multimodal-optimization/
├── setup.sh                           # Python environment setup
├── activate-mmo.sh                    # Environment activation (created by setup.sh)
├── requirements.txt                   # Python dependencies
├── test_mmo.py                        # Quick algorithm test
├── mmo/                               # Core SHGA algorithm
│   ├── minimize.py                    # Main MultiModalMinimizer class
│   ├── config.py                      # Algorithm configuration
│   ├── domain.py                      # Search space definition
│   ├── function.py                    # Function wrapper
│   ├── solutions.py                   # Solution tracking
│   ├── ga_dc.py                       # Genetic Algorithm (Deterministic Crowding)
│   ├── ssc.py                         # Sequential Seed-Solve-Collect
│   └── cma.py                         # CMA-ES local solver
├── benchmarks/CEC2013/python3/        # CEC2013 benchmark suite
├── data/                              # Benchmark data files
├── demonstrator.ipynb                 # Interactive notebook
└── content/                           # Documentation (this site)
```

## Dependencies

The following packages are installed via `setup.sh`:

- `numpy`, `pandas` - Data manipulation
- `scikit-learn` - Utilities
- `scipy` - Scientific computing (CMA-ES, Nelder-Mead)
- `matplotlib`, `seaborn` - Visualization
- `deap` - Evolutionary algorithms
- `jupyterlab`, `ipywidgets` - Interactive notebooks

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `git: command not found` | Run VM init: `sudo apt install -y git` |
| Connection refused | Verify VM is running with `ping <VM_IP>` |
| Permission denied | `chmod 600 /path/to/your-key.pem` |
| SSH "Permissions too open" (Windows) | Use Git Bash (`chmod 600`) or fix via icacls — see Episode 02 |
| SSH connection timed out | Your IP may not be whitelisted — add it at orchestrator.naic.no |
| Host key error | `ssh-keygen -R <VM_IP>` (VM IP changed) |
| `ModuleNotFoundError: cec2013` | Use `source activate-mmo.sh` instead of `source venv/bin/activate` |
| `ModuleNotFoundError: mmo` | Use `source activate-mmo.sh` instead of `source venv/bin/activate` |
| Jupyter not accessible | Check tunnel is running; verify correct port |

```{keypoints}
- Set SSH key permissions with `chmod 600` before connecting (use Git Bash on Windows)
- Initialize fresh VMs with `sudo apt install -y build-essential git python3-dev python3-venv`
- Clone this repository directly -- all code and data are included
- Run `./setup.sh` to automatically set up the Python environment
- Always use `source activate-mmo.sh` to activate (sets PYTHONPATH automatically)
- Use tmux for persistent Jupyter Lab sessions
- Windows users: Git Bash is recommended for the best experience with SSH and Unix commands
```
