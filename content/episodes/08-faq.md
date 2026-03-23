# FAQ and Troubleshooting

```{objectives}
- Resolve common issues
- Understand algorithm behavior
- Get help when needed
```

## Installation Issues

### ModuleNotFoundError: cec2013 or mmo

**Problem:** Python can't find the CEC2013 or mmo module.

**Solution:** Use the activation wrapper script instead of directly activating the venv:

```bash
# Wrong (doesn't set PYTHONPATH)
source venv/bin/activate

# Correct (sets PYTHONPATH automatically)
source activate-mmo.sh
```

The `activate-mmo.sh` script is created by `setup.sh` and automatically configures PYTHONPATH to include both the project root and the CEC2013 benchmarks directory.

### ImportError: numpy.distutils

**Problem:** Old numpy compatibility issue.

**Solution:** Upgrade numpy:
```bash
pip install --upgrade numpy
```

## Algorithm Issues

### Algorithm finds few/no solutions

**Possible causes:**
1. Budget too low
2. Function has very narrow basins
3. Population size too small

**Solutions:**
1. Increase budget (try 2-5x)
2. Decrease CMA-ES initial sigma
3. Increase GA population size

### Solutions are duplicates

**Problem:** Multiple solutions at the same location.

**Cause:** Clustering threshold too small or CMA-ES converging to same point from multiple seeds.

**Solution:** Increase solution uniqueness threshold in post-processing:

```python
from scipy.spatial.distance import pdist, squareform

def remove_duplicates(solutions, threshold=1e-4):
    """Remove duplicate solutions."""
    if len(solutions) <= 1:
        return solutions

    distances = squareform(pdist(solutions[:, :-1]))
    keep = [0]
    for i in range(1, len(solutions)):
        if np.min(distances[i, keep]) > threshold:
            keep.append(i)
    return solutions[keep]
```

### Algorithm is slow

**Possible causes:**
1. Function evaluation is expensive (not CEC2013 benchmarks)
2. Too many CMA-ES iterations
3. Large population size

**Solutions:**
1. For expensive custom functions: reduce evaluation cost or budget
2. Reduce CMA-ES max iterations
3. Use a surrogate model for expensive functions

## VM Issues

### Connection refused

**Cause:** VM not running or firewall blocking.

**Solutions:**
1. Check VM status in NAIC portal
2. Verify your IP is whitelisted
3. Try: `ping <VM_IP>`

### Permission denied (publickey)

**Cause:** SSH key issue.

**Solutions:**
```bash
# Fix key permissions
chmod 600 /path/to/your-key.pem

# Verify key path is correct
ls -la /path/to/your-key.pem
```

### Host key verification failed

**Cause:** VM IP changed (common with NAIC Orchestrator).

**Solution:**
```bash
ssh-keygen -R <old_VM_IP>
```

### Jupyter not accessible

**Causes:**
1. SSH tunnel not running
2. Jupyter not started
3. Wrong port

**Solutions:**
1. Start tunnel: `ssh -N -L 8888:localhost:8888 -i key.pem ubuntu@<VM_IP>`
2. Start Jupyter: `jupyter lab --no-browser --ip=127.0.0.1 --port=8888`
3. Use alternative port if 8888 is busy

## Performance Questions

### What budget should I use?

Guidelines based on CEC2013:

| Dimension | Optima | Recommended Budget |
|-----------|--------|-------------------|
| 1-2D | < 10 | 50,000 |
| 2D | 10-50 | 200,000 |
| 2-10D | > 50 | 400,000 |
| 10-20D | any | 400,000+ |

## Getting Help

### Resources

- **NAIC Support:** support@naic.no
- **VM Workflows Guide:** https://training.pages.sigma2.no/tutorials/naic-cloud-vm-workflows/
- **NAIC Portal:** https://orchestrator.naic.no

### Reporting Issues

When reporting issues, include:
1. Error message (full traceback)
2. Python version: `python --version`
3. Package versions: `pip list`
4. Steps to reproduce

### Algorithm Reference

For algorithm details, see the research paper:

> Johannsen et al. (2022). *A scalable, hybrid genetic algorithm for continuous multimodal optimization in moderate dimensions.* Nordic Machine Intelligence.

```{keypoints}
- Most import errors are solved by using `source activate-mmo.sh` (not `source venv/bin/activate`)
- Increase budget for difficult functions with many optima
- VM connection issues often stem from IP changes or key permissions
- Contact support@naic.no for NAIC-specific issues
```
