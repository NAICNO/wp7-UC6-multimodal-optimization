# Visualization Guide

```{objectives}
- Understand why CEC2013 benchmark plots need special handling
- Know how percentile-based z-axis scaling works
- Troubleshoot common visualization issues
```

## Understanding CEC2013 Benchmark Function Visualizations

### The Issue

CEC2013 benchmark functions are **transformed versions** of standard test functions. These transformations (scaling, shifting, rotation) create large value ranges that can make visualizations misleading.

**Example value ranges on the valid domain:**

| Function | Dim | Domain | Z Range | Notes |
|----------|-----|--------|---------|-------|
| **2D Functions** |
| F4 (Himmelblau) | 2 | [-6, -6] to [6, 6] | -1986 to 200 | Huge range! |
| F5 (Six-Hump Camel) | 2 | [-1.9, -1.1] to [1.9, 1.1] | -5.86 to 1.03 | Reasonable |
| F6 (Shubert) | 2 | [-10, -10] to [10, 10] | -207 to 185 | Large range |
| F7 (Vincent) | 2 | [0.2, 0.2] to [10, 10] | -1.0 to 0.99 | Reasonable |
| F10 (Modified Rastrigin) | 2 | [0, 0] to [1, 1] | -38 to -2 | Moderate |
| **3D+ Functions** |
| F8 (Shubert 3D) | 3 | [-10]³ to [10]³ | -263 to 687 | Extreme range! |
| F9 (Vincent 3D) | 3 | [0.25]³ to [10]³ | -0.79 to 0.95 | Reasonable |
| F11-F13 (Composition) | 2 | [-5, -5] to [5, 5] | -3039 to -32 | Extreme! |
| F14-F15 (Composition 3D) | 3 | [-5]³ to [5]³ | -3861 to -251 | Extreme! |
| F16-F17 (Composition 5D) | 5 | [-5]⁵ to [5]⁵ | -2246 to -583 | Very large |
| F18-F19 (Composition 10D) | 10 | [-5]¹⁰ to [5]¹⁰ | -2682 to -702 | Large |
| F20 (Composition 20D) | 20 | [-5]²⁰ to [5]²⁰ | -2047 to -959 | Large |

### Why Plots Can Look Wrong

**Before Fix (commits f41ebec + 7a745a2):**
- matplotlib auto-scaled z-axis to show full surface/scatter range
- **2D functions** (F4-F7, F10): z-axis showed full surface range
  - F4 (Himmelblau): z-axis -1986 to 200
  - F6 (Shubert): z-axis -207 to 185
- **3D+ functions** (F8-F20): z-axis showed full scatter range
  - F8 (Shubert 3D): z-axis -263 to 687
  - F14 (Composition 3D): z-axis -3065 to -487
- **Problem**: Extreme values dominated visualization, optima invisible

**After Fix:**
- Z-axis limited to 5th-95th percentile of **solution points** (not surface/extrema)
- Focus on region where optima actually exist
- Visualization shows context, but scale emphasizes solutions
- **Applies to both 2D and 3D+ functions**

### How the Fix Works

The plotting code now uses percentile-based z-axis limits for **both 2D and 3D+ functions**:

**For 2D functions** (`plot_3d` - commit f41ebec):
1. Creates surface mesh by evaluating f(x,y) on grid
2. Collects z-values from scatter data (solutions, seeds, population, true optima)
3. Sets z-axis limits to 5th-95th percentile of scatter data
4. Surface shown with transparency, but z-axis focuses on solutions

**For 3D+ functions** (`plot_composite` - commit 7a745a2):
1. Projects high-dimensional data to 2D using PCA
2. Evaluates f(x) at all scatter points in original space
3. Uses f(x) values as z-coordinate in 3D scatter plot
4. Sets z-axis limits to 5th-95th percentile of scatter z-values

**Code pattern (both functions):**
```python
# Collect z-values from all scatter data
data_z = []
# ... evaluate f(x) for population, seeds, solutions, true optima ...
data_z.extend(zvals)

# Compute z-axis limits from percentiles
lower_z, upper_z = np.percentile(data_z, [5, 95])
z_center = (lower_z + upper_z) / 2
z_range = (upper_z - lower_z) * 1.2  # 20% margin
ax.set_zlim(z_center - z_range/2, z_center + z_range/2)
```

**Result:**
- Extreme surface/scatter values clipped from view
- Optima and solutions clearly visible
- Axis labels reflect actual solution value range

### Example: Six-Hump Camel Back (F5)

**Standard function:**
- Global minimum value: ≈ -1.0316
- Two global optima at (±0.0898, ∓0.7126)
- Domain: [-2, -1] to [2, 1]

**CEC2013 transformed version:**
- Global **maximum** value: 1.0316 (note: maximization!)
- Same optima locations
- Domain: [-1.9, -1.1] to [1.9, 1.1]
- Surface range: -5.86 to 1.03

**Visualization:**
- Z-axis focused on -2 to +2 (solution range)
- Optima clearly visible at z ≈ 1.03
- Surface provides context without dominating

### Interpreting the Plots

When viewing 3D surface plots:

1. **Surface mesh** (transparent): Shows function landscape
2. **Blue triangles** (^): Known global optima locations
3. **Red stars** (*): Solutions found by algorithm
4. **Orange X**: True solutions (if provided separately)
5. **Gray dots** (o): Population samples

**Key observations:**
- Red stars should cluster near blue triangles (if algorithm successful)
- Z-axis range focuses on solution values, not full surface range
- Colorbar shows actual function values on surface

### Visualization by Dimensionality

**1D functions (F1-F3):**
- Skipped due to implementation constraints
- Use `RUN_FUNCTIONS = list(range(4, 15))` to avoid

**2D functions (F4-F7, F10-F13):**
- 3D surface plot with scatter overlay
- Surface mesh shows function landscape
- Z-axis focused on solution region using percentiles
- Fixed by commit f41ebec

**3D+ functions (F8-F9, F14-F20):**
- PCA projection to 2D for visualization
- Left subplot: 2D contour (PCA space)
- Right subplot: 3D scatter (PCA x-y, f(x) as z)
- Z-axis focused on solution region using percentiles
- Fixed by commit 7a745a2

Both visualization types now use the same percentile-based z-axis limiting strategy.

### Running with Plotting Enabled

**In Jupyter Notebook/Lab:**
```python
# Enable plotting in configuration cell
PLOT_LAST = True  # Show plots for 2D functions
```

**In batch mode (`jupyter nbconvert`):**
```python
PLOT_LAST = False  # Disable plotting for faster execution
```

### Advanced: Evaluating Outside Domain Bounds

⚠️ **WARNING**: CEC2013 functions return **huge negative values** when evaluated outside their valid domain.

Example for F5 (Six-Hump Camel):
```
Inside domain  [0, 0]: -0.00
Inside domain  [0.0898, -0.7126]: 1.03  (global optimum)
Outside domain [-5, -5]: -6420.83  (extreme!)
Outside domain [-3, 0]: -108.90   (large!)
```

**Implication**: If plotting code evaluates outside bounds, visualizations will be completely wrong. The fix ensures z-axis limits are based on valid solution data, not surface extrema.

### Troubleshooting

**Issue**: "Z-axis shows huge range like -3000 to 100 or -1000 to 687"
- **Cause**: Using older code before commits f41ebec (2D) or 7a745a2 (3D+)
- **Affected**: All CEC2013 functions with large value ranges
- **Fix**: `git pull` to get latest version with percentile-based z-limits

**Issue**: "3D scatter plots (F8-F20) have wrong z-axis scale"
- **Cause**: Using code before commit 7a745a2
- **Example**: F8 (Shubert 3D) showed z-axis -263 to 687 instead of focusing on optima
- **Fix**: Update to latest version (commit 7a745a2 or newer)

**Issue**: "Composition functions (F11-F20) show extreme negative values"
- **Cause**: CEC2013 composition functions have ranges like -3861 to -86
- **Expected**: Z-axis now automatically focuses on 5th-95th percentile
- **Note**: This is correct behavior after the fix

**Issue**: "No plots appear when running notebook"
- **Cause**: `PLOT_LAST = False` or running via `jupyter nbconvert`
- **Fix**: Set `PLOT_LAST = True` and run in Jupyter Lab/Notebook

**Issue**: "Plots look different from documentation examples"
- **Cause**: Stochastic algorithm finds different solutions each run
- **Expected**: Optima locations should be consistent, but exact paths vary

**Issue**: "F1-F3 (1D functions) crash when plotting"
- **Cause**: Array shape incompatibility with SHGA
- **Fix**: Skip 1D functions, use `RUN_FUNCTIONS = list(range(4, 15))`

**Issue**: "ValueError: math domain error" when plotting Vincent functions (F7, F9)
- **Cause**: Multiple sources of out-of-domain evaluations:
  - PCA inverse transform produces grid points outside [0.25, 10]
  - CMA-ES local search produces solutions near domain boundaries
  - Plotting evaluates solutions/seeds/population without clipping
- **Symptom**: Crash with `math.log(x[i])` error in Vincent function
- **Fix**: Update to commit e8c66f8 or newer (comprehensive domain clipping)
  - Grid points clipped (81307e7)
  - Solutions clipped before how_many_goptima (f4579d2)
  - All plot evaluations clipped (e8c66f8)

### References

- CEC2013 benchmark suite: [Liang et al., 2013]
- SHGA algorithm: [Johannsen et al., 2022]
- Visualization fixes:
  - commit f41ebec (2026-01-26): 2D functions (plot_3d)
  - commit 7a745a2 (2026-01-26): 3D+ functions (plot_composite)

```{keypoints}
- CEC2013 functions have extreme value ranges that require special z-axis handling
- Percentile-based z-axis limits (5th-95th of solution values) focus plots on the relevant region
- Always clip coordinates to domain bounds before evaluating CEC2013 functions
- 2D functions use surface plots; 3D+ functions use PCA projection to 2D
- Set `PLOT_LAST = True` in the notebook for interactive visualization
```
