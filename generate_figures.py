#!/usr/bin/env python3
"""Generate tutorial figures for the SHGA multi-modal optimization tutorial.

Produces PNG files in content/images/ for use in Sphinx episodes.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Setup paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'benchmarks', 'CEC2013', 'python3'))

from cec2013.cec2013 import CEC2013, how_many_goptima
from mmo.domain import Domain
from mmo.minimize import MultiModalMinimizer

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'content', 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DPI = 150


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}")


# =============================================================================
# Figure 1: Himmelblau's function surface with 4 global optima marked
# =============================================================================
print("Generating: himmelblau_surface.png")

x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2

# Known global minima
minima = np.array([
    [3.0, 2.0],
    [-2.805118, 3.131312],
    [-3.779310, -3.283186],
    [3.584428, -1.848126]
])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: contour
ax = axes[0]
cs = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
fig.colorbar(cs, ax=ax, label='f(x,y)')
ax.scatter(minima[:, 0], minima[:, 1], c='red', s=120, marker='*',
           edgecolors='white', linewidths=1.5, zorder=5, label='Global minima')
for i, (mx, my) in enumerate(minima):
    ax.annotate(f'({mx:.1f}, {my:.1f})', (mx, my),
                textcoords='offset points', xytext=(8, 8),
                fontsize=8, color='white', fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("Himmelblau's Function — Contour Map")
ax.legend(loc='upper left')

# Right: 3D surface
ax3 = fig.add_subplot(122, projection='3d')
ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, linewidth=0)
Z_min = np.zeros(len(minima))
ax3.scatter(minima[:, 0], minima[:, 1], Z_min, c='red', s=100, marker='*',
            edgecolors='white', linewidths=1, zorder=5)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('f(x,y)')
ax3.set_title('3D Surface')
ax3.view_init(elev=30, azim=-60)

fig.suptitle("Himmelblau's Function: 4 Global Minima", fontsize=14, fontweight='bold')
plt.tight_layout()
save(fig, 'himmelblau_surface.png')


# =============================================================================
# Figure 2: Run SHGA on Himmelblau (F4) and show results
# =============================================================================
print("Generating: shga_himmelblau_result.png")

f4 = CEC2013(4)
info4 = f4.get_info()
dim4 = f4.get_dimension()
lb4 = np.array([f4.get_lbound(k) for k in range(dim4)])
ub4 = np.array([f4.get_ubound(k) for k in range(dim4)])
domain4 = Domain(boundary=[lb4.tolist(), ub4.tolist()])

# Run SHGA
sols_per_iter = []
optima_per_iter = []
fev_per_iter = []
final_result = None

for result in MultiModalMinimizer(f=f4, domain=domain4, budget=info4['maxfes'], max_iter=50, verbose=0):
    final_result = result
    sols_per_iter.append(result.n_sol)
    fev_per_iter.append(result.n_fev)
    X_clipped = np.clip(result.x, lb4, ub4)
    count, seeds = how_many_goptima(X_clipped, f4, 0.0001)
    optima_per_iter.append(count)

# Create a 2D contour with found solutions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: contour with solutions
ax = axes[0]
x_grid = np.linspace(lb4[0], ub4[0], 100)
y_grid = np.linspace(lb4[1], ub4[1], 100)
Xg, Yg = np.meshgrid(x_grid, y_grid)
Zg = np.array([f4.evaluate([xi, yi]) for xi, yi in
               zip(Xg.ravel(), Yg.ravel())]).reshape(Xg.shape)

cs = ax.contourf(Xg, Yg, Zg, levels=30, cmap='RdYlGn')
fig.colorbar(cs, ax=ax, label='f(x,y)')

# Plot found solutions
if final_result is not None and final_result.x is not None:
    sol_x = final_result.x[:, 0]
    sol_y = final_result.x[:, 1]
    ax.scatter(sol_x, sol_y, c='blue', s=80, marker='*',
               edgecolors='white', linewidths=1, zorder=5, label=f'Found ({final_result.n_sol} solutions)')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'SHGA Results on Himmelblau (F4)')
ax.legend()

# Right: convergence
ax2 = axes[1]
iters = range(1, len(optima_per_iter) + 1)
ax2.plot(iters, optima_per_iter, 'g-o', markersize=4, label='Global optima found')
ax2.axhline(y=info4['nogoptima'], color='red', linestyle='--', alpha=0.7,
            label=f'Total: {info4["nogoptima"]}')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Count')
ax2.set_title('Convergence: Optima Found per Iteration')
ax2.legend()
ax2.set_ylim(0, info4['nogoptima'] + 1)
ax2.grid(True, alpha=0.3)

fig.suptitle(f'SHGA on Himmelblau — {optima_per_iter[-1]}/{info4["nogoptima"]} optima found',
             fontsize=14, fontweight='bold')
plt.tight_layout()
save(fig, 'shga_himmelblau_result.png')


# =============================================================================
# Figure 3: Run SHGA on Six-Hump Camel (F5) and show results
# =============================================================================
print("Generating: shga_camel_result.png")

f5 = CEC2013(5)
info5 = f5.get_info()
dim5 = f5.get_dimension()
lb5 = np.array([f5.get_lbound(k) for k in range(dim5)])
ub5 = np.array([f5.get_ubound(k) for k in range(dim5)])
domain5 = Domain(boundary=[lb5.tolist(), ub5.tolist()])

final_result5 = None
optima_per_iter5 = []
for result in MultiModalMinimizer(f=f5, domain=domain5, budget=info5['maxfes'], max_iter=50, verbose=0):
    final_result5 = result
    X_clipped = np.clip(result.x, lb5, ub5)
    count, seeds = how_many_goptima(X_clipped, f5, 0.0001)
    optima_per_iter5.append(count)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: contour with solutions
ax = axes[0]
x_grid = np.linspace(lb5[0], ub5[0], 100)
y_grid = np.linspace(lb5[1], ub5[1], 100)
Xg, Yg = np.meshgrid(x_grid, y_grid)
Zg = np.array([f5.evaluate([xi, yi]) for xi, yi in
               zip(Xg.ravel(), Yg.ravel())]).reshape(Xg.shape)

cs = ax.contourf(Xg, Yg, Zg, levels=30, cmap='RdYlGn')
fig.colorbar(cs, ax=ax, label='f(x,y)')

if final_result5 is not None and final_result5.x is not None:
    sol_x = final_result5.x[:, 0]
    sol_y = final_result5.x[:, 1]
    ax.scatter(sol_x, sol_y, c='blue', s=80, marker='*',
               edgecolors='white', linewidths=1, zorder=5, label=f'Found ({final_result5.n_sol} solutions)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'SHGA Results on Six-Hump Camel (F5)')
ax.legend()

# Right: convergence
ax2 = axes[1]
iters = range(1, len(optima_per_iter5) + 1)
ax2.plot(iters, optima_per_iter5, 'g-o', markersize=4, label='Global optima found')
ax2.axhline(y=info5['nogoptima'], color='red', linestyle='--', alpha=0.7,
            label=f'Total: {info5["nogoptima"]}')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Count')
ax2.set_title('Convergence: Optima Found per Iteration')
ax2.legend()
ax2.set_ylim(0, max(info5['nogoptima'] + 1, 4))
ax2.grid(True, alpha=0.3)

fig.suptitle(f'SHGA on Six-Hump Camel — {optima_per_iter5[-1]}/{info5["nogoptima"]} optima found',
             fontsize=14, fontweight='bold')
plt.tight_layout()
save(fig, 'shga_camel_result.png')


# =============================================================================
# Figure 4: Run on Shubert (F6) — 18 global optima
# =============================================================================
print("Generating: shga_shubert_result.png")

f6 = CEC2013(6)
info6 = f6.get_info()
dim6 = f6.get_dimension()
lb6 = np.array([f6.get_lbound(k) for k in range(dim6)])
ub6 = np.array([f6.get_ubound(k) for k in range(dim6)])
domain6 = Domain(boundary=[lb6.tolist(), ub6.tolist()])

final_result6 = None
optima_per_iter6 = []
for result in MultiModalMinimizer(f=f6, domain=domain6, budget=info6['maxfes'], max_iter=50, verbose=0):
    final_result6 = result
    X_clipped = np.clip(result.x, lb6, ub6)
    count, seeds = how_many_goptima(X_clipped, f6, 0.0001)
    optima_per_iter6.append(count)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: contour with solutions
ax = axes[0]
x_grid = np.linspace(lb6[0], ub6[0], 100)
y_grid = np.linspace(lb6[1], ub6[1], 100)
Xg, Yg = np.meshgrid(x_grid, y_grid)
Zg = np.array([f6.evaluate([xi, yi]) for xi, yi in
               zip(Xg.ravel(), Yg.ravel())]).reshape(Xg.shape)

cs = ax.contourf(Xg, Yg, Zg, levels=30, cmap='RdYlGn')
fig.colorbar(cs, ax=ax, label='f(x,y)')

if final_result6 is not None and final_result6.x is not None:
    sol_x = final_result6.x[:, 0]
    sol_y = final_result6.x[:, 1]
    ax.scatter(sol_x, sol_y, c='blue', s=80, marker='*',
               edgecolors='white', linewidths=1, zorder=5, label=f'Found ({final_result6.n_sol} solutions)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'SHGA Results on Shubert (F6) — 18 global optima')
ax.legend()

# Right: convergence
ax2 = axes[1]
iters = range(1, len(optima_per_iter6) + 1)
ax2.plot(iters, optima_per_iter6, 'g-o', markersize=4, label='Global optima found')
ax2.axhline(y=info6['nogoptima'], color='red', linestyle='--', alpha=0.7,
            label=f'Total: {info6["nogoptima"]}')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Count')
ax2.set_title('Convergence')
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle(f'SHGA on Shubert — {optima_per_iter6[-1]}/{info6["nogoptima"]} optima found',
             fontsize=14, fontweight='bold')
plt.tight_layout()
save(fig, 'shga_shubert_result.png')


# =============================================================================
# Figure 5: CEC2013 benchmark catalog overview
# =============================================================================
print("Generating: cec2013_catalog.png")

functions_info = []
for i in range(1, 21):
    f = CEC2013(i)
    info = f.get_info()
    functions_info.append({
        'ID': f'F{i}',
        'Name': info['name'],
        'Dim': f.get_dimension(),
        'Global Optima': info['nogoptima'],
        'Budget': info['maxfes']
    })

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: optima count per function
ids = [fi['ID'] for fi in functions_info]
optima = [fi['Global Optima'] for fi in functions_info]
dims = [fi['Dim'] for fi in functions_info]

colors = ['#2ecc71' if d <= 2 else '#3498db' if d <= 5 else '#e74c3c' for d in dims]
ax = axes[0]
bars = ax.bar(ids, optima, color=colors, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Function')
ax.set_ylabel('Number of Global Optima')
ax.set_title('CEC2013 Benchmark: Global Optima per Function')
ax.tick_params(axis='x', rotation=45)
# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='1-2D'),
    Patch(facecolor='#3498db', label='3-5D'),
    Patch(facecolor='#e74c3c', label='10-20D')
]
ax.legend(handles=legend_elements, title='Dimensions')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

# Right: dimension vs budget
ax2 = axes[1]
unique_dims = sorted(set(dims))
for d in unique_dims:
    funcs = [fi for fi in functions_info if fi['Dim'] == d]
    budgets = [fi['Budget'] for fi in funcs]
    ax2.scatter([d] * len(budgets), budgets, s=100, zorder=5)
    for fi in funcs:
        ax2.annotate(fi['ID'], (d, fi['Budget']),
                     textcoords='offset points', xytext=(5, 5), fontsize=7)
ax2.set_xlabel('Dimension')
ax2.set_ylabel('Function Evaluation Budget')
ax2.set_title('Budget vs Dimension')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

fig.suptitle('CEC2013 Niching Benchmark Suite (20 Functions)', fontsize=14, fontweight='bold')
plt.tight_layout()
save(fig, 'cec2013_catalog.png')


# =============================================================================
# Figure 6: SHGA Algorithm Flowchart (text-based visual)
# =============================================================================
print("Generating: shga_algorithm.png")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Draw boxes and arrows
boxes = [
    (5, 11, 'Initialize Population\n(random sampling)', '#3498db'),
    (5, 9, 'Genetic Algorithm\nw/ Deterministic Crowding\n(global exploration)', '#2ecc71'),
    (5, 7, 'Seed Detection\n(nearest-neighbor clustering)', '#f39c12'),
    (5, 5, 'CMA-ES Local Refinement\n(per-seed optimization)', '#e74c3c'),
    (5, 3, 'Merge Solutions\n(remove duplicates)', '#9b59b6'),
    (5, 1, 'Scale Population\n(increase size, repeat)', '#1abc9c'),
]

for x, y, text, color in boxes:
    bbox = dict(boxstyle='round,pad=0.6', facecolor=color, alpha=0.85, edgecolor='white', linewidth=2)
    ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold',
            color='white', bbox=bbox)

# Arrows
for i in range(len(boxes) - 1):
    ax.annotate('', xy=(5, boxes[i+1][1] + 0.6), xytext=(5, boxes[i][1] - 0.6),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))

# Loop arrow from bottom back to GA
ax.annotate('', xy=(2, 9), xytext=(2, 1),
            arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2,
                           connectionstyle='arc3,rad=0.3'))
ax.text(1.0, 5, 'Outer\nLoop', ha='center', va='center', fontsize=10,
        fontweight='bold', color='#2c3e50', rotation=90)

fig.suptitle('SHGA Algorithm: Seed-Solve-Collect Loop', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
save(fig, 'shga_algorithm.png')


# =============================================================================
# Figure 7: Benchmark summary (run F4-F14)
# =============================================================================
print("Generating: benchmark_summary.png (running F4-F14...)")

benchmark_results = []
for func_id in range(4, 15):
    f = CEC2013(func_id)
    info = f.get_info()
    dim = f.get_dimension()
    lb = np.array([f.get_lbound(k) for k in range(dim)])
    ub = np.array([f.get_ubound(k) for k in range(dim)])
    domain = Domain(boundary=[lb.tolist(), ub.tolist()])

    final = None
    for result in MultiModalMinimizer(f=f, domain=domain, budget=info['maxfes'], max_iter=50, verbose=0):
        final = result

    if final is not None:
        X_clipped = np.clip(final.x, lb, ub)
        count, _ = how_many_goptima(X_clipped, f, 0.0001)
        pr = count / info['nogoptima']
    else:
        count = 0
        pr = 0.0

    benchmark_results.append({
        'id': func_id,
        'name': info['name'],
        'dim': dim,
        'total': info['nogoptima'],
        'found': count,
        'PR': pr
    })
    print(f"  F{func_id}: {count}/{info['nogoptima']} ({pr:.0%})")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Peak Ratio bar chart
ax = axes[0]
ids = [f"F{r['id']}" for r in benchmark_results]
prs = [r['PR'] for r in benchmark_results]
colors = ['#2ecc71' if pr >= 1.0 else '#f39c12' if pr >= 0.5 else '#e74c3c' for pr in prs]
bars = ax.bar(ids, prs, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect (100%)')
ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Good (80%)')
ax.set_xlabel('CEC2013 Function')
ax.set_ylabel('Peak Ratio')
ax.set_title('SHGA Performance: Peak Ratio per Function')
ax.set_ylim(0, 1.15)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(axis='x', rotation=45)

# Right: Found vs Total optima
ax2 = axes[1]
totals = [r['total'] for r in benchmark_results]
founds = [r['found'] for r in benchmark_results]
x_pos = range(len(ids))
width = 0.35
ax2.bar([p - width/2 for p in x_pos], totals, width, label='Total Optima', color='#bdc3c7', edgecolor='white')
ax2.bar([p + width/2 for p in x_pos], founds, width, label='Found by SHGA', color='#2ecc71', edgecolor='white')
ax2.set_xlabel('CEC2013 Function')
ax2.set_ylabel('Count')
ax2.set_title('Global Optima: Total vs Found')
ax2.set_xticks(list(x_pos))
ax2.set_xticklabels(ids, rotation=45)
ax2.legend()
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')

avg_pr = np.mean(prs)
fig.suptitle(f'CEC2013 Benchmark Results — Average Peak Ratio: {avg_pr:.0%}',
             fontsize=14, fontweight='bold')
plt.tight_layout()
save(fig, 'benchmark_summary.png')


print("\n=== All figures generated! ===")
print(f"Output directory: {OUTPUT_DIR}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png') and f not in ['NRIS-Logo.png', 'orchestrator1.png',
                                         'orchestrator2.png', 'orchestrator3.png', 'orchestrator4.png']:
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f} ({size/1024:.0f} KB)")
