#!/usr/bin/env python
"""Generate benchmark_summary.png with reduced budget for speed."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cec2013.cec2013 import CEC2013, how_many_goptima
from mmo.minimize import MultiModalMinimizer
from mmo.domain import Domain

benchmark_results = []
for func_id in range(4, 15):
    f = CEC2013(func_id)
    info = f.get_info()
    dim = f.get_dimension()
    lb = np.array([f.get_lbound(k) for k in range(dim)])
    ub = np.array([f.get_ubound(k) for k in range(dim)])
    domain = Domain(boundary=[lb.tolist(), ub.tolist()])

    # Use reduced budget/iterations for faster execution
    budget = min(info['maxfes'], 50000)
    max_iter = 15

    final = None
    for result in MultiModalMinimizer(f=f, domain=domain, budget=budget, max_iter=max_iter, verbose=0):
        final = result

    if final is not None:
        X_clipped = np.clip(final.x, lb, ub)
        count, _ = how_many_goptima(X_clipped, f, 0.0001)
        pr = count / info['nogoptima']
    else:
        count = 0
        pr = 0.0

    benchmark_results.append({
        'id': func_id, 'name': info['name'], 'dim': dim,
        'total': info['nogoptima'], 'found': count, 'PR': pr
    })
    print(f'F{func_id}: {count}/{info["nogoptima"]} ({pr:.0%})', flush=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ids = [f'F{r["id"]}' for r in benchmark_results]
prs = [r['PR'] for r in benchmark_results]
colors = ['#2ecc71' if pr >= 1.0 else '#f39c12' if pr >= 0.5 else '#e74c3c' for pr in prs]
ax.bar(ids, prs, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect (100%)')
ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Good (80%)')
ax.set_xlabel('CEC2013 Function')
ax.set_ylabel('Peak Ratio')
ax.set_title('SHGA Performance: Peak Ratio per Function')
ax.set_ylim(0, 1.15)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(axis='x', rotation=45)

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
fig.suptitle(f'CEC2013 Benchmark Results — Average Peak Ratio: {avg_pr:.0%}', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig('content/images/benchmark_summary.png', dpi=150, bbox_inches='tight')
print(f'Saved benchmark_summary.png (avg PR: {avg_pr:.0%})', flush=True)
