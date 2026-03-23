# Introduction to Multi-Modal Optimization

```{objectives}
- Understand what multi-modal optimization is and why it matters
- Learn about the SHGA algorithm and its advantages
- Know the project objectives and use cases
```

```{admonition} Why This Matters
:class: tip

**The Scenario:** A turbine design team has found one blade geometry that meets all stress and efficiency constraints — but is it the *only* good design? A manufacturing change could make a different geometry cheaper to produce, or a slightly different shape could perform better at off-design conditions. They need an algorithm that finds *all* viable designs in a single run, not just the single best.

**The Research Question:** Can a hybrid optimization algorithm — combining genetic algorithms for global exploration with CMA-ES for local refinement — reliably find all global optima of a multi-modal function, from simple 2D problems to challenging 20-dimensional landscapes?

**What This Episode Gives You:** The big picture — what multi-modal optimization is, how SHGA works at a high level, and why finding multiple solutions matters more than finding just one.
```

## Overview

This repository provides a complete framework for multi-modal optimization using the Scalable Hybrid Genetic Algorithm (SHGA). Multi-modal optimization is the task of finding **multiple** local or global optima of a function, rather than just a single best solution.

Many real-world optimization problems have multiple good solutions:
- **Engineering design:** Multiple valid configurations that meet constraints
- **Neural network training:** Multiple weight configurations with similar loss
- **Scheduling problems:** Multiple feasible schedules with equal quality
- **Scientific discovery:** Multiple parameter sets that explain observed data

The SHGA algorithm efficiently finds all (or many) of these solutions in a single run.

```{figure} ../images/himmelblau_surface.png
:alt: Himmelblau function surface with 4 global optima
:width: 80%

The Himmelblau function — a classic multi-modal test function with 4 global optima (marked as red stars). Multi-modal optimization aims to find all of them in a single run.
```

## The SHGA Algorithm

SHGA (Scalable Hybrid Genetic Algorithm) combines two powerful optimization techniques:

| Component | Purpose |
|-----------|---------|
| **Deterministic Crowding GA** | Global search - explores the entire domain to find promising regions |
| **CMA-ES** | Local refinement - accurately locates optima within promising regions |

This hybrid approach provides:
- **Completeness:** Finds many/all optima, not just one
- **Accuracy:** CMA-ES provides high-precision local solutions
- **Scalability:** Works efficiently up to moderate dimensions (10-20D)
 

### Research Paper

This implementation is based on:

> Johannsen et al. (2022). *A scalable, hybrid genetic algorithm for continuous multimodal optimization in moderate dimensions.* Nordic Machine Intelligence.

## Self-Contained Repository

This repository is self-contained. Everything you need is included:

| Component | Location |
|-----------|----------|
| SHGA algorithm | `mmo/` (MultiModalMinimizer class) |
| CEC2013 benchmarks | `benchmarks/CEC2013/python3/` |
| Benchmark data | `data/` |
| Interactive notebook | `demonstrator.ipynb` |
| Dependencies | `requirements.txt` |

Simply clone the repository and follow the setup instructions to get started.

## Using AI Coding Assistants

If you're using an AI coding assistant like **Claude Code**, **GitHub Copilot**, or **Cursor**, the repository includes an `AGENT.md` file with machine-readable instructions. Simply tell your assistant:

> "Read AGENT.md and help me run the multi-modal optimization demonstrator on my NAIC VM."

The agent will be able to set up the environment and run experiments automatically.

## What You Will Learn

| Episode | Topic |
|---------|-------|
| 02 | Provisioning a NAIC VM |
| 03 | Setting up the environment |
| 04 | Understanding the SHGA algorithm |
| 05 | Running optimization experiments |
| 06 | CEC2013 benchmark functions |
| 07 | Analyzing results |
| 08 | FAQ and troubleshooting |
| 09 | Parallelization on multi-core VMs |
| 10 | Visualization guide |

## Resources

- NAIC Portal: https://orchestrator.naic.no
- VM Workflows Guide: https://training.pages.sigma2.no/tutorials/naic-cloud-vm-workflows/
- This Repository: https://github.com/NAICNO/wp7-UC6-multimodal-optimization

```{keypoints}
- Multi-modal optimization finds multiple local/global optima, not just one
- SHGA combines genetic algorithm (global search) with CMA-ES (local refinement)
- The algorithm scales to moderate dimensions (10-20D)
- All code and benchmark data are included in this repository
```
