# Expanded FAQ for Multimodal Optimization Use Case

## Overview

This FAQ provides comprehensive guidance for the **Multimodal Optimization (MMO)** use case, focusing on the **Scalable Hybrid Genetic Algorithm (SHGA)** demonstrator. The demonstrator is designed to run on the **Sigma2 High-Performance Computing (HPC)** infrastructure, accessed via the **Norwegian Artificial Intelligence Cloud (NAIC)** platform. NAIC is not a server itself but a platform that facilitates access to national e-infrastructure resources, including Sigma2’s HPC systems and the **Sigma2 Service Platform (SP)**.

The SHGA demonstrator, developed by Johannsen et al. (2022), integrates a genetic algorithm with deterministic crowding for global exploration and CMA-ES for local refinement. It is tailored for continuous multimodal optimization and is best executed on Sigma2’s HPC clusters due to its computational demands.

While the Sigma2 Service Platform offers a user-friendly environment for deploying AI workflows and services, it currently has resource limitations that make it unsuitable for running the SHGA demonstrator efficiently. However, the SP can be used to **connect to and orchestrate jobs on the HPC systems**, enabling hybrid workflows that combine ease of use with computational scalability.

This FAQ provides comprehensive guidance for the **Multimodal Optimization (MMO)** use case, focusing on the **Scalable Hybrid Genetic Algorithm (SHGA)** demonstrator hosted on the **Norwegian Artificial Intelligence Cloud (NAIC)**. SHGA, developed by Johannsen et al. (2022), integrates a genetic algorithm with deterministic crowding for global exploration and CMA-ES for local refinement, tailored for continuous multimodal optimization. This document addresses infrastructure, technical setup, data handling, and collaboration, emphasizing the SHGA demonstrator’s application within NAIC’s high-performance computing environment.

---

## A. Understanding the MMO Use Case

### 1. What is Multimodal Optimization and how does it differ from traditional optimization?
Multimodal Optimization (MMO) aims to locate multiple optimal or near-optimal solutions in a problem space, unlike traditional optimization, which targets a single global optimum. Traditional methods, such as gradient descent or linear programming, excel in unimodal problems where one best solution exists, but they struggle with multimodal landscapes featuring multiple peaks. MMO employs techniques like niching—seen in SHGA’s deterministic crowding—to maintain population diversity and explore various optima. For instance, in designing a wind turbine blade, traditional optimization might yield one shape with maximum efficiency, while MMO could identify several shapes with similar efficiency but differing weights or costs, offering practical trade-offs. This flexibility makes MMO invaluable in domains requiring diverse, viable solutions rather than a singular “best” answer.

### 2. What is the SHGA algorithm and how does it work in the demonstrator?
The Scalable Hybrid Genetic Algorithm (SHGA) combines global and local search strategies for continuous multimodal optimization (Johannsen et al., 2022). It uses a genetic algorithm (GA) enhanced with deterministic crowding, where offspring compete only with their closest parent, preserving diversity across the solution space. This is paired with the Covariance Matrix Adaptation Evolution Strategy (CMA-ES), which refines solutions locally by adapting its search distribution to the fitness landscape’s curvature. In the demonstrator (`demonstrator_non_dask.ipynb`), SHGA initializes a population and iteratively evolves solutions over generations. Outputs include logs (e.g., "Iteration 10: found 6 optima") and visualizations like scatter plots of solutions (e.g., optima at x = [0.1, 0.3, 0.5] for the Equal Maxima function), showcasing its ability to detect and refine multiple peaks efficiently.

### 3. Why is Multimodal Optimization important for my work?
MMO addresses real-world problems where multiple solutions offer distinct advantages, unlike single-optimum methods that may overlook practical alternatives. In mechanical engineering, MMO might optimize a gear system for efficiency, revealing designs with similar performance but varying durability or cost—options a traditional approach might miss. In bioinformatics, it could identify multiple protein configurations with comparable binding affinity, aiding drug design. For the SHGA demonstrator, this means you can leverage NAIC’s computational power to explore such scenarios, whether tuning machine learning models, optimizing supply chains, or designing sustainable infrastructure. MMO’s strength lies in its ability to provide a spectrum of solutions, enhancing decision-making across disciplines.

### 4. What types of problems is the SHGA demonstrator best suited for?
The SHGA demonstrator excels in continuous multimodal optimization problems with moderate dimensionality (e.g., 2–20 variables). It’s ideal for non-convex problems with multiple optima, such as mathematical benchmarks (Himmelblau, Shubert) or applied tasks like optimizing antenna placement for signal coverage, where multiple configurations might perform similarly but differ in cost or feasibility. The demonstrator shines when objective functions are computationally expensive—e.g., simulations or finite element analyses—due to its parallel evaluation capabilities on NAIC. It’s less suited for discrete or unimodal problems, where simpler algorithms might suffice, but its hybrid design makes it versatile for complex, real-world challenges requiring both exploration and precision.

### 5. How do I interpret the output of the SHGA demonstrator?
The SHGA demonstrator generates detailed outputs to assess optimization results. These include a list of solutions (e.g., variable vectors and fitness values like `[0.1, 0.9], fitness = 1.0`), logged progress (e.g., "Iteration 20: 8 solutions found"), and Matplotlib visualizations—convergence plots tracking fitness over generations and scatter plots mapping solutions in the search space. For a problem like the Shubert function, you might see optima clustered around known peaks, with fitness values approaching theoretical maxima. To interpret, verify solution diversity (are multiple peaks captured?), compare fitness to expected benchmarks, and use plots to assess exploration (wide coverage) versus exploitation (tight clusters). This multifaceted output helps gauge SHGA’s effectiveness for your specific problem.

---

## B. Running the Demonstrator

### 6. What are the prerequisites to run the SHGA demonstrator?
Running the SHGA demonstrator requires specific software and knowledge:
- **Software**: Python 3.8+, with libraries like NumPy, DEAP, Pandas, Scikit-learn, Jupyter, and Matplotlib. Install via `pip install -r requirements.txt`.
- **Hardware**: A local machine suffices for small tests, but NAIC’s Sigma2 clusters (e.g., SAGA) are recommended for scalability, requiring an account from [metacenter.no](https://metacenter.no).
- **Skills**: Basic Python proficiency, understanding of optimization (e.g., fitness functions), and optional familiarity with HPC tools like SLURM.  
For example, a novice might install Python locally and run a 2-variable test, while an advanced user could configure a 20-variable problem on NAIC with parallel workers. The notebook includes setup instructions, making it accessible with minimal prior HPC experience.

### 7. How do I run the SHGA demonstrator locally versus on NAIC?

- **Local Execution**: Suitable for small-scale testing. Install dependencies (`pip install -r requirements.txt`), launch Jupyter (`jupyter notebook`), and run `demonstrator.ipynb`. This is ideal for quick experiments with low-dimensional problems.

- **Sigma2 HPC Execution via NAIC**: The SHGA demonstrator is designed to run on Sigma2’s HPC systems (e.g., SAGA, FRAM, BETZY), which provide the necessary computational resources. Access is managed through the NAIC platform, which facilitates authentication, resource allocation, and job submission.

- **Sigma2 Service Platform (SP)**: The SP offers a cloud-like environment for deploying AI services and workflows. However, due to current resource limitations, the SHGA demonstrator cannot run efficiently on the SP alone. Instead, the SP can be used to **interface with the HPC systems**, enabling workflows where data preprocessing or orchestration occurs on the SP, while heavy computation is offloaded to HPC.

For example, a user might use the SP to manage datasets and trigger SHGA runs on SAGA via NAIC’s integrated tools.

- **NAIC Execution**: Log into SAGA (`ssh username@login.saga.sigma2.no`), load Python (`module load Python/3.10.4`), create a virtual environment (`python -m venv myenv; source myenv/bin/activate`), install dependencies, and request resources (`srun --ntasks=1 --cpus-per-task=8 --mem=16G --time=2:00:00 --pty bash`). Launch Jupyter Lab (`jupyter lab --ip=0.0.0.0 --port=8888 --no-browser`), tunnel locally (`ssh -L 8888:<node>:8888 username@login.saga.sigma2.no`), and access via `http://localhost:8888`. This scales to larger problems (e.g., 20 variables, 10,000 evaluations), leveraging NAIC’s parallel processing to complete in under an hour.

### 8. What does "Parallel Evaluation" mean in the context of SHGA?
Parallel Evaluation in SHGA means distributing fitness calculations across multiple CPU cores or nodes, accelerating computation for large populations or costly objective functions. For a simulation-heavy problem (e.g., aerodynamic modeling), this could cut a 5-hour sequential run substantially with sufficient compute resources.

### 9. Why do I get "library $x$ not found" when running block 3 of the notebook?
This error arises if a dependency (e.g., DEAP) isn’t installed or accessible. Locally, you might have skipped `pip install -r requirements.txt` or used an incompatible Python version. On NAIC, you may not have activated your virtual environment (`source myenv/bin/activate`) or loaded the correct Python module (`module load Python/3.10.4`). To fix, reinstall dependencies (`pip install -r requirements.txt`), ensure the environment matches the notebook’s requirements, and check logs for specifics.

### 10. What should I do if the demonstrator takes too long with my data?
Slow execution signals resource or configuration issues. Locally, test with a smaller population (e.g., 50 instead of 500) or fewer variables to isolate bottlenecks. On NAIC, increase parallelism (`cluster.scale(jobs=20)`), request more resources (`--cpus-per-task=16 --mem=32G`), or optimize your objective function—e.g., replace a full simulation with a surrogate model for initial runs. If a 20-variable problem takes 5 hours, logs might reveal excessive evaluation time; reducing to 10 variables might drop it to 1 hour, confirming scalability limits. Enable verbose logging in the notebook to pinpoint delays, ensuring efficient use of NAIC’s infrastructure.

---

## C. Customizing and Using Your Data

### 11. How do I adapt the SHGA demonstrator for my own optimization problem?
Customizing SHGA involves:
1. **Objective Function**: Code your problem, e.g., `def my_func(x): return sum(x**2) - np.cos(10 * x)`, ensuring it returns a scalar fitness.
2. **Bounds**: Set variable ranges, e.g., `bounds = [(-5, 5)] * 10` for 10 variables.
3. **Notebook Edit**: Replace the default `evaluate` function in `demonstrator.ipynb` with yours, adjusting population size or generations as needed.
4. **Validation**: Run locally with small settings (e.g., 100 evaluations) to check outputs match expectations.
5. **Scale**: Deploy on NAIC with larger settings, leveraging parallel evaluation.  
For example, optimizing a logistic model might involve defining a loss function over 5 parameters, testing locally, then running 10,000 evaluations on NAIC to find multiple minima.

### 12. Where can I find training data for the demonstrator?
The demonstrator uses the CEC 2013 niching benchmark suite, featuring functions like Equal Maxima and Shubert, available at [github.com/mikeagn/CEC2013](https://github.com/mikeagn/CEC2013). Download the suite, extract function definitions (e.g., Python or MATLAB implementations), and integrate them into the notebook via NumPy or direct coding. For custom data, prepare a similar structure—input vectors and expected optima—ensuring compatibility with SHGA’s continuous optimization framework. This public dataset provides a robust starting point for testing and validation.

### 13. How do I upload and share my results from the demonstrator?
On NAIC, save results to `/cluster/projects/myproject` (e.g., `np.save('results.npy', solutions)`), a persistent, shareable directory. Generate plots (`plt.savefig('convergence.png')`) and logs, then transfer files locally via `scp` or share directly with collaborators via NAIC permissions. For broader dissemination, upload to GitHub or Zenodo, including the notebook, results, and a README detailing your problem and findings. For instance, a repository might contain `demonstrator.ipynb`, `results.csv`, and a plot showing 5 optima, with instructions for replication.

### 14. Am I allowed to share the benchmarking data?
Yes, the CEC 2013 data is open-source, permitting use and sharing for research. When distributing, attribute it as “Benchmark functions from the 2013 IEEE CEC niching competition” to credit the source. You could include it in a repository with your results, ensuring transparency and enabling others to replicate or build on your work, all while adhering to academic norms.

### 15. How can I ensure reproducibility of my SHGA runs?
Reproducibility requires:
- **Seed**: Fix randomness with `np.random.seed(123)`.
- **Parameters**: Document settings (e.g., population = 200, generations = 50) in the notebook or a config file.
- **Dependencies**: Share `requirements.txt` or a Docker setup.
- **Data**: Use CEC 2013 benchmarks or version custom data with checksums.
- **Notes**: Add comments explaining key steps (e.g., “Fitness evaluation parallelized over 8 cores”).  
For example, a colleague could rerun your 10-variable Himmelblau optimization with identical solutions (e.g., `[3, 2], fitness = 0`), validating your findings.

---

## D. Leveraging NAIC for MMO

### 16. How do I determine the resources needed for my SHGA run?
Estimate based on:
- **Problem Complexity**: A 10-variable problem with 1,000 evaluations might need 8 cores, 16 GB RAM, and 2 hours.
- **Function Cost**: Expensive evaluations (e.g., 1 second each) demand more parallelism (e.g., 16 cores).
- **Testing**: Run locally first—e.g., 100 evaluations in 5 minutes suggests 1,000 on NAIC with 10 cores takes ~10 minutes.  
Request via SLURM: `srun --ntasks=1 --cpus-per-task=8 --mem=16G --time=2:00:00`. Adjust after initial runs to optimize NAIC usage.

### 17. Where can I find information on available NAIC resources?
[Sigma2.no](https://www.sigma2.no) details NAIC clusters:
- **SAGA**: General-purpose, good for SHGA’s moderate needs.
- **FRAM**: High-memory, ideal for large datasets.
- **BETZY**: Compute-intensive, suited for heavy parallel tasks.  
Each offers node specs (e.g., 128 GB/node on SAGA) and SLURM guides. For SHGA, SAGA’s balance of CPU and memory typically suffices, but FRAM might help with memory-intensive visualizations.

### 18. How much does it cost to run the SHGA demonstrator on NAIC?
NAIC charges per CPU hour (details at Sigma2.no), but research allocations may be free—apply via their portal. Estimate: 1,000 evaluations at 0.1 seconds each, 10 cores = 10 seconds wall time, 0.028 CPU hours. A 24-hour run with 10 cores = 240 CPU hours. Test small runs to refine estimates, minimizing costs while meeting goals.

### 19. How do I set up Jupyter Lab on NAIC for the demonstrator?
Steps:
1. Log in: `ssh username@login.saga.sigma2.no`.
2. Load Python: `module load Python/3.10.4`.
3. Create environment: `python -m venv myenv; source myenv/bin/activate; pip install -r requirements.txt`.
4. Request node: `srun --ntasks=1 --cpus-per-task=4 --mem=8G --time=2:00:00 --pty bash`.
5. Start Jupyter: `jupyter lab --ip=0.0.0.0 --port=8888 --no-browser`.
6. Tunnel: `ssh -L 8888:<node>:8888 username@login.saga.sigma2.no`.
7. Access: `http://localhost:8888`.  
This setup supports interactive SHGA runs with real-time visualization.

### 20. How do I use GPUs for my optimization problem if needed?
For GPU-accelerated functions (e.g., neural network evaluations):
- Request: `srun --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=2:00:00`.
- Load: `module load CUDA/11.0 cuDNN/8.0`.
- Code: Modify your objective to use GPUs (e.g., PyTorch’s `tensor.cuda()`).
- Check: `nvidia-smi` for usage.  
SHGA’s core doesn’t need GPUs, but a GPU-enabled function could cut evaluation time significantly, e.g., from 1 second to 0.01 seconds per call.

---

## E. Collaboration and Next Steps

### 21. How do I get support for adapting the SHGA demonstrator to my project?
Email NAIC support via their portal, detailing your problem (e.g., 15-variable supply chain optimization), current setup (e.g., local vs. NAIC), and issues (e.g., slow convergence). They can advise on code tweaks, resource allocation, or debugging—e.g., suggesting a higher mutation rate for better diversity. Response times vary, but detailed queries get faster, actionable help.

### 22. How should I cite the SHGA demonstrator in my work?
Use: “This research utilized the Scalable Hybrid Genetic Algorithm (SHGA) demonstrator on the Norwegian Artificial Intelligence Cloud (NAIC), based on Johannsen et al. (2022).” Reference: Johannsen, K., et al. (2022). *Nordic Machine Intelligence, 2*(1), 16–27. [DOI:10.5617/nmi.9633](https://doi.org/10.5617/nmi.9633). For major adaptations, discuss co-authorship with NAIC developers via email.

### 23. Can I share the benchmarking data used in the demonstrator?
Yes, CEC 2013 data is public. Share it with attribution: “Data from the 2013 IEEE CEC niching competition.” Include it in repositories or papers, enhancing reproducibility without legal concerns.

### 24. What are the next steps after running the demonstrator?
- **Evaluate**: Check if solutions meet goals (e.g., 5 optima within 1% of known values).
- **Expand**: Test larger problems or refine parameters on NAIC.
- **Share**: Publish results, citing NAIC/SHGA, or present at conferences.
- **Iterate**: Add constraints (e.g., budget limits) and rerun.  
For example, after finding 3 optima in a design task, you might scale to 10 variables or collaborate with NAIC on a custom version.

### 25. How can I contribute to improving the SHGA demonstrator?
Submit feedback or bug reports via NAIC’s portal or GitHub (if public). Propose features (e.g., interactive plots), share code patches (e.g., new niching methods), or enhance docs with examples. For instance, adding a tutorial for a 3-variable problem could broaden its appeal, benefiting the community.

---

## References
Johannsen, K., et al. (2022). A scalable, hybrid genetic algorithm for continuous multimodal optimization in moderate dimensions. *Nordic Machine Intelligence, 2*(1), 16–27. https://doi.org/10.5617/nmi.9633