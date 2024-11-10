# Crested Porcupine Optimizer (CPO) in Python

### Overview

This project provides a **Python implementation and replication** of the **Crested Porcupine Optimizer (CPO)**, a nature-inspired metaheuristic algorithm originally proposed in the paper *"Crested Porcupine Optimizer: A New Nature-Inspired Metaheuristic"* by Mohamed Abdel-Basset et al. Our implementation aims to faithfully reproduce the algorithm's behaviors and validate its effectiveness through various tests, including **benchmark functions** and **path planning** applications.

### Features

- **CPO Algorithm Replication**: A complete Python replication of the original CPO algorithm, with configurable parameters for various optimization tasks.
- **Benchmark Testing on CEC Functions**: Validated on CEC benchmark functions (CEC2017, CEC2020, etc.), comparing the algorithm’s performance and robustness.
- **A-Enhanced Path Planning with CPO**: Applies CPO to enhance traditional A* path planning, balancing exploration and exploitation for optimized paths in complex environments.
- **Randomized Obstacle Map Generation**: Generates random obstacle maps to test path planning performance in varied scenarios.
- **Multi-Objective Optimization**: Extends CPO for multi-objective optimization, supporting applications that require Pareto-optimal front analysis.

### Repository Structure

- **`CPO_cec.py`**: Core implementation of the Crested Porcupine Optimizer with benchmark functions, replicating the original paper’s approach.
- **`CPO_algorithm_vX.py`**: Variants of the CPO algorithm, exploring optimizations and visualization enhancements.
- **`CPO_path_planning.py`**: CPO-based path planning for grid maps, integrating A* for a hybrid approach.
- **`CPO_multiobjective.py`**: Multi-objective optimization using CPO, enabling Pareto front analysis.
- **`optimization.py`**: Utility functions for general optimization tasks and benchmarking.
- **Reports and Data**: Experimental results, performance comparisons, and data files supporting reproducibility.

### Key Concepts

- **Defensive Mechanisms of CPO**: Each phase of CPO mirrors a defensive behavior observed in porcupines:
  - *Sight*: Broad random exploration.
  - *Sound*: Social interaction leading to further exploration.
  - *Odor*: Convergence towards promising solutions with diversity retention.
  - *Physical Attack*: Final precision adjustments around optimal solutions.
- **Cyclic Population Reduction**: The population size decreases over time, focusing on promising solutions, replicating the porcupine’s response under threat.

### Getting Started

1. **Dependencies**: Install required libraries.
   ```bash
   pip install numpy matplotlib

2. **Run a Benchmark Test**:

   ```bash
   python CPO_cec.py
   ```

3. **Path Planning Example**:

   ```bash
   python CPO_path_planning.py
   ```

### Example Results

This repository includes sample results comparing the CPO algorithm with other popular optimizers on benchmark functions such as Sphere, Rosenbrock, and Schwefel functions, showcasing CPO's balance between exploration and convergence. For path planning, CPO demonstrates potential in generating non-overlapping, direct paths and avoiding local minima.

### Acknowledgment

This repository is a replication and extension of the CPO algorithm based on the original paper:

- **Original Paper**: "Crested Porcupine Optimizer: A New Nature-Inspired Metaheuristic" by Mohamed Abdel-Basset et al., published in *Knowledge-Based Systems*.

We would also like to express our sincere thanks to **Dr.Chengzhi Qu** from **Nanjing university of information and science technoledge**, for his valuable guidance and for providing the topic for this project.


### Additional Documentation

- **NUIST Experiment Report**: Contains detailed documentation of our experimental setup, results, and analysis of CPO on various benchmark functions and path planning tasks.



