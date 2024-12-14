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

#### 1. **Path Planning Implementations**
- **`A*-CPO_path_2d.py`**: Implements 2D path planning by combining the Crested Porcupine Optimizer (CPO) with the A* algorithm for hybrid route planning.
- **`A*-CPO_path_3d.py`**: Extends the 2D path planning approach to 3D environments using the A*-CPO hybrid method.
- **`A*-CPO_path_3d_v2.py`**: An improved version of the 3D path planning script with optimizations for enhanced performance.

#### 2. **Benchmarking and Testing**
- **`CPO_cec2017.py`**: Contains benchmark functions from the CEC 2017 suite, supporting the testing of CPO algorithm performance.
- **`CPO_cec2022.py`**: Similar to `CPO_cec2017.py`, but using benchmark functions from the CEC 2022 suite.
- **`CPO_convergenceCurve.py`**: Visualizes convergence curves to assess the performance and stability of various CPO algorithm versions.

#### 3. **Multi-Objective Optimization**
- **`CPO_multi_v3.py`**: Implements multi-objective optimization using CPO, with support for Pareto front visualization.
- **`CPO_multi_v4.py`**: Extends the multi-objective optimization capabilities of CPO, adding support for three objectives.

#### 4. **Population Dynamics and Core Algorithm**
- **`CPO_population.py`**: Manages population dynamics within the CPO framework, supporting algorithm functionality.
- **`CPO_algorithm_v4_class.py`**: Implements the object-oriented version of the Crested Porcupine Optimizer (CPO) algorithm using classes, encapsulating the core functionalities, including fitness functions, particles, and the optimizer itself.
- **`CPO_algorithm_v4_OOP.py`**: Implements the refactored Crested Porcupine Optimizer (CPO) algorithm using object-oriented programming principles. This file demonstrates the application of encapsulation, inheritance, polymorphism, and other OOP features to improve the maintainability and scalability of the algorithm.

### 5. **Other Files**
- **`expriment_report.pdf`**: A report documenting experimental results, performance comparisons, and analyses of the original CPO algorithm.
- **`CPO_OOP_report.pdf`**: A report discussing the object-oriented version of the CPO algorithm, explaining the application of classes, private members, protected members, inheritance, and polymorphism, and how these features improve maintainability and scalability of the algorithm.
- **`CPO_OOP_report2.pdf`**: A detailed report accompanying the `CPO_algorithm_v4_OOP.py` implementation. The report explains the object-oriented design, its benefits, and performance evaluations compared to the procedural version.
- **`README.md`**: Documentation and overview of the repository, providing an introduction to the CPO algorithm, its various implementations, and instructions on usage.



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

We would also like to express our sincere thanks to **Dr.Chengzhi Qu** from **Nanjing university of information science and technoledge**, for his valuable guidance and for providing the topic for this project.


### Additional Documentation

- **NUIST Experiment Report**: Contains detailed documentation of our experimental setup, results, and analysis of CPO on various benchmark functions and path planning tasks.

### CEC2017 Benchmark Function Visualization

The following figures showcase the performance of the Crested Porcupine Optimizer (CPO) on various CEC2017 benchmark functions (F1-F6). Each function includes a surface plot, convergence curve, and performance comparison with other optimizers.

#### F1: Sphere Function
| Function Surface and Contour | Convergence Curve | Performance Comparison |
|:-----------------------------:|:-----------------:|:----------------------:|
| ![F1 Surface and Contour](images/F_1.3.png) | ![F1 Convergence](images/F_1.png) | ![F1 Performance](images/F_1.2.png) |

#### F2: Rosenbrock Function
| Function Surface and Contour | Convergence Curve | Performance Comparison |
|:-----------------------------:|:-----------------:|:----------------------:|
| ![F2 Surface and Contour](images/F_2.3.png) | ![F2 Convergence](images/F_2.png) | ![F2 Performance](images/F_2.2.png) |

#### F3: Ackley Function
| Function Surface and Contour | Convergence Curve | Performance Comparison |
|:-----------------------------:|:-----------------:|:----------------------:|
| ![F3 Surface and Contour](images/F_3.3.png) | ![F3 Convergence](images/F_3.png) | ![F3 Performance](images/F_3.2.png) |

#### F4: Rastrigin Function
| Function Surface and Contour | Convergence Curve | Performance Comparison |
|:-----------------------------:|:-----------------:|:----------------------:|
| ![F4 Surface and Contour](images/F_4.3.png) | ![F4 Convergence](images/F_4.png) | ![F4 Performance](images/F_4.2.png) |

#### F5: Griewank Function
| Function Surface and Contour | Convergence Curve | Performance Comparison |
|:-----------------------------:|:-----------------:|:----------------------:|
| ![F5 Surface and Contour](images/F_5.3.png) | ![F5 Convergence](images/F_5.png) | ![F5 Performance](images/F_5.2.png) |

#### F6: Schwefel Function
| Function Surface and Contour | Convergence Curve | Performance Comparison |
|:-----------------------------:|:-----------------:|:----------------------:|
| ![F6 Surface and Contour](images/F_6.3.png) | ![F6 Convergence](images/F_6.png) | ![F6 Performance](images/F_6.2.png) |

### Additional Analysis

These visualizations demonstrate the Crested Porcupine Optimizer’s ability to handle various optimization landscapes, showing convergence speed and accuracy across different types of functions. Each function presents unique challenges, and CPO's performance on these benchmarks highlights its robustness and adaptability.

## License

This project is licensed under the **MIT License (Adapted for Research Code Reproduction)**. 

This repository contains a reproduction of the Crested Porcupine Optimizer (CPO) algorithm as described in:
- Abdel-Basset, Mohamed et al. (2024). "Crested Porcupine Optimizer: A New Nature-Inspired Metaheuristic".

For more details, see the [LICENSE](LICENSE) file.

## Reference

This repository implements and extends the **Crested Porcupine Optimizer (CPO)**, a novel nature-inspired metaheuristic algorithm introduced in the following paper:

> **Crested Porcupine Optimizer: A New Nature-Inspired Metaheuristic**  
> Mohamed Abdel-Basset, Reda Mohamed, and Mohamed Abouhawwash  
> *Knowledge-Based Systems, Volume 284, 2024, 111257.*  
> [DOI: https://doi.org/10.1016/j.knosys.2023.111257](https://doi.org/10.1016/j.knosys.2023.111257)

### About This Repository

This repository provides a Python implementation of the CPO algorithm, which was originally proposed in the above paper. While the original paper does not provide python code, this repository reproduces the algorithm in Python and extends it using **object-oriented programming (OOP)** principles to improve maintainability, modularity, and scalability. Additional features and applications, such as path planning and benchmarking, are also included.

If you use this repository or any part of the implementation in your research or projects, please cite both the original paper and this GitHub repository as follows:

### Citation

#### Original Paper
```bibtex
@article{abdel2024crested,
  title={Crested Porcupine Optimizer: A new nature-inspired metaheuristic},
  author={Abdel-Basset, Mohamed and Mohamed, Reda and Abouhawwash, Mohamed},
  journal={Knowledge-Based Systems},
  volume={284},
  pages={111257},
  year={2024},
  publisher={Elsevier}
}
```
```bibtex
@misc{wang2024cpo,
  author       = {Ziheng Wang,Di Wu},
  title        = {Python Implementation of the Crested Porcupine Optimizer (CPO)},
  year         = {2024},
  url          = {https://github.com/Nickory/CPO-python},
  note         = {A Python implementation of the CPO algorithm based on the paper by Mohamed Abdel-Basset et al.}
}
```

