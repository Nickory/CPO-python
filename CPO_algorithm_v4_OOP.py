"""
Crested Porcupine Optimizer (CPO) - Python Implementation

This project provides a **Python implementation and replication** of the **Crested Porcupine Optimizer (CPO)**,
a nature-inspired metaheuristic algorithm originally proposed in the paper *"Crested Porcupine Optimizer:
A New Nature-Inspired Metaheuristic"* by Mohamed Abdel-Basset et al. The CPO algorithm simulates the
behavior of the crested porcupine to solve optimization problems.

Our Python implementation faithfully reproduces the algorithm's behaviors and validates its effectiveness
through various tests, including **benchmark functions** and **path planning** applications.

The original algorithm was adapted and implemented in Python by Ziheng Wang (zhwang@nuist.edu.cn).
This version improves upon the original by utilizing object-oriented programming (OOP) principles,
encapsulation, and exception handling to ensure robustness, modularity, and maintainability.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re


class BenchmarkFunction:
    """
    Base class for benchmark functions, supporting inheritance and encapsulation.
    Provides various mathematical benchmark functions used for optimization problems.
    """
    _functions = {
        1: "sphere_func",
        2: "schwefel_222_func",
        3: "powell_sum_func",
        4: "schwefel_12_func",
        5: "schwefel_221_func",
        6: "rosenbrock_func",
        7: "step_func",
        8: "quartic_func",
        9: "zakharov_func",
    }

    @staticmethod
    def sphere_func(x):
        """
        Sphere function: f(x) = sum(x_i^2)
        """
        return np.sum(x ** 2)

    @staticmethod
    def schwefel_222_func(x):
        """
        Schwefel 2.22 function: f(x) = sum(abs(x_i)) + prod(abs(x_i))
        """
        return np.sum(np.abs(x)) + np.prod(np.abs(x))

    @staticmethod
    def powell_sum_func(x):
        """
        Powell sum function: f(x) = sum(abs(x_i)^(i+1))
        """
        return np.sum(np.abs(x) ** (np.arange(len(x)) + 1))

    @staticmethod
    def schwefel_12_func(x):
        """
        Schwefel 1.2 function: f(x) = sum(sum(x[:i+1])^2)
        """
        return np.sum([np.sum(x[:i + 1]) ** 2 for i in range(len(x))])

    @staticmethod
    def schwefel_221_func(x):
        """
        Schwefel 2.21 function: f(x) = max(abs(x_i))
        """
        return np.max(np.abs(x))

    @staticmethod
    def rosenbrock_func(x):
        """
        Rosenbrock function: f(x) = sum(100*(x_i+1 - x_i^2)^2 + (x_i - 1)^2)
        """
        return np.sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)])

    @staticmethod
    def step_func(x):
        """
        Step function: f(x) = sum((x_i + 0.5)^2)
        """
        return np.sum((x + 0.5) ** 2)

    @staticmethod
    def quartic_func(x):
        """
        Quartic function with noise: f(x) = sum(i*x_i^4) + random_noise
        """
        return np.sum((np.arange(1, len(x) + 1) * x ** 4)) + np.random.uniform(0, 1)

    @staticmethod
    def zakharov_func(x):
        """
        Zakharov function: f(x) = sum(x_i^2) + (sum(0.5 * i * x_i))^2 + (sum(0.5 * i * x_i))^4
        """
        term1 = np.sum(x ** 2)
        term2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 2
        term3 = np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 4
        return term1 + term2 + term3

    @classmethod
    def get_function(cls, func_num):
        """
        Returns the function corresponding to the func_num.
        Ensures that the provided func_num is valid.
        """
        if not re.match(r"^[1-9]$", str(func_num)):
            raise ValueError("Invalid function number. Must be between 1 and 9.")

        func_name = cls._functions.get(func_num)
        if func_name is None:
            raise ValueError("Function number out of range.")

        return getattr(cls, func_name)


class CPOOptimizer:
    """
    CPOOptimizer implements the Cognitive Particle Optimization (CPO) algorithm for optimization tasks.
    It uses exception handling and encapsulation to ensure robust and maintainable code.
    """

    def __init__(self, pop_size, Tmax, ub, lb, dim, func_num):
        """
        Initializes the optimizer with necessary parameters for optimization.

        Parameters:
        - pop_size (int): Size of the population
        - Tmax (int): Maximum number of iterations
        - ub (float): Upper bound of the search space
        - lb (float): Lower bound of the search space
        - dim (int): Dimensionality of the problem
        - func_num (int): Number representing the benchmark function to optimize
        """
        try:
            self.__pop_size = pop_size
            self.__Tmax = Tmax
            self.__ub = ub
            self.__lb = lb
            self.__dim = dim
            self.__func_num = func_num
            self.__benchmark_func = BenchmarkFunction.get_function(func_num)
        except ValueError as e:
            raise RuntimeError(f"Initialization error: {e}")

    def _initialization(self):
        """
        Initializes the population with random values within the defined bounds.
        """
        return np.random.rand(self.__pop_size, self.__dim) * (self.__ub - self.__lb) + self.__lb

    def optimize(self):
        """
        Runs the CPO optimization process and returns the best solution found.
        """
        try:
            Gb_Fit = np.inf  # Global best fitness value
            Gb_Sol = None  # Global best solution
            Conv_curve = np.zeros(self.__Tmax)  # Convergence curve
            X = self._initialization()  # Initialize population
            fitness = np.array([self.__benchmark_func(X[i, :]) for i in range(self.__pop_size)])
            Gb_Fit, index = np.min(fitness), np.argmin(fitness)
            Gb_Sol = X[index, :]
            Xp = np.copy(X)  # Store previous population
            t = 0

            # Main optimization loop
            while t < self.__Tmax and Gb_Fit > 0:
                for i in range(len(X)):
                    # Update each individual based on CPO strategy
                    U1 = np.random.rand(self.__dim) > np.random.rand(self.__dim)
                    rand_index1 = np.random.randint(len(X))
                    rand_index2 = np.random.randint(len(X))

                    if np.random.rand() < np.random.rand():
                        y = (X[i, :] + X[rand_index1, :]) / 2
                        X[i, :] = X[i, :] + np.random.randn(self.__dim) * np.abs(2 * np.random.rand() * Gb_Sol - y)
                    else:
                        Yt = 2 * np.random.rand() * (1 - t / self.__Tmax) ** (t / self.__Tmax)
                        U2 = np.random.rand(self.__dim) < 0.5
                        S = np.random.rand() * U2
                        if np.random.rand() < 0.8:
                            St = np.exp(fitness[i] / (np.sum(fitness) + np.finfo(float).eps))
                            S = S * Yt * St
                            X[i, :] = (1 - U1) * X[i, :] + U1 * (
                                    X[rand_index1, :] + St * (X[rand_index2, :] - X[rand_index1, :]) - S)
                        else:
                            Mt = np.exp(fitness[i] / (np.sum(fitness) + np.finfo(float).eps))
                            Vtp = X[rand_index1, :]
                            Ft = np.random.rand(self.__dim) * (Mt * (-X[i, :] + Vtp))
                            S = S * Yt * Ft
                            X[i, :] = (Gb_Sol + (0.2 * (1 - np.random.rand()) + np.random.rand()) * (
                                    U2 * Gb_Sol - X[i, :])) - S

                    X[i, :] = np.clip(X[i, :], self.__lb, self.__ub)  # Ensure solution is within bounds
                    nF = self.__benchmark_func(X[i, :])  # Calculate fitness for new solution
                    if fitness[i] < nF:
                        X[i, :] = Xp[i, :]  # Revert solution if fitness worsens
                    else:
                        Xp[i, :] = X[i, :]
                        fitness[i] = nF
                        if nF <= Gb_Fit:
                            Gb_Sol = X[i, :]
                            Gb_Fit = nF

                Conv_curve[t] = Gb_Fit  # Store the fitness at each iteration
                t += 1

            return Gb_Fit, Gb_Sol, Conv_curve  # Return final fitness, solution, and convergence curve

        except Exception as e:
            raise RuntimeError(f"Optimization error: {e}")


def run_all_tests():
    """
    Runs optimization tests for all benchmark functions (1 through 9).
    """
    for func_num in range(1, 10):
        print(f"Testing function {func_num}...")
        optimizer = CPOOptimizer(pop_size=50, Tmax=500, ub=100, lb=-100, dim=10, func_num=func_num)
        Gb_Fit, Gb_Sol, Conv_curve = optimizer.optimize()
        print(f"Function {func_num}: Final fitness: {Gb_Fit}")


def main():
    """
    Main function to execute the optimization tests.
    """
    run_all_tests()


if __name__ == "__main__":
    main()

