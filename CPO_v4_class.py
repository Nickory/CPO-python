import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class BenchmarkFunction:
    """
    This class provides various benchmark functions for optimization algorithms.
    All functions are static methods, as they do not depend on instance attributes.
    """

    @staticmethod
    def sphere_func(x):
        """Sphere function: f(x) = sum(x_i^2)"""
        return np.sum(x ** 2)

    @staticmethod
    def schwefel_222_func(x):
        """Schwefel 2.22 function: f(x) = sum(abs(x_i)) + prod(abs(x_i))"""
        return np.sum(np.abs(x)) + np.prod(np.abs(x))

    @staticmethod
    def powell_sum_func(x):
        """Powell sum function: f(x) = sum(abs(x_i)^(i+1))"""
        return np.sum(np.abs(x) ** (np.arange(len(x)) + 1))

    @staticmethod
    def schwefel_12_func(x):
        """Schwefel 1.2 function: f(x) = sum(sum(x_1...x_i)^2)"""
        return np.sum([np.sum(x[:i + 1]) ** 2 for i in range(len(x))])

    @staticmethod
    def schwefel_221_func(x):
        """Schwefel 2.21 function: f(x) = max(abs(x_i))"""
        return np.max(np.abs(x))

    @staticmethod
    def rosenbrock_func(x):
        """Rosenbrock function: f(x) = sum(100*(x_i+1 - x_i^2)^2 + (x_i - 1)^2)"""
        return np.sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)])

    @staticmethod
    def step_func(x):
        """Step function: f(x) = sum((x_i + 0.5)^2)"""
        return np.sum((x + 0.5) ** 2)

    @staticmethod
    def quartic_func(x):
        """Quartic function with noise: f(x) = sum(i*x_i^4) + random noise"""
        return np.sum((np.arange(1, len(x) + 1) * x ** 4)) + np.random.uniform(0, 1)

    @staticmethod
    def zakharov_func(x):
        """Zakharov function: f(x) = sum(x_i^2) + sum(0.5*i*x_i)^2 + sum(0.5*i*x_i)^4"""
        term1 = np.sum(x ** 2)
        term2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 2
        term3 = np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 4
        return term1 + term2 + term3

    @classmethod
    def get_function(cls, func_num):
        """
        Returns the function corresponding to the func_num.
        """
        functions = {
            1: cls.sphere_func,
            2: cls.schwefel_222_func,
            3: cls.powell_sum_func,
            4: cls.schwefel_12_func,
            5: cls.schwefel_221_func,
            6: cls.rosenbrock_func,
            7: cls.step_func,
            8: cls.quartic_func,
            9: cls.zakharov_func
        }
        if func_num not in functions:
            raise ValueError("Invalid function number")
        return functions[func_num]


class CPOOptimizer:
    """
    CPOOptimizer implements the CPO (Cognitive Particle Optimization) algorithm.
    It includes the population initialization and optimization steps.
    """

    def __init__(self, pop_size, Tmax, ub, lb, dim, func_num):
        """
        Initialize the optimizer with key parameters.

        :param pop_size: Population size
        :param Tmax: Maximum iterations
        :param ub: Upper bound for the search space
        :param lb: Lower bound for the search space
        :param dim: Dimensionality of the problem
        :param func_num: Benchmark function number
        """
        self.pop_size = pop_size
        self.Tmax = Tmax
        self.ub = ub
        self.lb = lb
        self.dim = dim
        self.func_num = func_num
        self.benchmark_func = BenchmarkFunction.get_function(func_num)

    def _initialization(self):
        """Initialize the population with random solutions within bounds."""
        return np.random.rand(self.pop_size, self.dim) * (self.ub - self.lb) + self.lb

    def optimize(self):
        """
        Perform the CPO optimization algorithm.

        :return: Best fitness, best solution, and convergence curve
        """
        # Initialize variables
        Gb_Fit = np.inf
        Gb_Sol = None
        Conv_curve = np.zeros(self.Tmax)
        X = self._initialization()
        fitness = np.array([self.benchmark_func(X[i, :]) for i in range(self.pop_size)])
        Gb_Fit, index = np.min(fitness), np.argmin(fitness)
        Gb_Sol = X[index, :]
        Xp = np.copy(X)
        opt = 0
        t = 0

        # Main optimization loop
        while t < self.Tmax and Gb_Fit > opt:
            for i in range(len(X)):
                U1 = np.random.rand(self.dim) > np.random.rand(self.dim)
                rand_index1 = np.random.randint(len(X))
                rand_index2 = np.random.randint(len(X))

                if np.random.rand() < np.random.rand():
                    y = (X[i, :] + X[rand_index1, :]) / 2
                    X[i, :] = X[i, :] + np.random.randn(self.dim) * np.abs(2 * np.random.rand() * Gb_Sol - y)
                else:
                    Yt = 2 * np.random.rand() * (1 - t / self.Tmax) ** (t / self.Tmax)
                    U2 = np.random.rand(self.dim) < 0.5
                    S = np.random.rand() * U2
                    if np.random.rand() < 0.8:
                        St = np.exp(fitness[i] / (np.sum(fitness) + np.finfo(float).eps))
                        S = S * Yt * St
                        X[i, :] = (1 - U1) * X[i, :] + U1 * (
                                X[rand_index1, :] + St * (X[rand_index2, :] - X[rand_index1, :]) - S)
                    else:
                        Mt = np.exp(fitness[i] / (np.sum(fitness) + np.finfo(float).eps))
                        Vtp = X[rand_index1, :]
                        Ft = np.random.rand(self.dim) * (Mt * (-X[i, :] + Vtp))
                        S = S * Yt * Ft
                        X[i, :] = (Gb_Sol + (0.2 * (1 - np.random.rand()) + np.random.rand()) * (
                                U2 * Gb_Sol - X[i, :])) - S

                X[i, :] = np.clip(X[i, :], self.lb, self.ub)
                nF = self.benchmark_func(X[i, :])
                if fitness[i] < nF:
                    X[i, :] = Xp[i, :]
                else:
                    Xp[i, :] = X[i, :]
                    fitness[i] = nF
                    if nF <= Gb_Fit:
                        Gb_Sol = X[i, :]
                        Gb_Fit = nF

            Conv_curve[t] = Gb_Fit
            t += 1

        return Gb_Fit, Gb_Sol, Conv_curve


class Visualization:
    """
    The Visualization class handles the plotting and visualization of optimization results.
    """

    @staticmethod
    def visualize_function(benchmark_func, func_num, lb=-10, ub=10, dim=2):
        """
        Visualize the benchmark function in 3D.

        :param benchmark_func: Function to visualize
        :param func_num: Function number
        :param lb: Lower bound for the axes
        :param ub: Upper bound for the axes
        :param dim: Dimensionality of the problem
        """
        x = np.linspace(lb, ub, 100)
        y = np.linspace(lb, ub, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([benchmark_func(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = Z.reshape(X.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title(f"Function F{func_num} Visualization")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")
        plt.show()

    @staticmethod
    def plot_convergence(Conv_curve, func_num):
        """
        Plot the convergence curve of the optimization.

        :param Conv_curve: The convergence curve data
        :param func_num: Function number
        """
        plt.figure()
        plt.plot(Conv_curve, label=f"F{func_num} Convergence")
        plt.xlabel("Iterations")
        plt.ylabel("Best Fitness")
        plt.title(f"Convergence Curve for Function F{func_num}")
        plt.legend()
        plt.show()


def run_experiments(func_num, dim=10, pop_size=50, Tmax=500, ub=100, lb=-100):
    """
    Run optimization experiments for a specific benchmark function.

    :param func_num: Function number
    :param dim: Dimensionality of the problem
    :param pop_size: Population size
    :param Tmax: Maximum iterations
    :param ub: Upper bound for the search space
    :param lb: Lower bound for the search space
    """
    benchmark_func = BenchmarkFunction.get_function(func_num)
    optimizer = CPOOptimizer(pop_size, Tmax, ub, lb, dim, func_num)
    Gb_Fit, Gb_Sol, Conv_curve = optimizer.optimize()

    # Visualizations
    Visualization.visualize_function(benchmark_func, func_num)
    Visualization.plot_convergence(Conv_curve, func_num)

    print(f"Function F{func_num} Final Best Fitness: {Gb_Fit:.10e}")


def main():
    """
    Main function to run the experiments for all benchmark functions.
    """
    for func_num in range(1, 10):
        print(f"Running Function F{func_num}...")
        run_experiments(func_num)


if __name__ == "__main__":
    main()
