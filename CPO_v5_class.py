import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class EvolutionaryComputationAlgorithm:
    def __init__(self, pop_size, Tmax, dim, ub, lb, func_num):
        self.pop_size = pop_size
        self.Tmax = Tmax
        self.dim = dim
        self.ub = ub
        self.lb = lb
        self.func_num = func_num
        self.Gb_Fit = np.inf
        self.Conv_curve = np.zeros(Tmax)
        self.diversity_history = []
        self.avg_fitness_history = []
        self.population_history = []
        self.best_positions = []
        self.X = self._initialization()
        self.fitness = self._evaluate_fitness(self.X)
        self.Gb_Fit = np.min(self.fitness)
        self.Gb_Sol = self.X[np.argmin(self.fitness)]
        self.Conv_curve[0] = self.Gb_Fit
        self.diversity_history.append(np.std(self.fitness))
        self.avg_fitness_history.append(np.mean(self.fitness))
        self.population_history.append(np.copy(self.X))
        self.best_positions.append(np.copy(self.Gb_Sol))

    def _initialization(self):
        """Initialize population"""
        return np.random.rand(self.pop_size, self.dim) * (self.ub - self.lb) + self.lb

    def _evaluate_fitness(self, X):
        """Evaluate the fitness of the population"""
        return np.array([self._fhd(X[i, :]) for i in range(self.pop_size)])

    def _fhd(self, x):
        """Wrapper function to select test function"""
        if self.func_num == 1:
            return self.sphere_func(x)
        elif self.func_num == 2:
            return self.schwefel_222_func(x)
        elif self.func_num == 3:
            return self.powell_sum_func(x)
        elif self.func_num == 4:
            return self.schwefel_12_func(x)
        elif self.func_num == 5:
            return self.schwefel_221_func(x)
        elif self.func_num == 6:
            return self.rosenbrock_func(x)
        elif self.func_num == 7:
            return self.step_func(x)
        elif self.func_num == 8:
            return self.quartic_func(x)
        elif self.func_num == 9:
            return self.zakharov_func(x)
        else:
            raise ValueError("Invalid function number")

    def sphere_func(self, x):
        return np.sum(x ** 2)

    def schwefel_222_func(self, x):
        return np.sum(np.abs(x)) + np.prod(np.abs(x))

    def powell_sum_func(self, x):
        return np.sum(np.abs(x) ** (np.arange(len(x)) + 1))

    def schwefel_12_func(self, x):
        return np.sum([np.sum(x[:i + 1]) ** 2 for i in range(len(x))])

    def schwefel_221_func(self, x):
        return np.max(np.abs(x))

    def rosenbrock_func(self, x):
        return np.sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)])

    def step_func(self, x):
        return np.sum((x + 0.5) ** 2)

    def quartic_func(self, x):
        return np.sum(np.arange(1, len(x) + 1) * x ** 4) + np.random.uniform(0, 1)

    def zakharov_func(self, x):
        term1 = np.sum(x ** 2)
        term2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 2
        term3 = np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 4
        return term1 + term2 + term3

    def optimize(self):
        """Main optimization loop"""
        for t in range(1, self.Tmax):
            for i in range(self.pop_size):
                if np.random.rand() < 0.5:
                    self.X[i, :] += np.random.randn(self.dim) * (self.Gb_Fit - self.X[i, :])
                else:
                    self.X[i, :] -= np.random.randn(self.dim) * (self.Gb_Fit + self.X[i, :])
                self.X[i, :] = np.clip(self.X[i, :], self.lb, self.ub)
                self.fitness[i] = self._fhd(self.X[i, :])

            # Update global best fitness
            min_fitness = np.min(self.fitness)
            if min_fitness < self.Gb_Fit:
                self.Gb_Fit = min_fitness
                self.Gb_Sol = self.X[np.argmin(self.fitness)]

            self.Conv_curve[t] = self.Gb_Fit
            self.diversity_history.append(np.std(self.fitness))
            self.avg_fitness_history.append(np.mean(self.fitness))
            self.population_history.append(np.copy(self.X))
            self.best_positions.append(np.copy(self.Gb_Sol))

    def visualize_search_history_dynamic_3D(self):
        """Visualization of search history in 3D"""
        x = np.linspace(self.lb, self.ub, 100)
        y = np.linspace(self.lb, self.ub, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([self._fhd(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = Z.reshape(X.shape)

        selected_iterations = [0, 20, 50, 75, 150, 499]
        fig = plt.figure(figsize=(20, 5))

        for idx, iteration in enumerate(selected_iterations):
            if iteration >= len(self.population_history):
                continue

            ax = fig.add_subplot(1, len(selected_iterations), idx + 1, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

            pop = self.population_history[iteration]
            pop_z = np.array([self._fhd(ind) for ind in pop])
            ax.scatter(pop[:, 0], pop[:, 1], pop_z, color='blue', s=10, label="Population")

            best_pos = self.best_positions[iteration]
            best_z = self._fhd(best_pos)
            ax.scatter(best_pos[0], best_pos[1], best_z, color='red', s=50, marker='x', label="Best Position")

            ax.set_title(f"Iteration {iteration}")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("Fitness")
            ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def run_experiments(func_num=1, dim=2, pop_size=200, Tmax=500, ub=100, lb=-100):
    algo = EvolutionaryComputationAlgorithm(pop_size, Tmax, dim, ub, lb, func_num)
    algo.optimize()
    print(f"Function F{func_num} Final Best Fitness: {algo.Gb_Fit:.10e}")
    algo.visualize_search_history_dynamic_3D()

# Run experiments for a specific function
run_experiments(func_num=1)

# Run experiments for each of the nine functions
for func_num in range(1, 10):
    print(f"Running Function F{func_num}...")
    run_experiments(func_num)
