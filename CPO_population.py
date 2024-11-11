import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Initialization function
def initialization(pop_size, dim, ub, lb):
    return np.random.rand(pop_size, dim) * (ub - lb) + lb


# Define the nine benchmark functions
def sphere_func(x):
    return np.sum(x ** 2)


def schwefel_222_func(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def powell_sum_func(x):
    return np.sum(np.abs(x) ** (np.arange(len(x)) + 1))


def schwefel_12_func(x):
    return np.sum([np.sum(x[:i + 1]) ** 2 for i in range(len(x))])


def schwefel_221_func(x):
    return np.max(np.abs(x))


def rosenbrock_func(x):
    return np.sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)])


def step_func(x):
    return np.sum((x + 0.5) ** 2)


def quartic_func(x):
    return np.sum(np.arange(1, len(x) + 1) * x ** 4) + np.random.uniform(0, 1)


def zakharov_func(x):
    term1 = np.sum(x ** 2)
    term2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 2
    term3 = np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 4
    return term1 + term2 + term3


# Wrapper function to select test function
def fhd(x, func_num):
    if func_num == 1:
        return sphere_func(x)
    elif func_num == 2:
        return schwefel_222_func(x)
    elif func_num == 3:
        return powell_sum_func(x)
    elif func_num == 4:
        return schwefel_12_func(x)
    elif func_num == 5:
        return schwefel_221_func(x)
    elif func_num == 6:
        return rosenbrock_func(x)
    elif func_num == 7:
        return step_func(x)
    elif func_num == 8:
        return quartic_func(x)
    elif func_num == 9:
        return zakharov_func(x)
    else:
        raise ValueError("Invalid function number")


# Main CPO algorithm with additional population history
def CPO(pop_size, Tmax, ub, lb, dim, func_num):
    Gb_Fit = np.inf
    Conv_curve = np.zeros(Tmax)
    diversity_history = []
    avg_fitness_history = []
    population_history = []
    best_positions = []

    # Initialize population
    X = initialization(pop_size, dim, ub, lb)
    fitness = np.array([fhd(X[i, :], func_num) for i in range(pop_size)])
    Gb_Fit = np.min(fitness)
    Gb_Sol = X[np.argmin(fitness)]
    Conv_curve[0] = Gb_Fit
    diversity_history.append(np.std(fitness))
    avg_fitness_history.append(np.mean(fitness))
    population_history.append(np.copy(X))
    best_positions.append(np.copy(Gb_Sol))

    # Optimization loop
    for t in range(1, Tmax):
        for i in range(pop_size):
            if np.random.rand() < 0.5:
                X[i, :] += np.random.randn(dim) * (Gb_Fit - X[i, :])
            else:
                X[i, :] -= np.random.randn(dim) * (Gb_Fit + X[i, :])
            X[i, :] = np.clip(X[i, :], lb, ub)
            fitness[i] = fhd(X[i, :], func_num)

        # Update global best fitness
        min_fitness = np.min(fitness)
        if min_fitness < Gb_Fit:
            Gb_Fit = min_fitness
            Gb_Sol = X[np.argmin(fitness)]

        Conv_curve[t] = Gb_Fit
        diversity_history.append(np.std(fitness))
        avg_fitness_history.append(np.mean(fitness))
        population_history.append(np.copy(X))
        best_positions.append(np.copy(Gb_Sol))

    return Gb_Fit, Conv_curve, diversity_history, avg_fitness_history, population_history, best_positions


# Visualization function for search history in 3D
def visualize_search_history_dynamic_3D(population_history, best_positions, func_num, ub, lb):
    # Set up 3D plot of the function surface
    x = np.linspace(lb, ub, 100)
    y = np.linspace(lb, ub, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([fhd(np.array([x, y]), func_num) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)

    # Only show the specified iterations
    selected_iterations = [0, 20, 50, 75,150, 499]
    fig = plt.figure(figsize=(20, 5))

    for idx, iteration in enumerate(selected_iterations):
        # Ensure iteration is within bounds
        if iteration >= len(population_history):
            continue

        ax = fig.add_subplot(1, len(selected_iterations), idx + 1, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

        # Plot population points for current iteration
        pop = population_history[iteration]
        pop_z = np.array([fhd(ind, func_num) for ind in pop])
        ax.scatter(pop[:, 0], pop[:, 1], pop_z, color='blue', s=10, label="Population")

        # Plot the best position
        best_pos = best_positions[iteration]
        best_z = fhd(best_pos, func_num)
        ax.scatter(best_pos[0], best_pos[1], best_z, color='red', s=50, marker='x', label="Best Position")

        ax.set_title(f"Iteration {iteration}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("Fitness")
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Main function to run experiments and plot visualizations
def run_experiments(func_num=1, dim=2, pop_size=200, Tmax=500, ub=100, lb=-100):
    Gb_Fit, Conv_curve, diversity_history, avg_fitness_history, population_history, best_positions = CPO(
        pop_size, Tmax, ub, lb, dim, func_num
    )
    print(f"Function F{func_num} Final Best Fitness: {Gb_Fit:.10e}")
    visualize_search_history_dynamic_3D(population_history, best_positions, func_num, ub, lb)


# Run experiments for a specific function
run_experiments(func_num=1)

# Run experiments for each of the nine functions
for func_num in range(1, 10):
    print(f"Running Function F{func_num}...")
    run_experiments(func_num)
