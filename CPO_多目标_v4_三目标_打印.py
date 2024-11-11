import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Initialization function
def initialization(pop_size, dim, ub, lb):
    return np.random.rand(pop_size, dim) * (ub - lb) + lb

# Define benchmark functions
def sphere_func(x):
    return np.sum(x ** 2)

def schwefel_222_func(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def quartic_func(x):
    return np.sum((np.arange(1, len(x) + 1) * x ** 4)) + np.random.uniform(0, 1)

def rosenbrock_func(x):
    return np.sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)])

def zakharov_func(x):
    term1 = np.sum(x ** 2)
    term2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 2
    term3 = np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 4
    return term1 + term2 + term3

def step_func(x):
    return np.sum((x + 0.5) ** 2)

# Define multi-objective combinations
def fhd(x, func_num):
    if func_num == 1:
        return np.array([sphere_func(x), schwefel_222_func(x), quartic_func(x)])
    elif func_num == 2:
        return np.array([rosenbrock_func(x), zakharov_func(x), step_func(x)])
    else:
        raise ValueError("Function number not supported")

# CPO Algorithm for Multi-objective optimization with added data output
def CPO(pop_size, Tmax, ub, lb, dim, func_num, key_generations):
    X = initialization(pop_size, dim, ub, lb)
    fitness = np.array([fhd(X[i, :], func_num) for i in range(pop_size)])
    Gb_Fit = np.min(fitness, axis=0)
    Gb_Sol = X[np.argmin(fitness[:, 0]), :]
    Xp = np.copy(X)

    Pareto_fronts = []
    stats = []  # Store statistics for selected generations

    for t in range(Tmax):
        # Collect statistics only for selected generations (initial, mid, final)
        if t in key_generations:
            avg_fitness = np.mean(fitness, axis=0)
            min_fitness = np.min(fitness, axis=0)
            stats.append((t, min_fitness, avg_fitness))  # Record generation, min, and average fitness

        for i in range(len(X)):
            U1 = np.random.rand(dim) > np.random.rand(dim)
            rand_index1, rand_index2 = np.random.randint(len(X)), np.random.randint(len(X))

            if np.random.rand() < np.random.rand():
                y = (X[i, :] + X[rand_index1, :]) / 2
                X[i, :] = X[i, :] + np.random.randn(dim) * np.abs(2 * np.random.rand() * Gb_Sol - y)
            else:
                Yt = 2 * np.random.rand() * (1 - t / Tmax) ** (t / Tmax)
                U2 = np.random.rand(dim) < 0.5
                S = np.random.rand() * U2
                if np.random.rand() < 0.8:
                    St = np.exp(fitness[i].sum() / (fitness.sum() + np.finfo(float).eps))
                    S = S * Yt * St
                    X[i, :] = (1 - U1) * X[i, :] + U1 * (
                                X[rand_index1, :] + St * (X[rand_index2, :] - X[rand_index1, :]) - S)
                else:
                    Mt = np.exp(fitness[i].sum() / (fitness.sum() + np.finfo(float).eps))
                    Vtp = X[rand_index1, :]
                    Ft = np.random.rand(dim) * (Mt * (-X[i, :] + Vtp))
                    S = S * Yt * Ft
                    X[i, :] = (Gb_Sol + (0.2 * (1 - np.random.rand()) + np.random.rand()) * (U2 * Gb_Sol - X[i, :])) - S

            X[i, :] = np.clip(X[i, :], lb, ub)
            nF = fhd(X[i, :], func_num)
            if np.all(fitness[i] < nF):
                X[i, :] = Xp[i, :]
            else:
                Xp[i, :] = X[i, :]
                fitness[i] = nF
                if np.all(nF <= Gb_Fit):
                    Gb_Sol = X[i, :]
                    Gb_Fit = nF

        Pareto_front = np.array([fitness[i] for i in range(len(fitness)) if np.all(fitness[i] <= Gb_Fit)])
        if Pareto_front.size > 0:
            Pareto_fronts.append(Pareto_front)

    # Print selected statistics
    print("Generation | Min Fitness (F1, F2, F3) | Avg Fitness (F1, F2, F3)")
    for stat in stats:
        generation, min_fit, avg_fit = stat
        print(f"{generation:10d} | {min_fit} | {avg_fit}")

    return Gb_Fit, Gb_Sol, Pareto_fronts

# 3D surface interpolation and visualization of the Pareto front
def plot_pareto_front_surface(pareto_fronts, func_name):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    epsilon = 1e-10  # Small shift to avoid log(0)

    # Gather all points from Pareto fronts
    all_points = np.vstack(pareto_fronts)
    x = all_points[:, 0] + epsilon
    y = all_points[:, 1] + epsilon
    z = all_points[:, 2] + epsilon

    # Generate a grid and interpolate to create a smooth surface
    grid_x, grid_y = np.meshgrid(
        np.linspace(np.min(x), np.max(x), 100),
        np.linspace(np.min(y), np.max(y), 100)
    )
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # Plot the surface and points
    ax.plot_surface(grid_x, grid_y, grid_z, color='cyan', alpha=0.6, edgecolor='w', rstride=1, cstride=1)
    ax.scatter(x, y, z, color='r', s=20)  # Increase `s` for larger points

    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    ax.set_zlabel("Objective 3")
    plt.title(f"Pareto Front Surface for {func_name} (3 Objectives)")
    plt.show()

# Run experiments for each function combination and visualize results
def run_experiments(func_num, func_name, dim=10, pop_size=30, Tmax=500, ub=100, lb=-100):
    # Define key generations for printing statistics (initial, mid, final)
    key_generations = [0, Tmax // 10, Tmax // 5,3*Tmax // 10,4*Tmax // 10,5*Tmax // 10,6*Tmax // 10,7*Tmax // 10,8*Tmax // 10,9*Tmax // 10,Tmax - 1]
    Gb_Fit, Gb_Sol, Pareto_fronts = CPO(pop_size, Tmax, ub, lb, dim, func_num, key_generations)
    plot_pareto_front_surface(Pareto_fronts, func_name)
    print(f"{func_name}: Final Best Fitness: {Gb_Fit}")

# Function names for labeling
function_names = [
    "Sphere, Schwefel's 2.22 & Quartic",
    "Rosenbrock, Zakharov & Step"
]

# Test the two three-objective function combinations
for func_num, func_name in enumerate(function_names, start=1):
    print(f"Running for function set {func_name}...")
    run_experiments(func_num, func_name)
