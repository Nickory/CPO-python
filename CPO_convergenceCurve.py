import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cec2017.functions import all_functions

# Initialization function
def initialization(pop_size, dim, ub, lb):
    return np.random.rand(pop_size, dim) * (ub - lb) + lb


# Benchmark functions from PDF
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
    return np.sum((np.arange(1, len(x) + 1) * x ** 4)) + np.random.uniform(0, 1)


def zakharov_func(x):
    term1 = np.sum(x ** 2)
    term2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 2
    term3 = np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 4
    return term1 + term2 + term3


# Wrapper for test functions
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








# CPO main algorithm
def CPO(pop_size, Tmax, ub, lb, dim, func_num):
    Gb_Fit = np.inf
    Gb_Sol = None
    Conv_curve = np.zeros(Tmax)
    X = initialization(pop_size, dim, ub, lb)
    fitness = np.array([fhd(X[i, :], func_num) for i in range(pop_size)])
    Gb_Fit, index = np.min(fitness), np.argmin(fitness)
    Gb_Sol = X[index, :]
    Xp = np.copy(X)
    opt = 0
    t = 0

    while t < Tmax and Gb_Fit > opt:
        for i in range(len(X)):
            U1 = np.random.rand(dim) > np.random.rand(dim)
            rand_index1 = np.random.randint(len(X))
            rand_index2 = np.random.randint(len(X))

            if np.random.rand() < np.random.rand():
                y = (X[i, :] + X[rand_index1, :]) / 2
                X[i, :] = X[i, :] + np.random.randn(dim) * np.abs(2 * np.random.rand() * Gb_Sol - y)
            else:
                Yt = 2 * np.random.rand() * (1 - t / Tmax) ** (t / Tmax)
                U2 = np.random.rand(dim) < 0.5
                S = np.random.rand() * U2
                if np.random.rand() < 0.8:
                    St = np.exp(fitness[i] / (np.sum(fitness) + np.finfo(float).eps))
                    S = S * Yt * St
                    X[i, :] = (1 - U1) * X[i, :] + U1 * (
                                X[rand_index1, :] + St * (X[rand_index2, :] - X[rand_index1, :]) - S)
                else:
                    Mt = np.exp(fitness[i] / (np.sum(fitness) + np.finfo(float).eps))
                    Vtp = X[rand_index1, :]
                    Ft = np.random.rand(dim) * (Mt * (-X[i, :] + Vtp))
                    S = S * Yt * Ft
                    X[i, :] = (Gb_Sol + (0.2 * (1 - np.random.rand()) + np.random.rand()) * (U2 * Gb_Sol - X[i, :])) - S

            X[i, :] = np.clip(X[i, :], lb, ub)
            nF = fhd(X[i, :], func_num)
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


# Visualize function in 3D
def visualize_function(func, func_num, lb=-10, ub=10, dim=2):
    x = np.linspace(lb, ub, 100)
    y = np.linspace(lb, ub, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([func(np.array([x, y]), func_num) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title(f"Function F{func_num} Visualization")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    plt.show()


# Run experiments and visualize convergence
def run_experiments(func_num, dim=10, pop_size=50, Tmax=500, ub=100, lb=-100):
    Gb_Fit, Gb_Sol, Conv_curve = CPO(pop_size, Tmax, ub, lb, dim, func_num)

    # Plot convergence curve
    plt.figure()
    plt.plot(Conv_curve, label=f"F{func_num} Convergence")
    plt.xlabel("Iterations")
    plt.ylabel("Best Fitness")
    plt.title(f"Convergence Curve for Function F{func_num}")
    plt.legend()
    plt.show()

    print(f"Function F{func_num} Final Best Fitness: {Gb_Fit:.10e}")


# Main testing loop
for func_num in range(1, 10):
    print(f"Running Function F{func_num}...")
    visualize_function(fhd, func_num)  # Visualize function in 3D
    run_experiments(func_num)  # Run CPO and plot convergence curve
