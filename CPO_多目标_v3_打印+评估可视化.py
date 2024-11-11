import numpy as np
import matplotlib.pyplot as plt

# 初始化种群
def initialization(pop_size, dim, ub, lb):
    return np.random.rand(pop_size, dim) * (ub - lb) + lb

# 测试函数定义
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

# 测试函数选择器
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

# CPO 主算法
def CPO(pop_size, Tmax, ub, lb, dim, func_num):
    Gb_Fit = np.inf
    Gb_Sol = None
    Conv_curve = np.zeros(Tmax)
    diversity_history = []
    avg_fitness_history = []
    trajectory_1st_dim = []

    # 初始化种群
    X = initialization(pop_size, dim, ub, lb)
    fitness = np.array([fhd(X[i, :], func_num) for i in range(pop_size)])
    Gb_Fit, index = np.min(fitness), np.argmin(fitness)
    Gb_Sol = X[index, :]
    Xp = np.copy(X)
    opt = 0
    t = 0

    while t < Tmax and Gb_Fit > opt:
        diversity_history.append(np.std(fitness))
        avg_fitness_history.append(np.mean(fitness))
        trajectory_1st_dim.append(Gb_Sol[0])  # 记录第一维度的轨迹

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

    return Gb_Fit, Gb_Sol, Conv_curve, diversity_history, avg_fitness_history, trajectory_1st_dim

# 数据平滑函数
def smooth_data(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 极小值替换处理
def preprocess_data(data):
    data = np.array(data)  # 将数据转换为 numpy 数组
    return np.where(data <= 1e-14, 1e-14, data)

# 可视化函数
def plot_results(Conv_curve, diversity_history, avg_fitness_history, trajectory_1st_dim):
    # 预处理并平滑数据
    safe_trajectory_1st_dim = preprocess_data(trajectory_1st_dim)
    smooth_trajectory_1st_dim = smooth_data(safe_trajectory_1st_dim)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Diversity plot with adaptive log scale
    axs[0].plot(diversity_history, 'r-', label="Diversity")
    axs[0].set_yscale("log")
    axs[0].set_title("Diversity")
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Diversity")

    # Convergence curve with log scale
    axs[1].plot(Conv_curve, 'b-', label="Convergence Curve")
    axs[1].set_yscale("log")
    axs[1].set_title("Convergence Curve")
    axs[1].set_xlabel("Iterations")
    axs[1].set_ylabel("Best Fitness")

    # Average fitness history
    axs[2].plot(avg_fitness_history, 'b--', label="Average fitness history")
    axs[2].set_title("Average Fitness History")
    axs[2].set_xlabel("Iterations")
    axs[2].set_ylabel("Fitness")
    axs[2].set_yscale("log")

    # Trajectory in the first dimension (log scale)
    axs[3].plot(smooth_trajectory_1st_dim, 'k-', label="Trajectory in 1st dimension")
    axs[3].set_yscale("log")
    axs[3].set_title("Trajectory in 1st Dimension")
    axs[3].set_xlabel("Iteration")
    axs[3].set_ylabel("Position (log scale)")

    plt.tight_layout()
    plt.show()

# 运行实验
def run_experiments(func_num=1, dim=10, pop_size=50, Tmax=500, ub=100, lb=-100):
    Gb_Fit, Gb_Sol, Conv_curve, diversity_history, avg_fitness_history, trajectory_1st_dim = CPO(
        pop_size, Tmax, ub, lb, dim, func_num
    )

    # 打印评估指标
    print(f"Function F{func_num} Final Best Fitness: {Gb_Fit:.10e}")
    print(f"Final Solution (Best Position): {Gb_Sol}")
    print(f"Mean Diversity: {np.mean(diversity_history):.2e}")
    print(f"Mean Average Fitness: {np.mean(avg_fitness_history):.2e}")
    print(f"Trajectory in 1st Dimension (final position): {trajectory_1st_dim[-1]:.2e}")

    # 可视化结果
    plot_results(Conv_curve, diversity_history, avg_fitness_history, trajectory_1st_dim)

# 测试每个函数
for func_num in range(1, 10):
    print(f"Running Function F{func_num}...")
    run_experiments(func_num)
