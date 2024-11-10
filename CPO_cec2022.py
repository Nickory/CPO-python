import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cec2017.functions import all_functions
import cec2022_sobo.CEC2022 as cec
import csv

# 创建输出文件夹
output_folder = "CPO_results_cec2022"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

functionSET = {'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F13', 'F14', 'F15',
               'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28', 'F29'}
fun_fitness = {}


# 初始化函数
def initialization(pop_size, dim, ub, lb):
    return np.random.rand(pop_size, dim) * (ub - lb) + lb


# 计算目标函数值
def fhd(x, func_num):
    fun = cec.cec2022_func(func_num)
    return fun.values(x)


# CPO主算法
def CPO(pop_size, Tmax, ub, lb, dim, func_num):
    Gb_Fit = np.inf
    Gb_Sol = None
    Conv_curve = np.zeros(Tmax)
    X = initialization(dim, pop_size, ub, lb)
    fit = fhd(X, func_num)
    X = X.T
    fitness = fit.ObjFunc
    Gb_Fit, index = np.min(fitness), np.argmin(fitness)
    Gb_Sol = X[index, :]
    Xp = np.copy(X)
    opt = 0
    t = 0

    # 主优化循环
    while t < Tmax and Gb_Fit > opt:
        for i in range(len(X)):
            U1 = np.random.rand(dim) > 0.5
            rand_index1 = np.random.randint(len(X))
            rand_index2 = np.random.randint(len(X))
            if np.random.rand() < 0.5:
                y = (X[i, :] + X[rand_index1, :]) / 2
                X[i, :] = X[i, :] + np.random.randn(dim) * np.abs(2 * np.random.rand() * Gb_Sol - y)
            else:
                Yt = 2 * np.random.rand() * (1 - t / Tmax) ** (t / Tmax + 1e-6)
                U2 = np.random.rand(dim) < 0.5
                S = np.random.rand() * U2
                if np.random.rand() < 0.8:
                    St = np.exp(-np.abs(fitness[i] / (np.sum(fitness) + np.finfo(float).eps)))
                    S = S * Yt * St
                    X[i, :] = (1 - U1) * X[i, :] + U1 * (
                            X[rand_index1, :] + St * (X[rand_index2, :] - X[rand_index1, :]) - S)
                else:
                    Mt = np.exp(-np.abs(fitness[i] / (np.sum(fitness) + np.finfo(float).eps)))
                    Vtp = X[rand_index1, :]
                    Ft = np.random.rand(dim) * (Mt * (-X[i, :] + Vtp))
                    S = S * Yt * Ft
                    X[i, :] = (Gb_Sol + (0.2 * (1 - np.random.rand()) + np.random.rand()) * (U2 * Gb_Sol - X[i, :])) - S
            X[i, :] = np.clip(X[i, :], lb, ub)
            nF = fhd(X[np.newaxis, i, :].T, func_num).ObjFunc
            if fitness[i] < nF:
                X[i, :] = Xp[i, :]
            else:
                Xp[i, :] = X[i, :]
                fitness[i] = nF.item()
                if nF <= Gb_Fit:
                    Gb_Sol = X[i, :]
                    Gb_Fit = nF.item()
        Conv_curve[t] = Gb_Fit
        t += 1

    return Gb_Fit, Gb_Sol, Conv_curve


# 运行实验并保存收敛曲线
def run_experiments(func_num, dim=10, pop_size=130, Tmax=10000, ub=100, lb=-100):
    Gb_Fit, Gb_Sol, Conv_curve = CPO(pop_size, Tmax, ub, lb, dim, func_num)

    # 保存收敛曲线图
    plt.figure()
    plt.plot(Conv_curve, label=f"F{func_num} Convergence")
    plt.xlabel("Iterations")
    plt.ylabel("Best Fitness")
    plt.title(f"Convergence Curve for Function F{func_num}")
    plt.legend()
    plt.savefig(f"{output_folder}/Convergence_F{func_num}.png")
    plt.close()

    # 保存最终的最优适应度值
    fun_fitness[func_num] = Gb_Fit


# 主测试循环
for func_num in range(1, 13):  # 测试所有F1到F30函数
    print(f"Running Function F{func_num}...")
    run_experiments(func_num)
    print(f"Function F{func_num} Final Best Fitness: {fun_fitness[func_num]:.10e}")

# 保存所有函数的最终适应度值到CSV文件
with open(f"{output_folder}/final_fitness_results.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Function", "Best Fitness"])
    for func_num, fitness in fun_fitness.items():
        writer.writerow([f"F{func_num}", fitness])

print("所有结果已保存到文件夹内。")
