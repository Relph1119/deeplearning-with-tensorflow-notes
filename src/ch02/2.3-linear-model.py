#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 2.3-linear-model.py
@time: 2020/2/14 19:13
@desc: 2.3 线性模型实战的代码
"""

import matplotlib.pyplot as plt
import numpy as np

# Default parameters for plots
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False

# def load_data(seed):
#     np.random.seed(seed)
#     # 保存样本集的列表
#     data = []
#     # 循环采样 100 个点
#     for i in range(100):
#         # 随机采样输入 x
#         x = np.random.uniform(-10., 10.)
#         # 采样高斯噪声
#         eps = np.random.normal(0., 0.01)
#         # 得到模型的输出
#         y = 1.477 * x + 0.089 + eps
#         # 保存样本点
#         data.append([x, y])
#
#     # 转换为 2D Numpy 数组
#     return np.array(data)


def mse(b, w, points):
    # 根据当前的 w,b 参数计算均方差损失
    total_error = 0

    # 循环迭代所有点
    for i in range(0, len(points)):
        # 获得 i 号点的输入 x
        x = points[i, 0]
        # 获得 i 号点的输出 y
        y = points[i, 1]
        # 计算差的平方，并累加
        total_error += (y - (w * x + b)) ** 2

    # 将累加的误差求平均，得到均方差
    return total_error / float(len(points))


def step_gradient(b_current, w_current, points, lr):
    # 计算误差函数在所有点上的导数，并更新 w,b
    b_gradient = 0
    w_gradient = 0
    # 总样本数
    m = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 误差函数对 b 的导数： grad_b = 2(wx+b-y)，参考公式(2.3)
        b_gradient += (2 / m) * ((w_current * x + b_current) - y)
        # 误差函数对 w 的导数： grad_w = 2(wx+b-y)*x，参考公式(2.2)
        w_gradient += (2 / m) * x * ((w_current * x + b_current) - y)

    # 根据梯度下降算法更新 w',b',其中 lr 为学习率
    new_b = b_current - (lr * b_gradient)
    new_w = w_current - (lr * w_gradient)

    return [new_b, new_w]


def gradient_descent(points, starting_b, starting_w, lr, num_iterations):
    losses = []

    # 循环更新 w,b 多次
    # b 的初始值
    b = starting_b
    # w 的初始值
    w = starting_w
    # 根据梯度下降算法更新多次
    for step in range(num_iterations):
        # 计算梯度并更新一次
        b, w = step_gradient(b, w, np.array(points), lr)
        # 计算当前的均方差，用于监控训练进度
        loss = mse(b, w, points)
        losses.append(loss)
        # 打印误差和实时的 w,b 值
        if step % 50 == 0:
            print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}")

    # 返回最后一次的 w,b
    return [b, w], losses


def main():
    # 加载训练集数据，这些数据是通过真实模型添加观测误差采样得到的
    data = np.genfromtxt("data.csv", delimiter=",")
    # 学习率
    lr = 0.0001
    # 初始化 b 为 0
    initial_b = 0
    # 初始化 w 为 0
    initial_w = 0
    num_iterations = 1000
    # 训练优化 1000 次，返回最优 w*,b*和训练 Loss 的下降过程
    [b, w], losses = gradient_descent(data, initial_b, initial_w, lr, num_iterations)
    # 计算最优数值解 w,b 上的均方差
    loss = mse(b, w, data)
    print(f'Final loss:{loss}, w:{w}, b:{b}')

    x = [i for i in range(0, 1000)]
    # 绘制曲线
    plt.plot(x, losses, 'C1')
    plt.plot(x, losses, color='C1', label='均方差')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('模型训练MSE下降曲线.png')
    plt.close()


if __name__ == '__main__':
    main()
