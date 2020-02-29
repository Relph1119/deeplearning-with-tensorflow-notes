#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 14.1.5-cartpole-policy.py
@time: 2020/2/29 22:37
@desc: 14.1.5 平衡杆游戏实战的代码（策略网络Policy）
"""

import os

import gym
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

# Default parameters for plots
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers

env = gym.make('CartPole-v1')  # 创建游戏环境
env.seed(2333)
tf.random.set_seed(2333)
np.random.seed(2333)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Policy(keras.Model):
    # 策略网络，生成动作的概率分布
    def __init__(self, learning_rate, gamma):
        super(Policy, self).__init__()
        self.data = []  # 存储轨迹
        # 输入为长度为4的向量，输出为左、右2个动作
        self.fc1 = layers.Dense(128, kernel_initializer='he_normal')
        self.fc2 = layers.Dense(2, kernel_initializer='he_normal')
        # 网络优化器
        self.optimizer = optimizers.Adam(lr=learning_rate)
        self.gamma = gamma

    def call(self, inputs, training=None):
        # 状态输入s的shape为向量：[4]
        x = tf.nn.relu(self.fc1(inputs))
        x = tf.nn.softmax(self.fc2(x), axis=1)
        return x

    def put_data(self, item):
        """
        在交互时，将每个时间戳上的状态输入s[t]，动作分布输出a[t]，
        环境奖励r[t]和新状态s[t+1]作为一个4元组item记录下来.
        """
        # 记录r,log_P(a|s)
        self.data.append(item)

    def train_net(self, tape):
        # 计算梯度并更新策略网络参数。tape为梯度记录器
        # 终结状态的初始回报为0
        R = 0
        # 逆序取轨迹数据
        for r, log_prob in self.data[::-1]:
            # 累加计算每个时间戳上的回报
            R = r + self.gamma * R
            loss = -log_prob * R
            with tape.stop_recording():
                # 优化策略网络
                grads = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # 清空轨迹
        self.data = []


def train(epoch, pi, print_interval, returns, score):
    s = env.reset()  # 回到游戏初始状态，返回s0
    with tf.GradientTape(persistent=True) as tape:
        for t in range(501):  # CartPole-v1 forced to terminates at 500 step.
            # 送入状态向量，获取策略
            s = tf.constant(s, dtype=tf.float32)
            # s: [4] => [1,4]
            s = tf.expand_dims(s, axis=0)
            # 动作分布:[1,2]
            prob = pi(s)
            # 从类别分布中采样1个动作, shape: [1]
            a = tf.random.categorical(tf.math.log(prob), 1)[0]
            # Tensor转数字
            a = int(a)
            s_prime, r, done, info = env.step(a)
            # 记录动作a和动作产生的奖励r
            # prob shape:[1,2]
            pi.put_data((r, tf.math.log(prob[0][a])))
            # 刷新状态
            s = s_prime
            # 累积奖励
            score += r

            if epoch > 1000:
                env.render()

            # 当前episode终止
            if done:
                break

        # episode终止后，训练一次网络
        pi.train_net(tape)
    del tape

    if epoch % print_interval == 0 and epoch != 0:
        returns.append(score / print_interval)
        # 每20次的平均得分
        print(f"# of episode :{epoch}, avg score : {score / print_interval}")
        score = 0.0


def main():
    learning_rate = 0.0002
    gamma = 0.98

    pi = Policy(learning_rate, gamma)  # 创建策略网络
    pi(tf.random.normal((4, 4)))
    pi.summary()
    score = 0.0  # 计分
    print_interval = 20  # 打印间隔
    returns = [0]
    epoch_num = 400

    for epoch in range(1, epoch_num + 1):
        train(epoch, pi, print_interval, returns, score)

    # 关闭环境
    env.close()

    plt.plot(np.arange(len(returns)) * print_interval, returns)
    plt.plot(np.arange(len(returns)) * print_interval, returns, 's')
    plt.xlabel('回合数')
    plt.ylabel('每20次的平均得分')
    plt.savefig('14.1.5-cartpole-policy.svg')


if __name__ == '__main__':
    main()
