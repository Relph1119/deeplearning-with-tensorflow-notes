#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 14.3.6-cartpole-ppo.py
@time: 2020/2/29 23:01
@desc: 14.3.6 PPO实战的代码
"""

import os
from collections import namedtuple

import gym
import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses

matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

env = gym.make('CartPole-v1')  # 创建游戏环境
env.seed(2222)
tf.random.set_seed(2222)
np.random.seed(2222)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建游戏环境
env = gym.make('CartPole-v0').unwrapped
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class Actor(keras.Model):
    def __init__(self):
        super(Actor, self).__init__()
        # 策略网络，也叫Actor网络，输出为概率分布pi(a|s)
        self.fc1 = layers.Dense(100, kernel_initializer='he_normal')
        self.fc2 = layers.Dense(2, kernel_initializer='he_normal')

    def call(self, inputs):
        x = tf.nn.relu(self.fc1(inputs))
        x = self.fc2(x)
        x = tf.nn.softmax(x, axis=1)  # 转换成概率
        return x


class Critic(keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        # 偏置b的估值网络，也叫Critic网络，输出为v(s)
        self.fc1 = layers.Dense(100, kernel_initializer='he_normal')
        self.fc2 = layers.Dense(1, kernel_initializer='he_normal')

    def call(self, inputs):
        x = tf.nn.relu(self.fc1(inputs))
        # 输出基准线 b 的估计
        x = self.fc2(x)
        return x


class PPO():
    # PPO算法主体
    def __init__(self, gamma, batch_size, epsilon):
        super(PPO, self).__init__()
        # 创建Actor网络
        self.actor = Actor()
        # 创建Critic网络
        self.critic = Critic()
        # 数据缓冲池
        self.buffer = []
        # Actor优化器
        self.actor_optimizer = optimizers.Adam(1e-3)
        # Critic优化器
        self.critic_optimizer = optimizers.Adam(3e-3)
        self._gamma = gamma
        self._batch_size = batch_size
        self._epsilon = epsilon

    def select_action(self, s):
        # 送入状态向量，获取策略: [4]
        s = tf.constant(s, dtype=tf.float32)
        # s: [4] => [1,4]
        s = tf.expand_dims(s, axis=0)
        # 获取策略分布: [1, 2]
        prob = self.actor(s)
        # 从类别分布中采样1个动作, shape: [1]
        a = tf.random.categorical(tf.math.log(prob), 1)[0]
        # Tensor转数字
        a = int(a)
        # 返回动作及其概率
        return a, float(prob[0][a])

    def get_value(self, s):
        # 送入状态向量，获取策略: [4]
        s = tf.constant(s, dtype=tf.float32)
        # s: [4] => [1,4]
        s = tf.expand_dims(s, axis=0)
        # 获取策略分布: [1, 2]
        v = self.critic(s)[0]
        return float(v)  # 返回v(s)

    def store_transition(self, transition):
        # 存储采样数据
        self.buffer.append(transition)

    def optimize(self):
        # 优化网络主函数
        # 从缓存中取出样本数据，转换成Tensor
        state = tf.constant([t.state for t in self.buffer], dtype=tf.float32)
        action = tf.constant([t.action for t in self.buffer], dtype=tf.int32)
        action = tf.reshape(action, [-1, 1])
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = tf.constant([t.a_log_prob for t in self.buffer], dtype=tf.float32)
        old_action_log_prob = tf.reshape(old_action_log_prob, [-1, 1])
        # 通过MC方法循环计算R(st)
        R = 0
        Rs = []
        for r in reward[::-1]:
            R = r + self._gamma * R
            Rs.insert(0, R)
        Rs = tf.constant(Rs, dtype=tf.float32)
        # 对缓冲池数据大致迭代10遍
        for _ in range(round(10 * len(self.buffer) / self._batch_size)):
            # 随机从缓冲池采样batch size大小样本
            index = np.random.choice(np.arange(len(self.buffer)), self._batch_size, replace=False)
            # 构建梯度跟踪环境
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                # 取出R(st)，[b,1]
                v_target = tf.expand_dims(tf.gather(Rs, index, axis=0), axis=1)
                # 计算v(s)预测值，也就是偏置b，我们后面会介绍为什么写成v
                v = self.critic(tf.gather(state, index, axis=0))
                delta = v_target - v  # 计算优势值
                advantage = tf.stop_gradient(delta)  # 断开梯度连接
                # 由于TF的gather_nd与pytorch的gather功能不一样，需要构造
                # gather_nd需要的坐标参数，indices:[b, 2]
                # pi_a = pi.gather(1, a) # pytorch只需要一行即可实现
                a = tf.gather(action, index, axis=0)  # 取出batch的动作at
                # batch的动作分布pi(a|st)
                pi = self.actor(tf.gather(state, index, axis=0))
                indices = tf.expand_dims(tf.range(a.shape[0]), axis=1)
                indices = tf.concat([indices, a], axis=1)
                pi_a = tf.gather_nd(pi, indices)  # 动作的概率值pi(at|st), [b]
                pi_a = tf.expand_dims(pi_a, axis=1)  # [b]=> [b,1]
                # 重要性采样
                ratio = (pi_a / tf.gather(old_action_log_prob, index, axis=0))
                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1 - self._epsilon, 1 + self._epsilon) * advantage
                # PPO误差函数
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                # 对于偏置v来说，希望与MC估计的R(st)越接近越好
                value_loss = losses.MSE(v_target, v)
            # 优化策略网络
            grads = tape1.gradient(policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
            # 优化偏置值网络
            grads = tape2.gradient(value_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        self.buffer = []  # 清空已训练数据


def train(agent, batch_size, epoch, returns, total, print_interval):
    # 复位环境
    state = env.reset()
    # 最多考虑500步
    for t in range(500):
        # 通过最新策略与环境交互
        action, action_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        # 构建样本并存储
        trans = Transition(state, action, action_prob, reward, next_state)
        agent.store_transition(trans)
        # 刷新状态
        state = next_state
        # 累积奖励
        total += reward
        # 合适的时间点训练网络
        if done:
            if len(agent.buffer) >= batch_size:
                # 训练网络
                agent.optimize()
            break

    # 每20个回合统计一次平均得分
    if epoch % print_interval == 0:
        returns.append(total / print_interval)
        print(f"# of episode :{epoch}, avg score : {total / print_interval}")
        total = 0


def main():
    gamma = 0.98  # 激励衰减因子
    epsilon = 0.2  # PPO误差超参数0.8~1.2
    batch_size = 32  # batch size
    epoch_num = 500
    print_interval = 20

    agent = PPO(gamma, batch_size, epsilon)
    # 统计总回报
    returns = [0]
    total = 0  # 一段时间内平均回报
    for epoch in range(1, epoch_num + 1):  # 训练回合数
        train(agent, batch_size, epoch, returns, total, print_interval)

    print(np.array(returns))
    plt.plot(np.arange(len(returns)) * 20, np.array(returns))
    plt.plot(np.arange(len(returns)) * 20, np.array(returns), 's')
    plt.xlabel('回合数')
    plt.ylabel('每20次的平均得分')
    plt.savefig('14.3.6-cartpole-ppo.svg')


if __name__ == '__main__':
    main()
    print("end")
