#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 14.1.2-cartpole-v1.py
@time: 2020/2/29 22:20
@desc: 14.1.2 Gym平台-平衡杆游戏的交互代码
       安装gym：
       1. 将gym文件夹放入venv/Lib/site-packages目录下
       2. cd venv/Lib/site-packages/gym
       3. 执行命令pip install -e .
       4. 检查gym包是否已经安装好
"""

# 导入 gym 游戏平台
import gym

# 创建平衡杆游戏环境
env = gym.make("CartPole-v1")
# 复位游戏，回到初始状态
observation = env.reset()
# 循环交互 1000 次
for _ in range(1000):
    # 显示当前时间戳的游戏画面
    env.render()
    # 随机生成一个动作
    action = env.action_space.sample()
    # 与环境交互，返回新的状态，奖励，是否结束标志，其他信息
    observation, reward, done, info = env.step(action)
    # 游戏回合结束，复位状态
    if done:
        observation = env.reset()

# 销毁游戏环境
env.close()
