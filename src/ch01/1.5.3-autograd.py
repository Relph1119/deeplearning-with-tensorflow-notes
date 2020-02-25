#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 1.5.3-autograd.py
@time: 2020/2/14 18:02
@desc: 1.5.3 功能演示-自动梯度的代码
"""

import tensorflow as tf

# 创建 4 个张量，并赋值
a = tf.constant(1.)
b = tf.constant(2.)
c = tf.constant(3.)
w = tf.constant(4.)

with tf.GradientTape() as tape:  # 构建梯度环境
    tape.watch([w])  # 将 w 加入梯度跟踪列表
    # 构建计算过程，函数表达式
    y = a * w ** 2 + b * w + c

# 自动求导
[dy_dw] = tape.gradient(y, [w])
print(dy_dw)  # 打印出导数
