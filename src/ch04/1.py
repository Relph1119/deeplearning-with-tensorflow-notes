#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 1.py
@time: 2020/2/14 23:47
@desc:
"""

import tensorflow as tf

import tensorflow.keras.datasets as datasets
# 加载 MNIST 数据集
(x, y), (x_val, y_val) = datasets.mnist.load_data()

# 每层的张量都需要被优化，故使用 Variable 类型，并使用截断的正太分布初始化权值张量
# 偏置向量初始化为 0 即可
# 第一层的参数
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
# 第二层的参数
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
# 第三层的参数
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

# 改变视图， [b, 28, 28] => [b, 28*28]
x = tf.reshape(x, (-1, 28*28))
# 第一层计算， [b, 784]@[784, 256] + [256] => [b, 256] + [256] => [b,256] + [b, 256]
h1 = x@w1 + tf.broadcast_to(b1, (x.shape[0], 256))
h1 = tf.nn.relu(h1) # 通过激活函数

# 第二层计算， [b, 256] => [b, 128]
h2 = h1@w2 + b2
h2 = tf.nn.relu(h2)
# 输出层计算， [b, 128] => [b, 10]
out = h2@w3 + b3

# 计算网络输出与标签之间的均方差， mse = mean(sum(y-out)^2)
# [b, 10]
loss = tf.square(y_onehot - out)
# 误差标量， mean: scalar
loss = tf.reduce_mean(loss)

# 自动梯度，需要求梯度的张量有[w1, b1, w2, b2, w3, b3]
grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

# 梯度更新， assign_sub 将当前值减去参数值，原地更新
w1.assign_sub(lr * grads[0])
b1.assign_sub(lr * grads[1])
w2.assign_sub(lr * grads[2])
b2.assign_sub(lr * grads[3])
w3.assign_sub(lr * grads[4])
b3.assign_sub(lr * grads[5])