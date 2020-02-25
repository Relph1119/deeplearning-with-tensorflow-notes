#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 3.8-non-linear-nn.py
@time: 2020/2/14 20:23
@desc: 3.8 手写数字图片识别体验的代码
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False

# 加载 MNIST 数据集
(x, y), (x_val, y_val) = datasets.mnist.load_data()
# 转换为浮点张量， 并缩放到-1~1
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
# 转换为整形张量
y = tf.convert_to_tensor(y, dtype=tf.int32)
# one-hot 编码
y = tf.one_hot(y, depth=10)
print(x.shape, y.shape)
# 构建数据集对象
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
# 批量训练
train_dataset = train_dataset.batch(200)

# 利用 Sequential 容器封装 3 个网络层，前网络层的输出默认作为下一层的输入
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])

optimizer = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):
    # Step4.loop
    for step, (x, y) in enumerate(train_dataset):
        # 构建梯度记录环境
        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28 * 28))
            # Step1. 得到模型输出 output [b, 784] => [b, 10]
            out = model(x)
            # Step2. compute loss
            # 计算每个样本的平均误差， [b]
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        # Step3. optimize and update w1, w2, w3, b1, b2, b3
        # 计算参数的梯度 w1, w2, w3, b1, b2, b3
        grads = tape.gradient(loss, model.trainable_variables)
        # w' = w - lr * grad
        # 更新网络参数
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())

    return loss.numpy()


def train():
    losses = []

    for epoch in range(50):
        loss = train_epoch(epoch)
        losses.append(loss)

    x = [i for i in range(0, 50)]
    # 绘制曲线
    plt.plot(x, losses, color='blue', marker='s', label='训练误差')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('MNIST数据集的训练误差曲线.png')
    plt.close()


if __name__ == '__main__':
    train()
