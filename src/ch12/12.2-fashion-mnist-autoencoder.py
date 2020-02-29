#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 12.2-fashion-mnist-autoencoder.py
@time: 2020/2/29 16:38
@desc: 12.2 Fashion MNIST 图片重建实战的代码
"""

import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import Sequential, layers

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def save_images(imgs, name):
    # 创建 280x280 大小图片阵列
    new_im = Image.new('L', (280, 280))

    index = 0
    # 10 行图片阵列
    for i in range(0, 280, 28):
        # 10 列图片阵列
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            # 写入对应位置
            new_im.paste(im, (i, j))
            index += 1
    # 保存图片阵列
    new_im.save(name)


def load_dataset(batchsz):
    # 加载 Fashion MNIST 图片数据集
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    # 归一化
    x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
    # 只需要通过图片数据即可构建数据集对象，不需要标签
    train_db = tf.data.Dataset.from_tensor_slices(x_train)
    train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
    # 构建测试集对象
    test_db = tf.data.Dataset.from_tensor_slices(x_test)
    test_db = test_db.batch(batchsz)
    return train_db, test_db


class AE(keras.Model):
    # 自编码器模型类，包含了 Encoder 和 Decoder2 个子网络
    def __init__(self, h_dim):
        super(AE, self).__init__()

        # 创建 Encoders 网络，实现在自编码器类的初始化函数中
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])

        # 创建 Decoders 网络
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])

    def call(self, inputs, training=None):
        # 前向传播函数
        # 编码获得隐藏向量 h,[b, 784] => [b, 20]
        h = self.encoder(inputs)
        # 解码获得重建图片， [b, 20] => [b, 784]
        x_hat = self.decoder(h)

        return x_hat


def build_model(h_dim):
    # 创建网络对象
    model = AE(h_dim)
    # 指定输入大小
    model.build(input_shape=(None, 784))
    # 打印网络信息
    model.summary()
    return model


def train(train_db, model, optimizer, epoch):
    # 遍历训练集
    for step, x in enumerate(train_db):

        # [b, 28, 28] => [b, 784]
        # 打平， [b, 28, 28] => [b, 784]
        x = tf.reshape(x, [-1, 784])
        # 构建梯度记录器
        with tf.GradientTape() as tape:
            # 前向计算获得重建的图片
            x_rec_logits = model(x)
            # 计算重建图片与输入之间的损失函数
            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            # 计算均值
            rec_loss = tf.reduce_mean(rec_loss)
        # 自动求导，包含了2个子网络的梯度
        grads = tape.gradient(rec_loss, model.trainable_variables)
        # 自动更新，同时更新2个子网络
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            # 间隔性打印训练误差
            print(epoch, step, float(rec_loss))

    return model


def evaluation(test_db, model, epoch):
    # evaluation
    # 重建图片，从测试集采样一批图片
    x = next(iter(test_db))
    # 打平并送入自编码器
    logits = model(tf.reshape(x, [-1, 784]))
    # 将输出转换为像素值，使用sigmoid函数
    x_hat = tf.sigmoid(logits)
    # 恢复为 28x28,[b, 784] => [b, 28, 28]
    x_hat = tf.reshape(x_hat, [-1, 28, 28])

    # 输入的前50张+重建的前50张图片合并， [b, 28, 28] => [2b, 28, 28]
    x_concat = tf.concat([x[:50], x_hat[:50]], axis=0)
    # 恢复为0~255范围
    x_concat = x_concat.numpy() * 255.
    # 转换为整型
    x_concat = x_concat.astype(np.uint8)
    # 保存图片
    save_images(x_concat, './ae_images/rec_epoch_%d.png' % epoch)


def main():
    h_dim = 20
    batchsz = 512
    lr = 1e-3

    train_db, test_db = load_dataset(batchsz)
    model = build_model(h_dim)
    # 创建优化器，并设置学习率
    optimizer = tf.optimizers.Adam(lr=lr)
    # 训练100个Epoch
    for epoch in range(100):
        model = train(train_db, model, optimizer, epoch)
        evaluation(test_db, model, epoch)


if __name__ == '__main__':
    main()
