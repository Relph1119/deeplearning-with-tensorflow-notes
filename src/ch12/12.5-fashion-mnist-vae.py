#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 12.5-fashion-mnist-vae.py
@time: 2020/2/29 16:38
@desc: 12.5 VAE图片生成实战的代码
"""

import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

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


class VAE(keras.Model):
    # 自编码器模型类，包含了 Encoder 和 Decoder2 个子网络
    def __init__(self, z_dim):
        super(VAE, self).__init__()

        # Encoder 网络
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(z_dim)  # 均值输出
        self.fc3 = layers.Dense(z_dim)  # 方差输出
        # Decoder 网络
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    def encoder(self, x):
        # 获得编码器的均值和方差
        h = tf.nn.relu(self.fc1(x))
        # 均值向量
        mu = self.fc2(h)
        # 方差的 log 向量
        log_var = self.fc3(h)
        return mu, log_var

    def decoder(self, z):
        # 根据隐藏变量 z 生成图片数据
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)
        # 返回图片数据， 784 向量
        return out

    def reparameterize(self, mu, log_var):
        # reparameterize 技巧，从正态分布采样 epsion
        eps = tf.random.normal(tf.shape(log_var))
        # 计算标准差
        std = tf.exp(log_var) ** 0.5
        # reparameterize 技巧
        z = mu + std * eps
        return z

    def call(self, inputs, training=None):
        # 前向计算
        # 编码器[b, 784] => [b, z_dim], [b, z_dim]
        mu, log_var = self.encoder(inputs)
        # 采样 reparameterization trick
        z = self.reparameterize(mu, log_var)
        # 通过解码器生成
        x_hat = self.decoder(z)
        # 返回生成样本，及其均值与方差
        return x_hat, mu, log_var


def build_model(z_dim):
    # 创建网络对象
    model = VAE(z_dim)
    # 指定输入大小
    model.build(input_shape=(None, 784))
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
            x_rec_logits, mu, log_var = model(x)
            # 重建损失值计算
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]

            # 计算 KL 散度 N(mu, var) VS N(0, 1)
            # 公式参考： https://stats.stackexchange.com/questions/7440/kldivergence-between-two-univariate-gaussians
            kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
            kl_div = tf.reduce_sum(kl_div) / x.shape[0]
            # 合并误差项
            loss = rec_loss + 1. * kl_div

        # 自动求导
        grads = tape.gradient(loss, model.trainable_variables)
        # 自动更新
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            # 打印训练误差
            print("epoch=%s," % epoch, "step=%s," % step, "kl_div=%f," % float(kl_div), "rec_loss=%f" % float(rec_loss))

    return model


def evaluation(test_db, model, epoch, batchsz, z_dim):
    z = tf.random.normal((batchsz, z_dim))
    # 仅通过解码器生成图片
    logits = model.decoder(z)
    # 转换为像素范围
    x_hat = tf.sigmoid(logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, './vae_images/epoch_%d_sampled.png' % epoch)  # 保存生成图片

    # 重建图片，从测试集采样图片
    x = next(iter(test_db))
    # 打平并送入自编码器
    logits, _, _ = model(tf.reshape(x, [-1, 784]))
    # 将输出转换为像素值
    x_hat = tf.sigmoid(logits)
    # 恢复为 28x28,[b, 784] => [b, 28, 28]
    x_hat = tf.reshape(x_hat, [-1, 28, 28])
    # 输入的前 50 张+重建的前 50 张图片合并， [b, 28, 28] => [2b, 28, 28]
    x_concat = tf.concat([x[:50], x_hat[:50]], axis=0)
    x_concat = x_concat.numpy() * 255.  # 恢复为 0~255 范围
    x_concat = x_concat.astype(np.uint8)
    save_images(x_concat, './vae_images/epoch_%d_rec.png' % epoch)  # 保存重建图片


def main():
    z_dim = 10
    batchsz = 512
    lr = 1e-3

    train_db, test_db = load_dataset(batchsz)
    model = build_model(z_dim)
    # 创建优化器，并设置学习率
    optimizer = tf.optimizers.Adam(lr=lr)
    # 训练100个Epoch
    for epoch in range(100):
        model = train(train_db, model, optimizer, epoch)
        evaluation(test_db, model, epoch, batchsz, z_dim)


if __name__ == '__main__':
    main()
