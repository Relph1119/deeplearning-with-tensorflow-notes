#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 13.8-faces-wgan-gp.py
@time: 2020/2/29 20:51
@desc: 13.8 WGAN-GP实战的代码
       将face文件夹放到ch13目录下，读取路径为/ch13/face/*.
"""

import glob
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

from src.ch13.dataset import make_anime_dataset
from src.ch13.gan import Generator, Discriminator

os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'

# 获取 GPU 设备列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置 GPU 为增长式占用
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 打印异常
        print(e)


def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(image_path)


def celoss_ones(logits):
    # 计算属于与标签为1的交叉熵
    y = tf.ones_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # 计算属于与便签为0的交叉熵
    y = tf.zeros_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def gradient_penalty(discriminator, batch_x, fake_image):
    # 梯度惩罚项计算函数
    batchsz = batch_x.shape[0]
    # 每个样本均随机采样 t,用于插值
    t = tf.random.uniform([batchsz, 1, 1, 1])
    # 自动扩展为 x 的形状， [b, 1, 1, 1] => [b, h, w, c]
    t = tf.broadcast_to(t, batch_x.shape)
    # 在真假图片之间做线性插值
    interplate = t * batch_x + (1 - t) * fake_image
    # 在梯度环境中计算 D 对插值样本的梯度
    with tf.GradientTape() as tape:
        tape.watch([interplate])  # 加入梯度观察列表
        d_interplote_logits = discriminator(interplate)
    grads = tape.gradient(d_interplote_logits, interplate)

    # 计算每个样本的梯度的范数:[b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)  # [b]
    # 计算梯度惩罚项
    gp = tf.reduce_mean((gp - 1.) ** 2)
    return gp


def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 计算 D 的损失函数
    fake_image = generator(batch_z, is_training)  # 假样本
    d_fake_logits = discriminator(fake_image, is_training)  # 假样本的输出
    d_real_logits = discriminator(batch_x, is_training)  # 真样本的输出
    # 计算梯度惩罚项
    gp = gradient_penalty(discriminator, batch_x, fake_image)
    # WGAN-GP D 损失函数的定义，这里并不是计算交叉熵，而是直接最大化正样本的输出
    # 最小化假样本的输出和梯度惩罚项
    loss = tf.reduce_mean(d_fake_logits) - tf.reduce_mean(d_real_logits) + 10. * gp
    return loss, gp


def g_loss_fn(generator, discriminator, batch_z, is_training):
    # 生成器的损失函数
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    # WGAN-GP G 损失函数，最大化假样本的输出值
    loss = - tf.reduce_mean(d_fake_logits)
    return loss


def main():
    tf.random.set_seed(3333)
    np.random.seed(3333)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    z_dim = 100  # 隐藏向量z的长度
    epochs = 3000000  # 训练步数
    batch_size = 64  # batch size
    learning_rate = 0.0002
    is_training = True

    # 获取数据集路径
    img_path = glob.glob(r'.\faces\*.jpg')
    print('images num:', len(img_path))
    # 构建数据集对象
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size, resize=64)
    print(dataset, img_shape)
    sample = next(iter(dataset))  # 采样
    print(sample.shape, tf.reduce_max(sample).numpy(),
          tf.reduce_min(sample).numpy())
    dataset = dataset.repeat(100)  # 重复循环
    db_iter = iter(dataset)

    discriminator, generator = build_network(z_dim)
    # 分别为生成器和判别器创建优化器
    g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    generator.load_weights('generator.ckpt')
    discriminator.load_weights('discriminator.ckpt')
    print('Loaded ckpt!!')

    d_losses, g_losses = [], []
    for epoch in range(epochs):  # 训练epochs次
        train(batch_size, d_losses, d_optimizer, db_iter, discriminator, epoch, g_losses, g_optimizer, generator,
              is_training, z_dim)


def train(batch_size, d_losses, d_optimizer, db_iter, discriminator, epoch, g_losses, g_optimizer, generator,
          is_training, z_dim):
    # 1. 训练判别器
    for _ in range(1):
        # 采样隐藏向量
        batch_z = tf.random.normal([batch_size, z_dim])
        batch_x = next(db_iter)  # 采样真实图片
        # 判别器前向计算
        with tf.GradientTape() as tape:
            d_loss, gp = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    # 2. 训练生成器
    # 采样隐藏向量
    batch_z = tf.random.normal([batch_size, z_dim])
    batch_x = next(db_iter)  # 采样真实图片
    # 生成器前向计算
    with tf.GradientTape() as tape:
        g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)

    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
    if epoch % 100 == 0:
        print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss))
        # 可视化
        z = tf.random.normal([100, z_dim])
        fake_image = generator(z, training=False)
        img_path = os.path.join('./gan_images', 'gan-%d.png' % epoch)
        save_result(fake_image.numpy(), 10, img_path, color_mode='P')

        d_losses.append(float(d_loss))
        g_losses.append(float(g_loss))

        if epoch % 10000 == 1:
            generator.save_weights('generator.ckpt')
            discriminator.save_weights('discriminator.ckpt')


def build_network(z_dim):
    # 创建生成器
    generator = Generator()
    generator.build(input_shape=(4, z_dim))
    # 创建判别器
    discriminator = Discriminator()
    discriminator.build(input_shape=(4, 64, 64, 3))
    return discriminator, generator


if __name__ == '__main__':
    main()
