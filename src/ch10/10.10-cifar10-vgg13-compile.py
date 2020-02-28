#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 10.10-cifar10-vgg13.py
@time: 2020/2/28 1:11
@desc: 10.10 CIFAR10与VGG13实战的代码（装配版本）
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Sequential, losses, optimizers, datasets

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
sys.stdout = open('10.10-cifar10-vgg13-output-'+str(time.time())+'.log', mode='w', encoding='utf-8')

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


def load_dataset():
    # 在线下载，加载 CIFAR10 数据集
    (x, y), (x_test, y_test) = datasets.cifar10.load_data()
    # 删除 y 的一个维度， [b,1] => [b]
    y = tf.squeeze(y, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    # 打印训练集和测试集的形状
    print(x.shape, y.shape, x_test.shape, y_test.shape)
    # 构建训练集对象，随机打乱，预处理，批量化
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    train_db = train_db.shuffle(1000).map(preprocess).batch(128)
    # 构建测试集对象，预处理，批量化
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess).batch(64)
    # 从训练集中采样一个 Batch，并观察
    sample = next(iter(train_db))
    print('sample:', sample[0].shape, sample[1].shape, tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))
    return train_db, test_db


def preprocess(x, y):
    # [0~1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


def build_network():
    # 先创建包含多网络层的列表
    conv_layers = [
        # Conv-Conv-Pooling 单元 1
        # 64 个 3x3 卷积核, 输入输出同大小
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        # 高宽减半
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # Conv-Conv-Pooling 单元 2,输出通道提升至 128，高宽大小减半
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # Conv-Conv-Pooling 单元 3,输出通道提升至 256，高宽大小减半
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # Conv-Conv-Pooling 单元 4,输出通道提升至 512，高宽大小减半
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # Conv-Conv-Pooling 单元 5,输出通道提升至 512，高宽大小减半
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

    ]

    fc_layers = [
        layers.Flatten(),
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(10, activation=None),
    ]

    conv_layers.extend(fc_layers)
    network = Sequential(conv_layers)
    network.build(input_shape=[None, 32, 32, 3])
    network.summary()
    network.compile(optimizer=optimizers.Adam(lr=1e-4),
                    loss=losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']  # 设置测量指标为准确率
                    )

    return network


def train(network, train_db, test_db, epoch_num):
    # 指定训练集为train_db，验证集为test_db，每个epoch 验证一次
    # 返回训练轨迹信息保存在 history 对象中
    history = network.fit(train_db, epochs=epoch_num, validation_data=test_db)

    print(history.history)
    return network, history


def predict(network, test_db):
    # 模型测试，测试在 test_db 上的性能表现
    network.evaluate(test_db)


def main():
    epoch_num = 50

    train_db, test_db = load_dataset()
    network = build_network()
    network, history = train(network, train_db, test_db, epoch_num)
    predict(network, test_db)

    x = range(epoch_num)
    plt.title("准确率")
    plt.plot(x, history.history['accuracy'], color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.savefig('cifar10-vgg13-accuracy.png')


if __name__ == '__main__':
    main()
