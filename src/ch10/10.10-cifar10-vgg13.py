#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 10.10-cifar10-vgg13.py
@time: 2020/2/28 1:11
@desc: 10.10 CIFAR10与VGG13实战的代码
"""

import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

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
    return x, y


def build_network():
    # 先创建包含多网络层的列表
    conv_layers = [  # 5 units of conv + max pooling
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

    # 利用前面创建的层列表构建网络容器
    conv_net = Sequential(conv_layers)

    # 创建 3 层全连接层子网络
    fc_net = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(10, activation=None),
    ])

    # build2 个子网络，并打印网络参数信息
    conv_net.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[None, 512])
    conv_net.summary()
    fc_net.summary()

    return conv_net, fc_net


def train(conv_net, fc_net, train_db, optimizer, variables, epoch):
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            # [b, 32, 32, 3] => [b, 1, 1, 512]
            out = conv_net(x)
            # flatten, => [b, 512]
            out = tf.reshape(out, [-1, 512])
            # [b, 512] => [b, 10]
            logits = fc_net(out)
            # [b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10)
            # compute loss
            loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
            loss = tf.reduce_mean(loss)

        # 对所有参数求梯度
        grads = tape.gradient(loss, variables)
        # 自动更新
        optimizer.apply_gradients(zip(grads, variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))

    return conv_net, fc_net


def predict(conv_net, fc_net, test_db, epoch):
    total_num = 0
    total_correct = 0
    for x, y in test_db:
        out = conv_net(x)
        out = tf.reshape(out, [-1, 512])
        logits = fc_net(out)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)

        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_num += x.shape[0]
        total_correct += int(correct)

    acc = total_correct / total_num
    print(epoch, 'acc:', acc)
    return acc


def main():
    epoch_num = 50

    train_db, test_db = load_dataset()
    conv_net, fc_net = build_network()
    optimizer = optimizers.Adam(lr=1e-4)

    # 列表合并，合并 2 个子网络的参数
    variables = conv_net.trainable_variables + fc_net.trainable_variables

    accs = []
    for epoch in range(epoch_num):
        conv_net, fc_net = train(conv_net, fc_net, train_db, optimizer, variables, epoch)
        acc = predict(conv_net, fc_net, test_db, epoch)
        accs.append(acc)

    x = range(epoch_num)
    plt.title("准确率")
    plt.plot(x, accs, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.savefig('cifar10-vgg13-accuracy.png')


if __name__ == '__main__':
    main()
