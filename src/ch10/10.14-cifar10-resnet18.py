#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 10.14-cifar10-resnet18.py
@time: 2020/2/28 12:22
@desc: 10.14 CIFAR10与RESNET18实战的代码
"""

import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, datasets

from src.ch10.resnet import resnet18

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


def preprocess(x, y):
    # 将数据映射到-1~1
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    # 类型转换
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def load_dataset():
    # 加载数据集
    (x, y), (x_test, y_test) = datasets.cifar10.load_data()
    # 删除不必要的维度
    y = tf.squeeze(y, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    print(x.shape, y.shape, x_test.shape, y_test.shape)

    # 构建训练集
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    # 随机打散，预处理，批量化
    train_db = train_db.shuffle(1000).map(preprocess).batch(64)

    # 构建测试集
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # 随机打散，预处理，批量化
    test_db = test_db.map(preprocess).batch(64)
    # 采样一个样本
    sample = next(iter(train_db))
    print('sample:', sample[0].shape, sample[1].shape,
          tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))
    return train_db, test_db


def predict(model, test_db):
    total_num = 0
    total_correct = 0
    for x, y in test_db:
        logits = model(x)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)

        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_num += x.shape[0]
        total_correct += int(correct)

    acc = total_correct / total_num
    return acc


def train(epoch, model, optimizer, train_db):
    for step, (x, y) in enumerate(train_db):

        with tf.GradientTape() as tape:
            # [b, 32, 32, 3] => [b, 10],前向传播
            logits = model(x)
            # [b] => [b, 10],one-hot编码
            y_onehot = tf.one_hot(y, depth=10)
            # 计算交叉熵
            loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
        # 计算梯度信息
        grads = tape.gradient(loss, model.trainable_variables)
        # 更新网络参数
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 50 == 0:
            print(epoch, step, 'loss:', float(loss))

    return model


def main():
    epoch_num = 50

    train_db, test_db = load_dataset()
    # [b, 32, 32, 3] => [b, 1, 1, 512]
    # ResNet18网络
    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    # 统计网络参数
    model.summary()
    # 构建优化器
    optimizer = optimizers.Adam(lr=1e-4)

    accs = []
    # 训练epoch
    for epoch in range(epoch_num):
        model = train(epoch, model, optimizer, train_db)
        acc = predict(model, test_db)
        print(epoch, 'acc:', acc)
        accs.append(acc)

    x = range(epoch_num)
    plt.title("准确率")
    plt.plot(x, accs, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.savefig('cifar10-resnet18-accuracy.png')


if __name__ == '__main__':
    main()
