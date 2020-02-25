#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 6.8-auto-mpg-efficency.py
@time: 2020/2/24 12:43
@desc: 6.8 汽车油耗预测实战的代码
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_dataset():
    # 在线下载汽车效能数据集
    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

    # 效能（公里数每加仑），气缸数，排量，马力，重量
    # 加速度，型号年份，产地
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)

    dataset = raw_dataset.copy()
    return dataset


def preprocess_dataset(dataset):
    dataset = dataset.copy()
    # 统计空白数据,并清除
    dataset = dataset.dropna()

    # 处理类别型数据，其中origin列代表了类别1,2,3,分布代表产地：美国、欧洲、日本
    # 其弹出这一列
    origin = dataset.pop('Origin')
    # 根据origin列来写入新列
    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0

    # 切分为训练集和测试集
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    return train_dataset, test_dataset


def get_train_stats(train_dataset):
    # 查看训练集的输入X的统计数据
    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    return train_stats


def norm(x, train_stats):
    """
    标准化数据
    :param x:
    :param train_stats: get_train_stats(train_dataset)
    :return:
    """
    return (x - train_stats['mean']) / train_stats['std']


class Network(keras.Model):
    # 回归网络
    def __init__(self):
        super(Network, self).__init__()
        # 创建3个全连接层
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1)

    def call(self, inputs):
        # 依次通过3个全连接层
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def build_model():
    # 创建网络
    model = Network()
    model.build(input_shape=(4, 9))
    model.summary()
    return model


def train(model, train_db, optimizer, normed_test_data, test_labels):
    train_mae_losses = []
    test_mae_losses = []
    for epoch in range(200):
        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                out = model(x)
                loss = tf.reduce_mean(losses.MSE(y, out))
                mae_loss = tf.reduce_mean(losses.MAE(y, out))

            if step % 10 == 0:
                print(epoch, step, float(loss))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_mae_losses.append(float(mae_loss))
        out = model(tf.constant(normed_test_data.values))
        test_mae_losses.append(tf.reduce_mean(losses.MAE(test_labels, out)))

    return train_mae_losses, test_mae_losses


def plot(train_mae_losses, test_mae_losses):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.plot(train_mae_losses, label='Train')
    plt.plot(test_mae_losses, label='Test')
    plt.legend()
    # plt.ylim([0,10])
    plt.legend()
    plt.savefig('MEA变化曲线.svg')


def main():
    dataset = load_dataset()
    train_dataset, test_dataset = preprocess_dataset(dataset)
    # 统计数据
    sns_plot = sns.pairplot(train_dataset[["Cylinders", "Displacement", "Weight", "MPG"]],
                            diag_kind="kde")
    sns_plot.savefig("特征之间的两两分布.svg")

    train_stats = get_train_stats(train_dataset)

    # 移动MPG油耗效能这一列为真实标签Y
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    # 进行标准化
    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)

    model = build_model()
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_labels.values))
    train_db = train_db.shuffle(100).batch(32)

    train_mae_losses, test_mae_losses = train(model, train_db, optimizer, normed_test_data, test_labels)
    plot(train_mae_losses, test_mae_losses)


if __name__ == '__main__':
    main()
