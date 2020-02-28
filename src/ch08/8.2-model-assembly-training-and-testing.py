#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 8.2-model-assembly-training-and-testing.py
@time: 2020/2/25 13:37
@desc: 8.2 模型装配、训练与测试的示例代码，
       由于书中得到的准确值太差了，笔者调整了代码，使得准确率提高了97%
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Sequential, losses, optimizers, datasets

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False


def build_network():
    # 创建 5 层的全连接网络
    network = Sequential([layers.Flatten(input_shape=(28, 28)),
                          layers.Dense(256, activation='relu'),
                          layers.Dense(128, activation='relu'),
                          layers.Dense(64, activation='relu'),
                          layers.Dense(32, activation='relu'),
                          layers.Dense(10)])
    network.summary()

    # 模型装配
    # 采用 Adam 优化器，学习率为 0.01;采用交叉熵损失函数，包含 Softmax
    # kears sparse_categorical_crossentropy说明：
    # from_logits=False，output为经过softmax输出的概率值。
    # from_logits=True，output为经过网络直接输出的logits张量。
    network.compile(optimizer=optimizers.Adam(learning_rate=0.01),
                    loss=losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']  # 设置测量指标为准确率
                    )

    return network


def preprocess(x, y):
    # [b, 28, 28], [b]
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)

    return x, y


def load_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()

    batchsz = 512
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    train_db = train_db.shuffle(1000).map(preprocess).batch(batchsz)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.shuffle(1000).map(preprocess).batch(batchsz)

    return train_db, test_db


def train(network, train_db, test_db, epochs=5):
    # 指定训练集为 train_db，验证集为 val_db,训练 5 个 epochs，每 2 个 epoch 验证一次
    # 返回训练轨迹信息保存在 history 对象中
    history = network.fit(train_db, epochs=epochs, validation_data=test_db, validation_freq=2)

    print(history.history)
    return network, history


def test_one_data(network, test_db):
    # 加载一个 batch 的测试数据
    x, y = next(iter(test_db))
    print('predict x:', x.shape)  # 打印当前 batch 的形状
    out = network.predict(x)  # 模型预测，预测结果保存在 out 中
    print(out)


def test_model(network, test_db):
    # 模型测试，测试在 db_test 上的性能表现
    network.evaluate(test_db)


def main():
    epochs = 30
    train_db, test_db = load_dataset()
    network = build_network()
    network, history = train(network, train_db, test_db, epochs)
    test_one_data(network, test_db)
    test_model(network, test_db)

    x = range(epochs)

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1, 2, 1)
    # 绘制MES曲线
    plt.title("训练误差曲线")
    plt.plot(x, history.history['loss'], color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')

    # 绘制Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.title("准确率")
    plt.plot(x, history.history['accuracy'], color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.savefig('模型性能.png')
    plt.close()


if __name__ == '__main__':
    main()
