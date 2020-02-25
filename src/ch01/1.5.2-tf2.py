#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 1.5.2-tf2.py
@time: 2020/2/14 15:34
@desc: 1.5.2 TensorFlow 2的代码
"""

import tensorflow as tf
# 此处代码需要使用 tf 2 版本运行
# 1.创建输入张量，并赋初始值
a = tf.constant(2.)
b = tf.constant(4.)
# 2.直接计算， 并打印结果

print('a+b=', a+b)