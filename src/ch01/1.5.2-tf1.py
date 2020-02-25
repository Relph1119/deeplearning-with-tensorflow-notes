#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 1.5.2-tf1.py
@time: 2020/2/14 15:30
@desc: 1.5.2 TensorFlow 1.x的代码
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 1.创建计算图阶段，此处代码需要使用 tf 1.x 版本运行
# 创建 2 个输入端子， 并指定类型和名字
a_ph = tf.placeholder(tf.float32, name='variable_a')
b_ph = tf.placeholder(tf.float32, name='variable_b')

# 创建输出端子的运算操作，并命名
c_op = tf.add(a_ph, b_ph, name='variable_c')


# 2.运行计算图阶段，此处代码需要使用 tf 1.x 版本运行
# 创建运行环境
sess = tf.InteractiveSession()
# 初始化步骤也需要作为操作运行
init = tf.global_variables_initializer()
sess.run(init) # 运行初始化操作，完成初始化
# 运行输出端子，需要给输入端子赋值
c_numpy = sess.run(c_op, feed_dict={a_ph: 2., b_ph: 4.})
# 运算完输出端子才能得到数值类型的 c_numpy
print('a+b=',c_numpy)