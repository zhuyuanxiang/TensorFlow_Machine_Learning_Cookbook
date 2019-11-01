# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0207_training.py
@Version    :   v0.1
@Time       :   2019-10-29 17:36
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0207，P34
@Desc       :   TensorFlow 进阶，TensorFlow 实现随机训练和批量训练
"""
# Common imports
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import pandas as pd

# 设置数据显示的精确度为小数点后7位
np.set_printoptions(precision = 7, suppress = True, threshold = np.inf, linewidth = 200)

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize = 14)
mpl.rc('xtick', labelsize = 12)
mpl.rc('ytick', labelsize = 12)

import os
import sys
import sklearn
import tensorflow as tf
from tensorflow.python.framework import ops

# 初始化默认的计算图
ops.reset_default_graph()
# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Open graph session
sess = tf.Session()


# 规范化的显示执行的效果
def show_values(var_name, variable, feed_dict = None, session = None):
    if session is None:
        session = tf.Session()
    print('-' * 50)
    print("{} = {}".format(var_name, variable))
    print("session.run({}) = ".format(var_name))
    result = session.run(variable, feed_dict = feed_dict)
    print(result)
    return result


# Stochastic Training:
number_title = "TensorFlow 随机训练"
print('\n', '-' * 5, number_title, '-' * 5)
# Create data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape = [1], dtype = tf.float32)
y_target = tf.placeholder(shape = [1], dtype = tf.float32)

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(shape = [1]))

# Add operation to graph
my_output = tf.multiply(x_data, A)

# Add L2 loss operation to graph
loss = tf.square(my_output - y_target)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

loss_stochastic = []
# Run Loop
for i in range(1000):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})
    if (i + 1) % 5 == 0:
        temp_loss = sess.run(loss, feed_dict = {x_data: rand_x, y_target: rand_y})
        if (i + 1) % 50 == 0:
            print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
            print('Loss = ' + str(temp_loss))
        loss_stochastic.append(temp_loss)
        pass
    pass
plt.plot(range(0, 1000, 5), loss_stochastic, 'b-', label = 'Stochastic Loss')

# Batch Training:
number_title = "TensorFlow 批量训练"
print('\n', '-' * 5, number_title, '-' * 5)

# Declare batch size
batch_size = 20

# Create data
x_vals = np.random.normal(1, 0.1, 100)
# y_vals = 3 * x_vals
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape = [None, 1], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Create variable ( one model parameter = A)
A = tf.Variable(tf.random_normal(shape = [1, 1]))

my_output = tf.matmul(x_data, A)

# Add L2 loss opeation to graph
# 因为批量处理，所以需要加上减少均值（tf.reduce_mean()），对这个批次数据结果的处理要求
loss = tf.reduce_mean(tf.square(y_target - my_output))
# loss = tf.reduce_mean(tf.abs(y_target - my_output))
# delta1 = tf.constant(0.25)
# loss = tf.reduce_mean(tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((y_target - my_output) / delta1)) - 1.))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

loss_batch = []
for i in range(1000):
    rand_index = np.random.choice(100, size = batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})
    if (i + 1) % 5 == 0:
        temp_loss = sess.run(loss, feed_dict = {x_data: rand_x, y_target: rand_y})
        # temp_loss = show_values("loss", loss, feed_dict = {x_data: rand_x, y_target: rand_y}, session = sess)
        if (i + 1) % 50 == 0:
            print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
            print('Loss = ' + str(temp_loss))
        loss_batch.append(temp_loss)
        pass
    pass
plt.plot(range(0, 1000, 5), loss_batch, 'r--', label = 'Batch Loss, size=20')
plt.legend(loc = 'upper right', prop = {'size': 11})

import winsound

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
