# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0107_activation_functions.py
@Version    :   v0.1
@Time       :   2019-10-29 10:26
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0107，P12
@Desc       :   TensorFlow 基础, Implementing Activation Functions
"""
# Common imports
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import pandas as pd

# 设置数据显示的精确度为小数点后3位
from tools import show_values

np.set_printoptions(precision = 3, suppress = True, threshold = np.inf, linewidth = 200)

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

test_data = [-10., -3., -1., 0., 1., 3., 10.]

# 部分线性的非线性函数
# 整流线性单元（Rectifier Linear Unit，ReLU），非线性函数。
show_values(tf.nn.relu(test_data), "tf.nn.relu({})".format(test_data))
# ReLUMax6
show_values(tf.nn.relu6(test_data), "tf.nn.relu6({})".format(test_data))
# softplus函数，ReLU函数的平滑版，log(exp(x)+1)
show_values(tf.nn.softplus(test_data), "tf.nn.softplus({})".format(test_data))
# ELU激励函数（Exponential Linear Unit，ELU），
# 与softplus函数相似，只是输入无限小时，趋近于-1，而softplus函数趋近于0.
show_values(tf.nn.elu(test_data), "tf.nn.elu({})".format(test_data))

# 都是类似于Logistic函数
# sigmoid函数，Logistic函数，1/(1+exp(-x))
show_values(tf.nn.sigmoid(test_data), "tf.nn.sigmoid({})".format(test_data))
# 双曲正切函数（Hyper Tangent，tanh），((exp(x)-exp(-x))/(exp(x)+exp(-x))
show_values(tf.nn.tanh(test_data), "tf.nn.tanh({})".format(test_data))
# softsign函数，x/(abs(x)+1)
show_values(tf.nn.softsign(test_data), "tf.nn.softsign({})".format(test_data))

# X range
x_vals = np.linspace(start = -10., stop = 10., num = 100)

# ReLU activation
y_relu = sess.run(tf.nn.relu(x_vals))

# ReLU-6 activation
y_relu6 = sess.run(tf.nn.relu6(x_vals))

# Sigmoid activation
y_sigmoid = sess.run(tf.nn.sigmoid(x_vals))

# Hyper Tangent activation
y_tanh = sess.run(tf.nn.tanh(x_vals))

# Softsign activation
y_softsign = sess.run(tf.nn.softsign(x_vals))

# Softplus activation
y_softplus = sess.run(tf.nn.softplus(x_vals))

# Exponential linear activation
y_elu = sess.run(tf.nn.elu(x_vals))

# Plot the different functions
plt.figure()
plt.plot(x_vals, y_softplus, 'r--', label = 'Softplus', linewidth = 2)
plt.plot(x_vals, y_relu, 'b:', label = 'ReLU', linewidth = 2)
plt.plot(x_vals, y_relu6, 'g-.', label = 'ReLU6', linewidth = 2)
plt.plot(x_vals, y_elu, 'k-', label = 'ExpLU', linewidth = 0.5)
plt.ylim([-1.5, 7])
plt.legend(loc = 'upper left')
plt.title("图1-3：ReLU、ReLU6、softplus 和 ELU 激励函数")

plt.figure()
plt.plot(x_vals, y_sigmoid, 'r--', label = 'Sigmoid', linewidth = 2)
plt.plot(x_vals, y_tanh, 'b:', label = 'Tanh', linewidth = 2)
plt.plot(x_vals, y_softsign, 'g-.', label = 'Softsign', linewidth = 2)
plt.ylim([-2, 2])
plt.legend(loc = 'upper left')
plt.title("图1-3：sigmoid、softsign 和 tanh 激励函数")

if len(plt.get_fignums()) != 0:
    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()
pass
