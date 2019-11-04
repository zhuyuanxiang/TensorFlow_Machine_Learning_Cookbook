# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0306_deming_regression.py
@Version    :   v0.1
@Time       :   2019-10-30 17:57
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0306，P55
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 Demming 回归算法
"""
# Common imports
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import pandas as pd

# 设置数据显示的精确度为小数点后7位
np.set_printoptions(precision = 7, suppress = True, threshold = np.inf, linewidth = 200)
# to make this notebook's output stable across runs
np.random.seed(42)

import os
import sys
import sklearn
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow as tf
from tensorflow.python.framework import ops

# To plot pretty figures
mpl.rc('axes', labelsize = 14)
mpl.rc('xtick', labelsize = 12)
mpl.rc('ytick', labelsize = 12)

# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 初始化默认的计算图
ops.reset_default_graph()
# Open graph session
sess = tf.Session()

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# Declare batch size
batch_size = 50
learning_rate = 0.1
iterations = 500

# Initialize placeholders
x_data = tf.placeholder(shape = [None, 1], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape = [1, 1]))
b = tf.Variable(tf.random_normal(shape = [1, 1]))

# Declare model operations
model_output = tf.add(tf.matmul(x_data, A), b)

# Declare Demming loss function
# d=abs(y-(Ax+b))/sqrt(A^2+1)
demming_numerator = tf.abs(tf.subtract(y_target, tf.add(tf.matmul(x_data, A), b)))
demming_denominator = tf.sqrt(tf.add(tf.square(A), 1))
loss = tf.reduce_mean(tf.truediv(demming_numerator, demming_denominator))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

# Training loop
loss_vec = []
for i in range(iterations):
    rand_index = np.random.choice(len(x_vals), size = batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    feed_dict = {x_data: rand_x, y_target: rand_y}
    sess.run(train_step, feed_dict = feed_dict)
    temp_loss = sess.run(loss, feed_dict = feed_dict)
    loss_vec.append(temp_loss)
    if i % 49 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))

# Get the optimal coefficients
[slope] = sess.run(A)
[y_intercept] = sess.run(b)

# Get best fit line
best_fit = []
for i in x_vals:
    best_fit.append(slope * i + y_intercept)

# Plot the result
plt.figure()
plt.plot(x_vals, y_vals, 'o', label = '数据')
plt.plot(x_vals, best_fit, 'r-', label = '最佳匹配线', linewidth = 3)
plt.legend(loc = 'upper left')
plt.legend(loc = 'upper left')
plt.xlabel('花瓣宽度')
plt.ylabel('花萼长度')
plt.suptitle("图3-9：iris 数据集上 Demming 回归算法的解")

# Plot loss over time
plt.figure()
plt.plot(loss_vec, 'r-', label = 'Demming Loss')
plt.xlabel("迭代次数")
plt.ylabel("Loss 损失函数")
plt.title("iris 数据线性回归的 Demming 损失函数\n学习率={}".format(learning_rate))

if len(plt.get_fignums()) != 0:
    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()
pass
