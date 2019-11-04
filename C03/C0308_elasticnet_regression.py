# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0308_elasticnet_regression.py
@Version    :   v0.1
@Time       :   2019-10-31 8:56
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0308，P60
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 弹性网络回归算法（Lasso + Ridge）
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

# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
# 使用其他三个数据来拟合花萼长度
iris = datasets.load_iris()
x_vals = np.array([[x[1], x[2], x[3]] for x in iris.data])
# x_vals = np.array([[x[3]] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# Declare batch size
batch_size = 50
learning_rate = 0.001
iterations = 1500

# Initialize placeholders
x_data = tf.placeholder(shape = [None, 3], dtype = tf.float32)
# x_data = tf.placeholder(shape = [None, 1], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape = [3, 1]))
# A = tf.Variable(tf.random_normal(shape = [1, 1]))
b = tf.Variable(tf.random_normal(shape = [1, 1]))

# Declare model operations
model_output = tf.add(tf.matmul(x_data, A), b)

# Declare the elastic net loss function
# 具体参考《机器学习》中的数学公式
elastic_param1 = tf.constant(1.)
elastic_param2 = tf.constant(1.)
l1_a_loss = tf.reduce_mean(tf.abs(A))
l2_a_loss = tf.reduce_mean(tf.square(A))
e1_term = tf.multiply(elastic_param1, l1_a_loss)
e2_term = tf.multiply(elastic_param2, l2_a_loss)
elastic_term = tf.add(e1_term, e2_term)
l2_loss = tf.reduce_mean(tf.square(y_target - model_output))
loss = tf.expand_dims(tf.add(l2_loss, elastic_term), 0)

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
    rand_x = x_vals[rand_index]
    # 多维数据已经是不同类别数据一行，同类别数据成列
    # 一维数据则是同类数据一行，需要转为列数据，必须先把每个数据变成列表，然后再把所有数据转成列数据
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict = {x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss[0])
    if (i + 1) % 250 == 0:
        print('Step #' + str(i + 1))
        print(' A = ')
        print(sess.run(A))
        print("l1_a_loss =", sess.run(l1_a_loss))
        print("l2_a_loss =", sess.run(l2_a_loss))
        print(' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))
        pass
    pass

# Get the optimal coefficients
# sw (Sepal Width), pl (Petal Length), pw (Petal Width)
[[sw_slope], [pl_slope], [pw_slope]] = sess.run(A)
# [[sw_slope]] = sess.run(A)
[y_intercept] = sess.run(b)

# Get best fit line
best_fit = []
for i in x_vals:
    # best_fit.append([list(sw_slope * i[0] + y_intercept),
    #                  list(pl_slope * i[1] + y_intercept),
    #                  list(pw_slope * i[2] + y_intercept)])
    best_fit.append(list(np.array(
            [sw_slope * i[0] + y_intercept,
             pl_slope * i[1] + y_intercept,
             pw_slope * i[2] +y_intercept]).ravel()))
    pass

best_fit = np.array(best_fit)
# print(best_fit)
# print(x_vals)


plt.figure()
plt.plot(x_vals, y_vals, 'o', label = "数据点")
plt.plot(x_vals[:, 0], best_fit[:, 0], 'r-', label = "花萼宽度的最佳匹配线")
plt.plot(x_vals[:, 1], best_fit[:, 1], 'b-', label = "花瓣长度的最佳匹配线")
plt.plot(x_vals[:, 2], best_fit[:, 2], 'g-', label = "花瓣宽度的最佳匹配线")
plt.xlabel('其他特征')
plt.ylabel('花萼长度')
plt.title("图3-3：iris 数据集中的数据点 和 TensorFlow 拟合的直线")
plt.legend(loc = 'upper left')

# Plot loss over time
plt.figure()
plt.plot(loss_vec, 'r-', label = 'Ridge Loss')
plt.xlabel("迭代次数")
plt.ylabel("Loss 损失函数")
plt.legend(loc = 'upper right')
plt.title("图3-10：iris 数据线性回归的 Elasticnet 损失函数\n学习率={}".format(learning_rate))

if len(plt.get_fignums()) != 0:
    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()
pass
