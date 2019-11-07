# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0603_activation_functions.py
@Version    :   v0.1
@Time       :   2019-11-05 10:37
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0603，P113
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 在神经网络中应用门函数和激活函数
"""
import os
import sys
import sklearn
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tools import show_values

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 3, suppress = True, threshold = np.inf,
                    linewidth = 200)
# to make this notebook's output stable across runs
np.random.seed(42)
tf.set_random_seed(5)

# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# numpy 1.16.4 is required
assert np.__version__ in ["1.16.5", "1.16.4"]
# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 初始化默认的计算图
ops.reset_default_graph()
# Open graph session
sess = tf.Session()

# Combining Gates and Activation Functions
# ----------------------------------
#
# This function shows how to implement
# various gates with activation functions
# in Tensorflow
#
# This function is an extension of the
# prior gates, but with various activation
# functions.
batch_size = 50

a1 = tf.Variable(tf.random_normal(shape = [1, 1]))
b1 = tf.Variable(tf.random_uniform(shape = [1, 1]))
a2 = tf.Variable(tf.random_normal(shape = [1, 1]))
b2 = tf.Variable(tf.random_uniform(shape = [1, 1]))
x = np.random.normal(2, 0.1, 500)
x_data = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# x_data和a1初始化为单个值，这里进行矩阵乘法，是因为批处理数据
sigmoid_activation = tf.sigmoid(x_data @ a1 + b1)
relu_activation = tf.nn.relu(x_data @ a2 + b2)

# Declare the loss function as the difference between
# the output and a target value, 0.75.
sigmoid_loss = tf.reduce_mean(tf.square(sigmoid_activation - 0.75))
relu_loss = tf.reduce_mean(tf.square(relu_activation - 0.75))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
sigmoid_train_step = my_opt.minimize(sigmoid_loss)
relu_train_step = my_opt.minimize(relu_loss)

# Run loop across gate
print('\nOptimizing Sigmoid AND Relu Output to 0.75')
sigmoid_loss_vec, relu_loss_vec, sigmoid_activations, relu_activations = [], [], [], []
for i in range(750):
    rand_indices = np.random.choice(len(x), size = batch_size)
    x_vals = np.transpose([x[rand_indices]])
    sess.run(sigmoid_train_step, feed_dict = {x_data: x_vals})
    sess.run(relu_train_step, feed_dict = {x_data: x_vals})

    sigmoid_loss_vec.append(sess.run(sigmoid_loss, feed_dict = {x_data: x_vals}))
    relu_loss_vec.append(sess.run(relu_loss, feed_dict = {x_data: x_vals}))

    sigmoid_activations.append(np.mean(sess.run(sigmoid_activation, feed_dict = {x_data: x_vals})))
    relu_activations.append(np.mean(sess.run(relu_activation, feed_dict = {x_data: x_vals})))
    pass

# Plot the activation values
plt.figure()
plt.plot(sigmoid_activations, 'b-', label = "Sigmoid Activation")
plt.plot(relu_activations, 'r--', label = "Relu Activation")
plt.ylim([0, 1.0])
plt.title("图6-2：带有 sigmoid 和 ReLU 激励函数的神经网络输出结果对比")
plt.xlabel("迭代次数")
plt.ylabel("输出结果")
plt.legend(loc = "upper right")

# Plot the loss
plt.figure()
plt.plot(sigmoid_loss_vec, 'b-', label = "Sigmoid Losss")
plt.plot(relu_loss_vec, 'r--', label = "ReLU Loss")
plt.ylim(([0, 1.0]))
plt.title("图6-3：带有 sigmoid 和 ReLU 激励函数的神经网络的损失函数值对比")
plt.xlabel("迭代次数")
plt.ylabel("损失函数值")
plt.legend(loc = "upper right")

if len(plt.get_fignums()) != 0:
    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()
pass
