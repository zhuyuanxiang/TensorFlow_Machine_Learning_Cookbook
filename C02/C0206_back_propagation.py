# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0206_back_propagation.py
@Version    :   v0.1
@Time       :   2019-10-29 15:33
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0206，P30
@Desc       :   TensorFlow 进阶，TensorFlow 实现反向传播
"""
# Common imports
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import pandas as pd

# 设置数据显示的精确度为小数点后7位
np.set_printoptions(precision = 7, suppress = True, threshold = np.inf,
                    linewidth = 200)

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


# Regression Example:
# We will create sample data as follows:
# x-data: 100 random samples from a normal ~ N(1, 0.1)
# target: 100 values of the value 10.
# We will fit the model:
# x-data * A = target
# Theoretically, A = 10.
def regression_example():
    # Create data
    x_vals = np.random.normal(1, 0.1, 100)
    y_vals = 3 * x_vals + 0.5
    # y_vals = 10 * x_vals+0.5
    # y_vals = 10 + x_vals
    # y_vals = np.repeat(10., 100)
    plt.scatter(x_vals, y_vals)

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
    learning_rate = 0.02
    my_opt = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = my_opt.minimize(loss)

    # Run Loop
    for i in range(1000):
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]
        feed_dict = {x_data: rand_x, y_target: rand_y}
        sess.run(train_step, feed_dict = feed_dict)
        if (i + 1) % 25 == 0:
            print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
            print('Loss = ' + str(sess.run(loss, feed_dict = feed_dict)))


# Classification Example
# We will create sample data as follows:
# x-data: sample 50 random values from a normal = N(-1, 1)
#         + sample 50 random values from a normal = N(1, 1)
# target: 50 values of 0 + 50 values of 1.
#         These are essentially 100 values of the corresponding output index
# We will fit the binary classification model:
# If sigmoid(x+A) < 0.5 -> 0 else 1
# Theoretically, A should be -(mean1 + mean2)/2

def classification_example():
    # Create data
    x_vals = np.concatenate(
            (np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
    y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
    # plt.scatter(x_vals, y_vals)
    # plt.title("原始数据的散点图")

    x_data = tf.placeholder(shape = [1], dtype = tf.float32)
    y_target = tf.placeholder(shape = [1], dtype = tf.float32)

    # Create variable (one model parameter = A)
    # A 是模型参数
    A = tf.Variable(tf.random_normal(mean = 10, shape = [1]))

    # Add operation to graph
    # Want to create the operation sigmoid(x + A)
    # Note, the sigmoid() part is in the loss function
    my_output = tf.add(x_data, A)

    # Now we have to add another dimension to each (batch size of 1)
    # 因为指定的损失函数期望批量数据增加一个批量数的维度。
    my_output_expanded = tf.expand_dims(my_output, 0)
    y_target_expanded = tf.expand_dims(y_target, 0)

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Add classification loss (cross entropy)
    # 原代码把输入数据和目标数据放反了。
    # xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = my_output_expanded, logits = y_target_expanded)
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
        logits = my_output_expanded, labels = y_target_expanded)
    # ToDo: 未来尝试更换其他代价函数
    # xentropy = - tf.matmul(y_target_expanded, tf.log(my_output_expanded)) \
    #            - tf.matmul((1. - y_target_expanded), tf.log(1. - my_output_expanded))
    # xentropy = -(y_target * tf.log(my_output)) - ((1. - y_target) * tf.log(1. - my_output))
    # xentropy = tf.maximum(0., 1. - tf.multiply(y_target_expanded, my_output_expanded))
    # xentropy= tf.maximum(0., 1. - tf.multiply(y_vals, x_vals))

    # Create Optimizer
    learning_rate = 0.05
    my_opt = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = my_opt.minimize(xentropy)

    # Run loop
    for i in range(1400):
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]

        sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})
        if (i + 1) % 200 == 0:
            print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
            print('Loss = ' + str(sess.run(tf.sigmoid(xentropy), feed_dict = {
                    x_data: rand_x, y_target: rand_y
            })))

    # Evaluate Predictions
    predictions = []
    for i in range(len(x_vals)):
        x_val = [x_vals[i]]
        prediction = sess.run(tf.round(tf.sigmoid(my_output)),
                              feed_dict = {x_data: x_val})
        predictions.append(prediction[0])

    accuracy = sum(x == y for x, y in zip(predictions, y_vals)) / 100.
    print('Ending Accuracy = ' + str(np.round(accuracy, 2)))


if __name__ == "__main__":
    # regression_example()
    classification_example()
    if len(plt.get_fignums()) != 0:
        import winsound

        # 运行结束的提醒
        winsound.Beep(600, 500)
        plt.show()
    pass
