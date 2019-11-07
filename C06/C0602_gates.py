# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0602_gates.py
@Version    :   v0.1
@Time       :   2019-11-05 9:59
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0602，P110
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 门函数
"""
import os
import sys
import sklearn
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

from C02.C0202_operations_on_a_graph import x_val
from tools import show_values

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 3, suppress = True, threshold = np.inf,
                    linewidth = 200)
# to make this notebook's output stable across runs
np.random.seed(42)

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


# Implementing Gates
# ----------------------------------
#
# This function shows how to implement
# various gates in Tensorflow
#
# One gate will be one operation with
# a variable and a placeholder.
# We will ask Tensorflow to change the
# variable based on our loss function

# ----------------------------------
# 创建乘法门函数
# Create a multiplication gate:
#   f(x) = a * x
#
#  a --
#      |
#      |---- (multiply) --> output
#  x --|
#
def multiplication_gate():
    # tf.constant()：函数可以不用，意思是一样的
    a = tf.Variable(tf.constant(4.))
    a = tf.Variable(4.)
    x_val = 5.
    x_data = tf.placeholder(dtype = tf.float32)

    multiplication = a * x_data

    # Declare the loss function as the difference between
    # the output and a target value, 50.
    loss = tf.square(multiplication - 50.)

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Declare optimizer
    my_opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = my_opt.minimize(loss)

    # Run loop across gate
    print('Optimizing a Multiplication Gate Output to 50.')
    for i in range(25):
        sess.run(train_step, feed_dict = {x_data: x_val})
        a_val = sess.run(a)
        mult_output = sess.run(multiplication, feed_dict = {x_data: x_val})
        show_values(loss, 'Loss', session = sess, feed_dict = {x_data: x_val})
        print("Step #", i, ')', "a * x = ", a_val, '*', x_val, '=', mult_output)
        pass
    pass


# ----------------------------------
# 创建嵌套操作的门函数
# Create a nested gate:
#   f(x) = a * x + b
#
#  a --
#      |
#      |-- (multiply)--
#  x --|              |
#                     |-- (add) --> output
#                 b --|
#
#
def nested_gate():
    a = tf.Variable(1.)
    b = tf.Variable(1.)
    x_val = 5.
    x_data = tf.placeholder(dtype = tf.float32)

    two_gate = a * x_data + b

    # Declare the loss function as the difference between
    # the output and a target value, 50.
    loss = tf.square(tf.subtract(two_gate, 50.))

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Declare optimizer
    my_opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = my_opt.minimize(loss)

    # Run loop across gate
    print('\nOptimizing Two Gate Output to 50.')
    for i in range(25):
        sess.run(train_step, feed_dict = {x_data: x_val})
        a_val, b_val = (sess.run(a), sess.run(b))
        two_gate_output = sess.run(two_gate, feed_dict = {x_data: x_val})
        show_values(loss, 'Loss', session = sess, feed_dict = {x_data: x_val})
        print("Step #", i, ')', "a * x + b =", a_val, '*', x_val, '+', b_val, '=', two_gate_output)
    pass


if __name__ == "__main__":
    multiplication_gate()
    # nested_gate()

    if len(plt.get_fignums()) != 0:
        import winsound

        # 运行结束的提醒
        winsound.Beep(600, 500)
        plt.show()
    pass
