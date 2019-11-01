# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0104_placeholders.py
@Version    :   v0.1
@Time       :   2019-10-29 11:41
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0104，P6
@Desc       :   TensorFlow 基础，使用占位符和变量
"""
# Common imports
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import pandas as pd

# 设置数据显示的精确度为小数点后3位
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


# 规范化的显示执行的效果
def show_values(var_name, variable, feed_dict = None):
    print('-' * 50)
    session = tf.Session()
    print("{} = {}".format(var_name, variable))
    print("session.run({}) = ".format(var_name))
    print(session.run(variable, feed_dict = feed_dict))
    pass


# 变量必须在 session 中初始化，才可以使用
def declare_variable():
    number_title = "TensorFlow 使用变量"
    print('\n', '-' * 5, number_title, '-' * 5)

    # Declare a variable
    my_var = tf.Variable(tf.zeros([1, 20]))

    # Initialize operation
    initialize_op = tf.global_variables_initializer()

    # Run initialization of variable
    sess.run(initialize_op)
    print("my_var = ", my_var)
    print("sess.run(my_var)", sess.run(my_var))

    show_values("initialize_op", initialize_op)
    # 不同的 session ，不同的环境初始化。
    # show_values("my_var", my_var)

    print('-' * 50)
    first_var = tf.Variable(tf.zeros([2, 3]))
    print("first_var", first_var)
    print("first_var.initializer = ", first_var.initializer)
    print("sess.run(first_var.initializer) = ", sess.run(first_var.initializer))
    print("sess.run(first_var) = ", sess.run(first_var))
    # show_values("first_var",first_var)
    # show_values("first_var.initializer", first_var.initializer)

    print('-' * 50)
    second_var = tf.Variable(tf.ones_like(first_var))
    print("second_var", second_var)
    print("second_var.initializer", second_var.initializer)
    print("sess.run(second_var.initializer) = ", sess.run(second_var.initializer))
    print("sess.run(second_var) = ", sess.run(second_var))
    # show_values("second_var.initializer", second_var.initializer)


def declare_placeholder():
    x = tf.placeholder(tf.float32, shape = (4, 4))
    y = tf.identity(x)
    z = tf.matmul(y, x)

    x_vals = np.random.rand(4, 4)
    print("x_vals = ")
    print(x_vals)
    show_values("tf.identity(tf.placeholder(tf.float32, shape = (4,4)))", y, feed_dict = {x: x_vals})
    show_values("tf.matmul(x_vals,x_vals)",tf.matmul(x_vals,x_vals))
    show_values("tf.matmul(2,tf.placeholder(tf.float32, shape = (4,4)))", z, feed_dict = {x: x_vals})

    # merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter("/tmp/variable_logs", sess.graph_def)


if __name__ == "__main__":
    # declare_variable()

    declare_placeholder()
    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass
