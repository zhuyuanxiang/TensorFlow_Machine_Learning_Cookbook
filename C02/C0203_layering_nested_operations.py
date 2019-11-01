# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0203_layering_nested_operations.py
@Version    :   v0.1
@Time       :   2019-10-29 14:28
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0203，P21
@Desc       :   TensorFlow 进阶， TensorFlow 的嵌入 Layer
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


# 2.3  层化的嵌入式操作
# 创建数据和占位符
my_array = np.array([[1., 3., 5., 7., 9.],
                     [-2., 0., 2., 4., 6.],
                     [-6., -3., 0., 3., 6.]])
x_vals = np.array([my_array, my_array + 1])
print("x_vals = ")
print(x_vals)
# x_data = tf.placeholder(tf.float32, shape = (3, 5))
# x_data = tf.placeholder(tf.float32, shape = (3, none))
x_data = tf.placeholder(tf.float32)

# 　创建常量矩阵
m1 = tf.constant([[1.], [0.], [-1.], [2.], [4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

show_values("m1", m1)
show_values("m2", m2)
show_values("a1", a1)

# 表示成计算图
prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2, a1)

# 使用计算图计算
for x_val in x_vals:
    show_values("tf.matmul(x_data, m1)", prod1, feed_dict = {x_data: x_val})
    show_values("tf.matmul(tf.matmul(x_data, m1), m2)", prod2, feed_dict = {x_data: x_val})
    show_values("tf.add(tf.matmul(tf.matmul(x_data, m1), m2), a1)", add1, feed_dict = {x_data: x_val})

import winsound

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
