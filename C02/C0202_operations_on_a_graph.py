# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0202_operations_on_a_graph.py
@Version    :   v0.1
@Time       :   2019-10-29 14:21
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0202，P20
@Desc       :   TensorFlow 进阶，计算图中的操作
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


# 2.2 计算图
# 规范化的显示执行的效果
def show_values(var_name, variable, feed_dict = None):
    print('-' * 50)
    session = tf.Session()
    print("{} = {}".format(var_name, variable))
    print("session.run({}) = ".format(var_name))
    print(session.run(variable, feed_dict = feed_dict))
    pass


# Create data to feed in
x_vals = np.array([1., 3., 5., 7., 9.])
# placeholder() 占位符
x_data = tf.placeholder(tf.float32)
m_const = tf.constant(3.)
my_product = tf.multiply(x_data, m_const)
print("x_vals = ", x_vals)
for x_val in x_vals:
    show_values("tf.multiply(tf.placeholder(tf.float32), tf.constant(3.)) = ",
                my_product, feed_dict = {x_data: x_val})
    pass
print('-' * 50)
my_product = x_data + m_const
replace_dict = {x_data: 15.}
with sess.as_default():
    print("my_product.eval(feed_dict = replace_dict)",my_product.eval(feed_dict = replace_dict))
    pass
print("sess.run(my_product,feed_dict = replace_dict)", sess.run(my_product, feed_dict = replace_dict))

import winsound

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
